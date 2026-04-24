#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import shlex
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SOURCE_PATTERNS = [
    "corpus/{harness_id}/**/*",
    "audits/{harness_id}/**/window_corpus/**/*",
    "runs/{harness_id}/**/corpus/**/*",
    "runs/{harness_id}/**/queue/**/*",
]


def _run(
    cmd: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> str:
    p = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc: {p.returncode}\n"
            f"  stdout:\n{p.stdout[-4000:]}\n"
            f"  stderr:\n{p.stderr[-4000:]}\n"
        )
    return p.stdout


def _parse_lcov_summary(text: str) -> Dict[str, int]:
    out = {"LH": 0, "LF": 0, "FNH": 0, "FNF": 0, "BRH": 0, "BRF": 0}
    for line in text.splitlines():
        line = line.strip()
        for k in list(out.keys()):
            if line.startswith(k + ":"):
                try:
                    out[k] += int(line.split(":", 1)[1])
                except ValueError:
                    pass
    return out


def _cov_summary_lcov(
    *,
    llvm_cov: str,
    profdata: Path,
    primary_object: Path,
    extra_objects: Sequence[Path],
    ignore_filename_regex: Optional[str],
) -> Dict[str, int]:
    cmd = [
        llvm_cov,
        "export",
        "-summary-only",
        "-format=lcov",
        f"-instr-profile={profdata}",
        str(primary_object),
    ]
    for obj in extra_objects:
        cmd += ["-object", str(obj)]
    if ignore_filename_regex:
        cmd += [f"-ignore-filename-regex={ignore_filename_regex}"]
    text = _run(cmd)
    return _parse_lcov_summary(text)


def _chunks(xs: Sequence[str], n: int) -> Iterable[List[str]]:
    if n <= 0:
        n = len(xs) or 1
    for i in range(0, len(xs), n):
        yield list(xs[i : i + n])


def _merge_profiles_chunked(
    llvm_profdata: str,
    inputs: Sequence[Path],
    out_profdata: Path,
    *,
    chunk_size: int = 64,
    tmp_dir: Optional[Path] = None,
) -> None:
    """
    Merge many profile files (profraw or profdata) in chunks to avoid OOM.

    Strategy:
      - If input count <= chunk_size: merge directly.
      - Else: merge each chunk into an intermediate .profdata,
              then recursively merge those intermediates.
    """
    inputs = [Path(p) for p in inputs if Path(p).exists()]
    if not inputs:
        raise RuntimeError("No profile files found to merge")

    if len(inputs) <= chunk_size:
        cmd = [
            llvm_profdata,
            "merge",
            "-sparse",
            *[str(p) for p in inputs],
            "-o",
            str(out_profdata),
        ]
        _run(cmd)
        return

    if tmp_dir is None:
        tmp_dir = out_profdata.parent / f".merge_tmp_{out_profdata.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    intermediates: List[Path] = []
    for idx, chunk in enumerate(_chunks([str(p) for p in inputs], chunk_size)):
        chunk_out = tmp_dir / f"{out_profdata.stem}.chunk_{idx:04d}.profdata"
        cmd = [llvm_profdata, "merge", "-sparse", *chunk, "-o", str(chunk_out)]
        _run(cmd)
        intermediates.append(chunk_out)

    _merge_profiles_chunked(
        llvm_profdata,
        intermediates,
        out_profdata,
        chunk_size=chunk_size,
        tmp_dir=tmp_dir,
    )


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_harnesses_json(path: Path) -> Dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Path] = {}
    for row in data:
        hid = str(row["harness_id"])
        hpath = Path(row["harness_path"]).resolve()
        out[hid] = hpath
        stripped = hid.strip()
        if stripped and stripped not in out:
            out[stripped] = hpath
    return out


def _safe_harness_file_id(harness_id: str) -> str:
    s = harness_id.strip()
    s = s.replace("/", "__")
    s = re.sub(r"\s+", "_", s)
    return s or "unknown_harness"


def _collect_candidate_files(
    root: Path,
    harness_id: str,
    patterns: Sequence[str],
) -> List[Tuple[Path, str]]:
    seen_paths = set()
    out: List[Tuple[Path, str]] = []
    for pat in patterns:
        expanded = pat.format(harness_id=harness_id)
        for p in root.glob(expanded):
            if not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            out.append((p.resolve(), expanded))
    return out


def _collect_unique_inputs(
    root: Path,
    harness_id: str,
    patterns: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    candidates = _collect_candidate_files(root, harness_id, patterns)
    by_hash: Dict[str, Dict[str, object]] = {}
    source_counts: Dict[str, int] = {}

    for p, pat in candidates:
        source_counts[pat] = source_counts.get(pat, 0) + 1
        try:
            sha1 = _sha1_file(p)
        except Exception:
            continue

        try:
            rel = str(p.relative_to(root))
        except Exception:
            rel = str(p)

        if sha1 not in by_hash:
            by_hash[sha1] = {
                "sha1": sha1,
                "path": str(p),
                "relpath": rel,
                "size": int(p.stat().st_size),
                "sources": [pat],
            }
        else:
            srcs = by_hash[sha1].setdefault("sources", [])
            if pat not in srcs:
                srcs.append(pat)

    items = sorted(by_hash.values(), key=lambda x: (str(x["relpath"]), str(x["sha1"])))
    return items, source_counts


# ---------------- worker ----------------


def _identity_mutate(spec, cfg, fdp, **kwargs):
    return cfg


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            import shutil
            shutil.copy2(src, dst)
        except Exception:
            dst.write_bytes(src.read_bytes())


def _split_flag_text(flag_text: Optional[str]) -> List[str]:
    if not flag_text:
        return []
    try:
        return shlex.split(flag_text)
    except Exception:
        return str(flag_text).split()


def _run_batch_via_libfuzzer(
    mod,
    harness_path: Path,
    file_list: Sequence[str],
    flag_text: Optional[str],
) -> int:
    import atheris

    with tempfile.TemporaryDirectory(prefix=f"replay_{harness_path.stem}_") as td:
        corpus_dir = Path(td) / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        for i, path_str in enumerate(file_list):
            src = Path(path_str)
            suffix = src.suffix or ".bin"
            dst = corpus_dir / f"seed_{i:06d}{suffix}"
            _symlink_or_copy(src, dst)

        fuzz_argv = [
            str(harness_path),
            str(corpus_dir),
            f"-runs={len(file_list)}",
            "-print_final_stats=1",
        ]
        fuzz_argv.extend(_split_flag_text(flag_text))

        atheris.Setup(fuzz_argv, mod.TestOneInput)
        atheris.Fuzz()
        return 0


def _worker_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--disable-harness-mutation", action="store_true")
    ap.add_argument("--gc-every", type=int, default=16)
    ap.add_argument("--cov_replay_extra", default=None)
    ap.add_argument("--replay_extra", default=None)
    args = ap.parse_args(list(argv))

    flag_text = (
        args.cov_replay_extra
        or args.replay_extra
        or "-rss_limit_mb=8192 -malloc_limit_mb=8192"
    )

    harness_path = Path(args.harness).resolve()
    harness_dir = harness_path.parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    module_name = f"coverage_replay_{harness_path.stem}_{os.getpid()}_{int(time.time() * 1000)}"
    mod = _load_module_from_path(module_name, harness_path)

    if args.disable_harness_mutation:
        mod.mutate_cfg = _identity_mutate

    if not hasattr(mod, "TestOneInput"):
        raise RuntimeError(f"Harness has no TestOneInput: {harness_path}")

    file_list = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not file_list:
        return 0

    return _run_batch_via_libfuzzer(mod, harness_path, file_list, flag_text)


# ---------------- main ----------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        return _worker_main(argv[1:])

    ap = argparse.ArgumentParser(
        description=(
            "Replay all discovered corpora through harness TestOneInput "
            "without harness-side mutation and build final coverage profdata."
        )
    )
    ap.add_argument("--root", required=True, help="screen run root")
    ap.add_argument("--harnesses_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--llvm_cov", default="llvm-cov")
    ap.add_argument("--llvm_profdata", default="llvm-profdata")
    ap.add_argument("--primary_object", required=True)
    ap.add_argument("--extra_object", action="append", default=[])
    ap.add_argument("--ignore_filename_regex", default=None)
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Additional glob pattern under --root. Use {harness_id} placeholder.",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--merge_chunk_size", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--disable_harness_mutation", action="store_true", default=False)
    ap.add_argument("--keep_empty_harnesses", action="store_true")
    ap.add_argument(
        "--cov_replay_extra",
        default=None,
        help='libFuzzer flags forwarded to replay workers, e.g. "-rss_limit_mb=8192 -malloc_limit_mb=8192".',
    )
    ap.add_argument(
        "--replay_extra",
        default=None,
        help="Alias for --cov_replay_extra.",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    profraw_dir = out_dir / "profraw"
    manifests_dir = out_dir / "input_manifests"
    per_harness_dir = out_dir / "per_harness"

    out_dir.mkdir(parents=True, exist_ok=True)
    profraw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    per_harness_dir.mkdir(parents=True, exist_ok=True)

    harness_map = _load_harnesses_json(Path(args.harnesses_json).resolve())
    patterns = list(DEFAULT_SOURCE_PATTERNS) + list(args.pattern or [])

    all_manifest_summary: Dict[str, object] = {}
    all_profraws: List[Path] = []
    all_module_profdata: List[Path] = []

    raw_rows = json.loads(Path(args.harnesses_json).read_text(encoding="utf-8"))
    harness_ids = [str(r["harness_id"]) for r in raw_rows]

    for harness_id in harness_ids:
        harness_path = harness_map.get(harness_id) or harness_map.get(harness_id.strip())
        if harness_path is None:
            continue

        safe_hid = _safe_harness_file_id(harness_id)
        unique_items, source_counts = _collect_unique_inputs(root, harness_id, patterns)

        if not unique_items and not args.keep_empty_harnesses:
            all_manifest_summary[harness_id] = {
                "harness_path": str(harness_path),
                "candidate_sources": source_counts,
                "unique_inputs": 0,
                "skipped": True,
            }
            continue

        unique_paths = [str(x["path"]) for x in unique_items]
        harness_manifest = manifests_dir / f"{safe_hid}.json"
        harness_manifest.write_text(json.dumps(unique_items, indent=2), encoding="utf-8")

        harness_profraws: List[Path] = []
        batches = list(_chunks(unique_paths, args.batch_size))

        for batch_idx, batch_paths in enumerate(batches):
            batch_manifest = manifests_dir / f"{safe_hid}__batch_{batch_idx:04d}.paths.json"
            batch_manifest.write_text(json.dumps(batch_paths, indent=2), encoding="utf-8")

            profraw_pattern = profraw_dir / f"{safe_hid}__batch_{batch_idx:04d}_%p.profraw"
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = str(profraw_pattern)

            cmd = [
                args.python,
                str(Path(__file__).resolve()),
                "--worker",
                "--harness",
                str(harness_path),
                "--manifest",
                str(batch_manifest),
            ]
            if args.disable_harness_mutation:
                cmd.append("--disable-harness-mutation")

            flag_text = (
                args.cov_replay_extra
                or args.replay_extra
                or "-rss_limit_mb=8192 -malloc_limit_mb=8192"
            )
            if flag_text:
                cmd += ["--cov_replay_extra", flag_text]

            _run(cmd, env=env, timeout=args.timeout)

            batch_profraws = sorted(profraw_dir.glob(f"{safe_hid}__batch_{batch_idx:04d}_*.profraw"))
            harness_profraws.extend(batch_profraws)
            all_profraws.extend(batch_profraws)

            if batch_idx % 8 == 0:
                gc.collect()

        per_profdata = per_harness_dir / f"{safe_hid}.profdata"
        per_summary: Dict[str, int] | None = None

        if harness_profraws:
            _merge_profiles_chunked(
                args.llvm_profdata,
                harness_profraws,
                per_profdata,
                chunk_size=args.merge_chunk_size,
            )
            all_module_profdata.append(per_profdata)

            per_summary = _cov_summary_lcov(
                llvm_cov=args.llvm_cov,
                profdata=per_profdata,
                primary_object=Path(args.primary_object).resolve(),
                extra_objects=[Path(x).resolve() for x in args.extra_object],
                ignore_filename_regex=args.ignore_filename_regex,
            )

            # 删除该 harness 的所有 profraw
            for p in harness_profraws:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

        all_manifest_summary[harness_id] = {
            "harness_path": str(harness_path),
            "candidate_sources": source_counts,
            "unique_inputs": len(unique_items),
            "batches": len(batches),
            "profraw_count": len(harness_profraws),
            "per_harness_profdata": str(per_profdata) if harness_profraws else None,
            "per_harness_totals": per_summary,
        }

    if not all_module_profdata:
        raise RuntimeError("No per-harness profdata files were produced; nothing was replayed")

    final_profdata = out_dir / "all_corpora.profdata"
    _merge_profiles_chunked(
        args.llvm_profdata,
        all_module_profdata,
        final_profdata,
        chunk_size=args.merge_chunk_size,
    )

    final_totals = _cov_summary_lcov(
        llvm_cov=args.llvm_cov,
        profdata=final_profdata,
        primary_object=Path(args.primary_object).resolve(),
        extra_objects=[Path(x).resolve() for x in args.extra_object],
        ignore_filename_regex=args.ignore_filename_regex,
    )

    summary = {
        "root": str(root),
        "harnesses_json": str(Path(args.harnesses_json).resolve()),
        "final_profdata": str(final_profdata),
        "final_totals": final_totals,
        "profraw_count": len(all_profraws),
        "per_harness_profdata_count": len(all_module_profdata),
        "disable_harness_mutation": bool(args.disable_harness_mutation),
        "cov_replay_extra": (
            args.cov_replay_extra
            or args.replay_extra
            or "-rss_limit_mb=8192 -malloc_limit_mb=8192"
        ),
        "merge_chunk_size": args.merge_chunk_size,
        "source_patterns": patterns,
        "per_harness": all_manifest_summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())