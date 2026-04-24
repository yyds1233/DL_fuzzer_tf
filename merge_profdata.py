#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence


def run(cmd: Sequence[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc: {p.returncode}\n"
            f"  stdout:\n{p.stdout[-4000:]}\n"
            f"  stderr:\n{p.stderr[-4000:]}\n"
        )


def chunked(xs: List[Path], n: int) -> List[List[Path]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def merge_once(llvm_profdata: str, inputs: List[Path], output: Path) -> None:
    cmd = [llvm_profdata, "merge", "-sparse", *[str(p) for p in inputs], "-o", str(output)]
    run(cmd)


def merge_profraw_in_batches(
    llvm_profdata: str,
    inputs: List[Path],
    output: Path,
    chunk_size: int,
    work_dir: Path,
) -> None:
    inputs = [p.resolve() for p in inputs if p.exists()]
    if not inputs:
        raise RuntimeError("No input .profraw files found")

    work_dir.mkdir(parents=True, exist_ok=True)

    round_idx = 0
    current = inputs

    while len(current) > 1:
        next_round: List[Path] = []
        groups = chunked(current, chunk_size)

        print(f"[merge] round={round_idx} inputs={len(current)} groups={len(groups)}")

        for i, group in enumerate(groups):
            out = work_dir / f"round_{round_idx:03d}_chunk_{i:04d}.profdata"
            print(f"  - chunk {i+1}/{len(groups)}: {len(group)} files -> {out.name}")
            merge_once(llvm_profdata, group, out)
            next_round.append(out)

        current = next_round
        round_idx += 1

    if current[0].resolve() != output.resolve():
        shutil.copy2(current[0], output)

    print(f"[done] final merged profdata: {output}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-merge many .profraw files into one .profdata")
    ap.add_argument("--input_dir", required=True, help="递归查找其中的 .profraw")
    ap.add_argument("--output", required=True, help="最终输出的 .profdata")
    ap.add_argument("--llvm_profdata", default="llvm-profdata")
    ap.add_argument("--chunk_size", type=int, default=32, help="每批 merge 的文件数")
    ap.add_argument("--pattern", default="*.profraw", help="默认匹配 *.profraw")
    ap.add_argument("--exclude", action="append", default=[], help="排除包含这些子串的路径，可重复传入")
    ap.add_argument("--work_dir", default=None, help="中间 chunk 文件目录")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output.parent / ".merge_tmp_profraw"

    files = sorted(p for p in input_dir.rglob(args.pattern) if p.is_file())
    if args.exclude:
        files = [p for p in files if not any(x in str(p) for x in args.exclude)]

    print(f"[scan] found {len(files)} profraw files under {input_dir}")
    if not files:
        raise RuntimeError("No .profraw files matched")

    output.parent.mkdir(parents=True, exist_ok=True)
    merge_profraw_in_batches(
        llvm_profdata=args.llvm_profdata,
        inputs=files,
        output=output,
        chunk_size=max(2, args.chunk_size),
        work_dir=work_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())