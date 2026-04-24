#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

TF_PYTHON = "/root/tf_cov/bin/python3"


@dataclass
class ScriptResult:
    script: str
    ok: bool
    exit_code: Optional[int]
    duration_sec: float
    raw_dir: str
    raw_files: List[str]
    emitted_profile: bool
    stdout_log: str
    stderr_log: str
    done_marker: str
    error: Optional[str] = None
    is_timeout: bool = False
    is_skipped: bool = False            # 已有 done 标记，直接跳过
    reused_existing_raw: bool = False   # 沿用未合并的旧 .profraw，不重新执行


class ProgressTracker:
    """用于多线程环境下安全地追踪进度"""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.success = 0
        self.failure = 0
        self.timeout = 0
        self.already_done = 0
        self.reused_raw = 0
        self._lock = threading.Lock()

    def update(
        self,
        ok: bool,
        is_timeout: bool,
        is_skipped: bool = False,
        reused_existing_raw: bool = False,
    ) -> None:
        with self._lock:
            self.completed += 1
            if is_skipped:
                self.already_done += 1
            elif reused_existing_raw:
                self.reused_raw += 1
            elif is_timeout:
                self.timeout += 1
            elif ok:
                self.success += 1
            else:
                self.failure += 1

            percent = (self.completed / self.total) * 100 if self.total else 100.0
            sys.stdout.write(
                f"\r[PROGRESS] {self.completed}/{self.total} ({percent:.1f}%) | "
                f"Success: {self.success} | Already_Done: {self.already_done} | "
                f"Reused_Raw: {self.reused_raw} | Fail: {self.failure} | Timeout: {self.timeout}"
            )
            sys.stdout.flush()


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def find_py_files(root: Path, name_glob: str, exclude_regex: Optional[str]) -> List[Path]:
    regex = re.compile(exclude_regex) if exclude_regex else None
    files: List[Path] = []
    for p in root.rglob(name_glob):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if regex and regex.search(rel):
            continue
        files.append(p)
    files.sort()
    return files


def group_by_parent(files: List[Path]) -> Dict[Path, List[Path]]:
    groups: Dict[Path, List[Path]] = {}
    for f in files:
        groups.setdefault(f.parent, []).append(f)
    return groups


def ensure_cmd_exists(cmd: str) -> str:
    found = shutil.which(cmd)
    if found:
        return found
    raise FileNotFoundError(f"找不到命令: {cmd}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_subprocess(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: Optional[int],
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def split_batches(items: List[Path], batch_size: int) -> List[List[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def make_tag(script: Path, root_dir: Path) -> str:
    rel_str = script.relative_to(root_dir).as_posix()
    return f"{script.stem}__{sha1_short(rel_str)}"


def get_task_raw_dir(script: Path, root_dir: Path, raw_root: Path) -> Path:
    rel = script.relative_to(root_dir)
    return raw_root / rel.parent / make_tag(script, root_dir)


def get_done_marker(script: Path, root_dir: Path, done_root: Path) -> Path:
    rel = script.relative_to(root_dir)
    return done_root / rel.parent / f"{make_tag(script, root_dir)}.done.json"


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def write_done_marker(result: ScriptResult, batch_index: int, batch_profdata: Path) -> None:
    payload = {
        "script": result.script,
        "batch_index": batch_index,
        "batch_profdata": str(batch_profdata),
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    write_text(Path(result.done_marker), json.dumps(payload, ensure_ascii=False, indent=2))


# 这里的 input_files 可以同时包含 .profraw 和 .profdata
# 这样可以把“旧 batch.profdata + 新 raw”继续合并回同一个 batch.profdata
def run_profile_merge(
    llvm_profdata: str,
    input_files: List[str],
    output_profdata: Path,
    manifest_file: Path,
) -> subprocess.CompletedProcess:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text("\n".join(input_files) + "\n", encoding="utf-8")
    output_profdata.parent.mkdir(parents=True, exist_ok=True)
    cmd = [llvm_profdata, "merge", "-sparse", f"--input-files={manifest_file}", f"-o={output_profdata}"]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )


def run_one_script(
    script: Path,
    root_dir: Path,
    raw_root: Path,
    done_root: Path,
    log_root: Path,
    python_exe: str,
    timeout: Optional[int],
    slot_sem: threading.BoundedSemaphore,
    tracker: ProgressTracker,
) -> ScriptResult:
    tag = make_tag(script, root_dir)
    task_raw_dir = get_task_raw_dir(script, root_dir, raw_root)
    done_marker = get_done_marker(script, root_dir, done_root)
    stdout_log = log_root / f"{tag}.stdout.log"
    stderr_log = log_root / f"{tag}.stderr.log"

    # 已经完成并且已经合并进某个 batch.profdata，直接跳过
    if done_marker.exists():
        tracker.update(ok=True, is_timeout=False, is_skipped=True)
        return ScriptResult(
            script=str(script),
            ok=True,
            exit_code=0,
            duration_sec=0.0,
            raw_dir=str(task_raw_dir),
            raw_files=[],
            emitted_profile=True,
            stdout_log=str(stdout_log) if stdout_log.exists() else "",
            stderr_log=str(stderr_log) if stderr_log.exists() else "",
            done_marker=str(done_marker),
            is_skipped=True,
        )

    # 之前已经跑出来 raw 但还没来得及 merge，直接复用，不重跑
    existing_raw_files = sorted(str(p) for p in task_raw_dir.rglob("*.profraw")) if task_raw_dir.exists() else []
    if existing_raw_files:
        tracker.update(ok=True, is_timeout=False, reused_existing_raw=True)
        return ScriptResult(
            script=str(script),
            ok=True,
            exit_code=0,
            duration_sec=0.0,
            raw_dir=str(task_raw_dir),
            raw_files=existing_raw_files,
            emitted_profile=True,
            stdout_log=str(stdout_log) if stdout_log.exists() else "",
            stderr_log=str(stderr_log) if stderr_log.exists() else "",
            done_marker=str(done_marker),
            reused_existing_raw=True,
        )

    task_raw_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = str(task_raw_dir / "run_%p_%m.profraw")
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [python_exe, "-u", script.name]
    t0 = time.time()
    exit_code, ok, is_timeout, error, out, err = None, False, False, None, "", ""

    slot_sem.acquire()
    try:
        try:
            cp = run_subprocess(cmd=cmd, cwd=script.parent, env=env, timeout=timeout)
            exit_code, out, err, ok = cp.returncode, cp.stdout, cp.stderr, (cp.returncode == 0)
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", "replace") if e.stdout else "")
            err = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", "replace") if e.stderr else "")
            err += f"\n[TIMEOUT] 超过 {timeout}s 跳过\n"
            ok, is_timeout, error = False, True, f"timeout({timeout}s)"
        except Exception as e:
            ok, error, err = False, repr(e), f"[EXCEPTION] {repr(e)}\n"
    finally:
        slot_sem.release()

    duration = time.time() - t0
    write_text(stdout_log, out)
    write_text(stderr_log, err)

    raw_files = sorted(str(p) for p in task_raw_dir.rglob("*.profraw"))
    emitted_profile = len(raw_files) > 0
    if error is None and ok and not emitted_profile:
        error = "script exit 0 but no .profraw emitted"

    tracker.update(ok, is_timeout, is_skipped=False)

    return ScriptResult(
        script=str(script),
        ok=ok,
        exit_code=exit_code,
        duration_sec=round(duration, 4),
        raw_dir=str(task_raw_dir),
        raw_files=raw_files,
        emitted_profile=emitted_profile,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        done_marker=str(done_marker),
        error=error,
        is_timeout=is_timeout,
        is_skipped=False,
        reused_existing_raw=False,
    )


def process_one_directory(
    dir_path: Path,
    scripts: List[Path],
    root_dir: Path,
    raw_root: Path,
    done_root: Path,
    log_root: Path,
    python_exe: str,
    timeout: Optional[int],
    file_workers: int,
    slot_sem: threading.BoundedSemaphore,
    tracker: ProgressTracker,
) -> List[ScriptResult]:
    _ = dir_path
    local_workers = max(1, min(file_workers, len(scripts)))
    results: List[ScriptResult] = []

    with cf.ThreadPoolExecutor(max_workers=local_workers) as ex:
        futs = [
            ex.submit(
                run_one_script,
                script,
                root_dir,
                raw_root,
                done_root,
                log_root,
                python_exe,
                timeout,
                slot_sem,
                tracker,
            )
            for script in scripts
        ]
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda x: x.script)
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="支持分批 merge / 清理 profraw / 断点重跑的 LLVM Coverage 采集工具")
    ap.add_argument("root_dir", help="待递归扫描的根目录")
    ap.add_argument("--out-dir", default="/root/replay_cov_out", help="输出目录")
    ap.add_argument("--python", default=TF_PYTHON, help="Python 解释器路径")
    ap.add_argument("--llvm-profdata", default="llvm-profdata", help="llvm-profdata 路径")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="总并发槽位")
    ap.add_argument("--dir-workers", type=int, default=4, help="目录级并发")
    ap.add_argument("--file-workers", type=int, default=4, help="文件级并发")
    ap.add_argument("--timeout", type=int, default=200, help="单脚本超时秒数")
    ap.add_argument("--batch-size", type=int, default=100, help="每批处理多少个脚本后立刻 merge 并清理 raw")
    ap.add_argument("--name-glob", default="*.py", help="匹配模式")
    ap.add_argument("--exclude-regex", default=None, help="排除正则")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[ERROR] root_dir 不存在: {root_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    raw_root = out_dir / "raw"
    log_root = out_dir / "logs"
    report_root = out_dir / "reports"
    tmp_root = out_dir / "tmp"
    done_root = out_dir / "done"
    batch_root = out_dir / "batches"
    batch_report_root = report_root / "batches"
    for d in (raw_root, log_root, report_root, tmp_root, done_root, batch_root, batch_report_root):
        d.mkdir(parents=True, exist_ok=True)

    python_exe = TF_PYTHON
    if not Path(python_exe).exists():
        raise FileNotFoundError(f"Python 不存在: {python_exe}")

    llvm_profdata = args.llvm_profdata
    if os.path.sep not in llvm_profdata:
        llvm_profdata = ensure_cmd_exists(llvm_profdata)
    else:
        llvm_profdata = str(Path(llvm_profdata).resolve())
        if not Path(llvm_profdata).exists():
            raise FileNotFoundError(f"llvm-profdata 不存在: {llvm_profdata}")

    py_files = find_py_files(root_dir, args.name_glob, args.exclude_regex)
    if not py_files:
        print("[WARN] 未找到匹配文件")
        return 0

    batches = split_batches(py_files, args.batch_size)
    tracker = ProgressTracker(total=len(py_files))
    slot_sem = threading.BoundedSemaphore(value=max(1, args.workers))
    all_results: List[ScriptResult] = []
    merged_batches_this_run = 0
    reused_existing_batch_profdata = 0
    merged_profile_inputs_this_run = 0

    t0 = time.time()
    total_batches = len(batches)

    for batch_index, batch_scripts in enumerate(batches, start=1):
        print(f"\n[INFO] 开始处理批次 {batch_index}/{total_batches}，脚本数: {len(batch_scripts)}")
        grouped = group_by_parent(batch_scripts)
        dir_items = sorted(grouped.items(), key=lambda x: str(x[0]))
        batch_results: List[ScriptResult] = []

        with cf.ThreadPoolExecutor(max_workers=max(1, args.dir_workers)) as outer_ex:
            futs = [
                outer_ex.submit(
                    process_one_directory,
                    dp,
                    ss,
                    root_dir,
                    raw_root,
                    done_root,
                    log_root,
                    python_exe,
                    args.timeout,
                    args.file_workers,
                    slot_sem,
                    tracker,
                )
                for dp, ss in dir_items
            ]
            for fut in cf.as_completed(futs):
                batch_results.extend(fut.result())

        batch_results.sort(key=lambda x: x.script)
        all_results.extend(batch_results)

        batch_raw_files: List[str] = []
        batch_merge_candidates: List[ScriptResult] = []
        for r in batch_results:
            if r.is_skipped:
                continue
            if r.ok and r.emitted_profile:
                batch_merge_candidates.append(r)
                batch_raw_files.extend(r.raw_files)

        batch_profdata = batch_root / f"batch_{batch_index:06d}.profdata"
        batch_manifest = tmp_root / f"batch_{batch_index:06d}_inputs.txt"
        batch_report = batch_report_root / f"batch_{batch_index:06d}.json"

        batch_payload = {
            "batch_index": batch_index,
            "script_count": len(batch_scripts),
            "raw_input_files": len(batch_raw_files),
            "had_existing_batch_profdata": batch_profdata.exists(),
            "scripts": [r.script for r in batch_results],
            "status": "not_merged",
        }

        if batch_profdata.exists() and batch_raw_files:
            merge_inputs = [str(batch_profdata)] + batch_raw_files
        else:
            merge_inputs = batch_raw_files[:]

        if merge_inputs:
            tmp_output = batch_profdata.with_suffix(".profdata.tmp")
            remove_path(tmp_output)
            merge_cp = run_profile_merge(llvm_profdata, merge_inputs, tmp_output, batch_manifest)
            batch_payload.update(
                {
                    "status": "merged" if merge_cp.returncode == 0 else "merge_failed",
                    "merge_stdout": merge_cp.stdout,
                    "merge_stderr": merge_cp.stderr,
                    "merge_input_count": len(merge_inputs),
                    "merge_inputs": merge_inputs,
                    "output_profdata": str(batch_profdata),
                }
            )
            write_text(batch_report, json.dumps(batch_payload, ensure_ascii=False, indent=2))

            if merge_cp.returncode != 0:
                remove_path(tmp_output)
                print(f"[ERROR] 批次 {batch_index} 合并失败: {merge_cp.stderr}", file=sys.stderr)
                return 4

            tmp_output.replace(batch_profdata)
            merged_batches_this_run += 1
            merged_profile_inputs_this_run += len(merge_inputs)

            for r in batch_merge_candidates:
                write_done_marker(r, batch_index, batch_profdata)
                remove_path(Path(r.raw_dir))

            print(
                f"[INFO] 批次 {batch_index} 合并成功 -> {batch_profdata} | "
                f"本次输入 {len(merge_inputs)} 个 profile | 已清理 {len(batch_merge_candidates)} 个脚本的 raw"
            )
        else:
            if batch_profdata.exists():
                reused_existing_batch_profdata += 1
                batch_payload.update(
                    {
                        "status": "reused_existing_batch_profdata",
                        "output_profdata": str(batch_profdata),
                    }
                )
                write_text(batch_report, json.dumps(batch_payload, ensure_ascii=False, indent=2))
                print(f"[INFO] 批次 {batch_index} 没有新增 raw，复用已有 {batch_profdata}")
            else:
                batch_payload.update({"status": "no_profile_generated"})
                write_text(batch_report, json.dumps(batch_payload, ensure_ascii=False, indent=2))
                print(f"[INFO] 批次 {batch_index} 本次没有可合并 profile（可能全失败/全超时/未产出 coverage）")

    print("\n")
    all_results.sort(key=lambda x: x.script)

    cnt_ok = 0
    cnt_fail = 0
    cnt_timeout = 0
    cnt_skipped = 0
    cnt_reused_raw = 0
    pending_raw_files = 0

    for r in all_results:
        pending_raw_files += len(r.raw_files)
        if r.is_skipped:
            cnt_skipped += 1
        elif r.reused_existing_raw:
            cnt_reused_raw += 1
        elif r.is_timeout:
            cnt_timeout += 1
        elif r.ok:
            cnt_ok += 1
        else:
            cnt_fail += 1

    elapsed = time.time() - t0
    batch_profdata_files = sorted(str(p) for p in batch_root.glob("batch_*.profdata"))

    result_json = report_root / "run_results.json"
    payload = {
        "summary": {
            "total": len(all_results),
            "new_success": cnt_ok,
            "already_done_skipped": cnt_skipped,
            "reused_existing_raw": cnt_reused_raw,
            "fail": cnt_fail,
            "timeout": cnt_timeout,
            "raw_profiles_seen_before_cleanup": pending_raw_files,
            "batch_profdata_files": len(batch_profdata_files),
            "merged_batches_this_run": merged_batches_this_run,
            "reused_existing_batch_profdata": reused_existing_batch_profdata,
            "merged_profile_inputs_this_run": merged_profile_inputs_this_run,
            "elapsed_sec": round(elapsed, 2),
        },
        "results": [asdict(r) for r in all_results],
    }
    write_text(result_json, json.dumps(payload, ensure_ascii=False, indent=2))

    print("--- 最终汇总 ---")
    print(f"[INFO] 总脚本数: {len(all_results)}")
    print(f"[INFO] 断点跳过 (已有 done): {cnt_skipped}")
    print(f"[INFO] 复用旧 raw (未重跑): {cnt_reused_raw}")
    print(f"[INFO] 本次新成功: {cnt_ok}")
    print(f"[INFO] 失败: {cnt_fail}")
    print(f"[INFO] 超时: {cnt_timeout}")
    print(f"[INFO] 本次新合并批次: {merged_batches_this_run}")
    print(f"[INFO] 复用已有 batch.profdata: {reused_existing_batch_profdata}")
    print(f"[INFO] 当前 batch.profdata 文件数: {len(batch_profdata_files)}")

    if not batch_profdata_files:
        print("[ERROR] 没有可合并的 batch .profdata", file=sys.stderr)
        return 3

    total_profdata = out_dir / "merged_total.profdata"
    total_manifest = tmp_root / "all_batch_profdata_manifest.txt"
    merge_cp = run_profile_merge(llvm_profdata, batch_profdata_files, total_profdata, total_manifest)

    if merge_cp.returncode == 0:
        print(f"[INFO] 成功合并总 profdata 至: {total_profdata}")
    else:
        print(f"[ERROR] 最终总合并失败: {merge_cp.stderr}", file=sys.stderr)
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())