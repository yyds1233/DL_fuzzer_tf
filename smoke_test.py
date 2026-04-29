import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


def to_text(x):
    """
    subprocess.TimeoutExpired 里的 stdout/stderr 有时会是 bytes。
    这里统一转成 str，避免 log_f.write(bytes) 报错。
    """
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def run_one_file(py_file: Path, timeout: int):
    cmd = ["python3", str(py_file)]
    start = datetime.now()

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        end = datetime.now()

        if proc.returncode == 0:
            status = "EXIT_OK"
        else:
            status = "FAILED"

        return {
            "file": str(py_file),
            "status": status,
            "returncode": proc.returncode,
            "stdout": to_text(proc.stdout),
            "stderr": to_text(proc.stderr),
            "start_time": start.isoformat(timespec="seconds"),
            "end_time": end.isoformat(timespec="seconds"),
            "duration_seconds": round((end - start).total_seconds(), 2),
            "cmd": " ".join(cmd),
        }

    except subprocess.TimeoutExpired as e:
        end = datetime.now()

        return {
            "file": str(py_file),
            "status": "TIMEOUT_OK",
            "returncode": None,
            "stdout": to_text(e.stdout),
            "stderr": to_text(e.stderr),
            "start_time": start.isoformat(timespec="seconds"),
            "end_time": end.isoformat(timespec="seconds"),
            "duration_seconds": round((end - start).total_seconds(), 2),
            "cmd": " ".join(cmd),
        }

    except Exception as e:
        end = datetime.now()

        return {
            "file": str(py_file),
            "status": "FAILED",
            "returncode": None,
            "stdout": "",
            "stderr": repr(e),
            "start_time": start.isoformat(timespec="seconds"),
            "end_time": end.isoformat(timespec="seconds"),
            "duration_seconds": round((end - start).total_seconds(), 2),
            "cmd": " ".join(cmd),
        }


def write_log_entry(log_f, result):
    log_f.write("=" * 100 + "\n")
    log_f.write(f"FILE: {result.get('file')}\n")
    log_f.write(f"STATUS: {result.get('status')}\n")
    log_f.write(f"RETURNCODE: {result.get('returncode')}\n")
    log_f.write(f"START: {result.get('start_time')}\n")
    log_f.write(f"END: {result.get('end_time')}\n")
    log_f.write(f"DURATION_SECONDS: {result.get('duration_seconds')}\n")
    log_f.write(f"CMD: {result.get('cmd')}\n")

    log_f.write("\nSTDOUT:\n")
    log_f.write(to_text(result.get("stdout")))
    log_f.write("\n")

    log_f.write("\nSTDERR:\n")
    log_f.write(to_text(result.get("stderr")))
    log_f.write("\n\n")

    log_f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="批量执行 harness py 文件，短时间内无报错则认为通过"
    )

    parser.add_argument(
        "input_dir",
        help="harness py 文件所在目录",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3,
        help="每个脚本最多执行多少秒，默认 3 秒",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录",
    )

    parser.add_argument(
        "--log",
        default="check_harness_runtime.log",
        help="日志文件路径",
    )

    parser.add_argument(
        "--failed-json",
        default="check_harness_failed.json",
        help="失败文件列表 JSON",
    )

    parser.add_argument(
        "--all-json",
        default="check_harness_all_results.json",
        help="所有结果 JSON",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只检查前 N 个文件，方便测试",
    )

    args = parser.parse_args()

    root = Path(args.input_dir)

    if not root.is_dir():
        raise NotADirectoryError(f"不是有效文件夹：{root}")

    py_files = sorted(root.rglob("*.py") if args.recursive else root.glob("*.py"))

    if args.limit is not None:
        py_files = py_files[: args.limit]

    print(f"发现 {len(py_files)} 个 py 文件")
    print(f"每个最多执行 {args.timeout} 秒")
    print(f"log: {args.log}")

    results = []
    failed = []

    with open(args.log, "w", encoding="utf-8") as log_f:
        log_f.write(f"BATCH START: {datetime.now().isoformat(timespec='seconds')}\n")
        log_f.write(f"TOTAL FILES: {len(py_files)}\n")
        log_f.write(f"TIMEOUT: {args.timeout}\n\n")
        log_f.flush()

        for idx, py_file in enumerate(py_files, 1):
            result = run_one_file(py_file, args.timeout)
            results.append(result)
            write_log_entry(log_f, result)

            if result["status"] == "FAILED":
                failed.append(result)

            print(
                f"[{idx}/{len(py_files)}] "
                f"{result['status']} "
                f"{py_file}"
            )

        success_like = sum(
            1 for r in results if r["status"] in ("TIMEOUT_OK", "EXIT_OK")
        )
        failed_count = len(failed)

        log_f.write("=" * 100 + "\n")
        log_f.write("SUMMARY\n")
        log_f.write(f"SUCCESS_LIKE: {success_like}\n")
        log_f.write(f"FAILED: {failed_count}\n")
        log_f.write(f"BATCH END: {datetime.now().isoformat(timespec='seconds')}\n")
        log_f.flush()

    with open(args.failed_json, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    with open(args.all_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("完成")
    print(f"总数：{len(results)}")
    print(f"疑似通过：{success_like}")
    print(f"失败：{failed_count}")
    print(f"日志：{Path(args.log).resolve()}")
    print(f"失败 JSON：{Path(args.failed_json).resolve()}")
    print(f"全部结果 JSON：{Path(args.all_json).resolve()}")


if __name__ == "__main__":
    main()