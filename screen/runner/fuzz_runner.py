# screen/runner/fuzz_runner.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EpochResult:
    """Lightweight result from a single fuzz epoch."""
    returncode: int
    timed_out: bool


def run_one_epoch(
    *,
    python: str,
    harness_path: Path,
    corpus_dir: Path,
    crash_dir: Path,
    log_path: Path,
    epoch_sec: int,
    fuzz_flags: List[str],
    profile_env: Dict[str, str],
) -> EpochResult:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    crash_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(profile_env)

    cmd = [
        python,
        str(harness_path),
        str(corpus_dir),
        f"-artifact_prefix={str(crash_dir)}/",
        f"-max_total_time={int(epoch_sec)}",
        *fuzz_flags,
    ]

    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="ignore") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)
        try:
            proc.wait(timeout=epoch_sec + 20)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            # Append a driver-level marker to the log for easier post-hoc parsing
            lf.write("\n[DRIVER] epoch killed after timeout\n")

    # Save stderr tail for observability (last 4KB)
    if timed_out or (proc.returncode and proc.returncode != 0):
        stderr_tail_path = log_path.with_suffix(".stderr_tail")
        try:
            log_text = log_path.read_text(errors="ignore")
            tail = log_text[-4096:] if len(log_text) > 4096 else log_text
            stderr_tail_path.write_text(
                f"[DRIVER] returncode={proc.returncode} timed_out={timed_out}\n{tail}",
                encoding="utf-8",
            )
        except Exception:
            pass

    return EpochResult(returncode=proc.returncode, timed_out=timed_out)
