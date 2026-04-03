# screen/metrics/parse_libfuzzer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


# --- Standard libFuzzer patterns ---
COV_RE = re.compile(r"\bcov:\s*([0-9]+)")
FT_RE = re.compile(r"\bft:\s*([0-9]+)")
EXECS_RE = re.compile(r"\bexec/s:\s*([0-9]+(?:\.[0-9]+)?)([kKmM]?)")

# Only capture cov/ft after INITED (skip seed replay/pulse)
INITED_COV_FT_RE = re.compile(r"^\s*#\d+\s+INITED\b.*\bcov:\s*(\d+)\s+ft:\s*(\d+)\b", re.M)
AFTER_INITED_COV_FT_RE = re.compile(r"^\s*#\d+\s+\w+\b.*\bcov:\s*(\d+)\s+ft:\s*(\d+)\b", re.M)

# --- Atheris fallback patterns ---
# Atheris may emit lines like: total_coverage=12345, features=6789
ATHERIS_COV_RE = re.compile(r"\btotal_coverage[=:]\s*([0-9]+)")
ATHERIS_FEAT_RE = re.compile(r"\bfeatures[=:]\s*([0-9]+)")


def _parse_num_with_suffix(x: str, suffix: str) -> float:
    v = float(x)
    if suffix.lower() == "k":
        return v * 1_000.0
    if suffix.lower() == "m":
        return v * 1_000_000.0
    return v


def parse_fuzzer_log(log_path: Path) -> Dict[str, Optional[float]]:
    text = log_path.read_text(errors="ignore")

    # --- exec/s: always try standard libFuzzer ---
    exec_s = None
    hits = EXECS_RE.findall(text)
    if hits:
        x, suf = hits[-1]
        exec_s = _parse_num_with_suffix(x, suf)

    # --- Path 1: Standard libFuzzer with INITED marker ---
    m = INITED_COV_FT_RE.search(text)
    if m:
        tail = text[m.start():]
        pairs = [(int(a), int(b)) for a, b in AFTER_INITED_COV_FT_RE.findall(tail)]
        if pairs:
            cov_first, ft_first = pairs[0]
            cov_last, ft_last = pairs[-1]
        else:
            cov_first = cov_last = int(m.group(1))
            ft_first = ft_last = int(m.group(2))
        return {
            "cov_first": cov_first,
            "cov_last": cov_last,
            "ft_first": ft_first,
            "ft_last": ft_last,
            "exec_s_last": exec_s,
        }

    # --- Path 2: Global standard cov/ft (no INITED marker) ---
    covs = [int(x) for x in COV_RE.findall(text)]
    fts = [int(x) for x in FT_RE.findall(text)]
    if covs or fts:
        return {
            "cov_first": covs[0] if covs else None,
            "cov_last": covs[-1] if covs else None,
            "ft_first": fts[0] if fts else None,
            "ft_last": fts[-1] if fts else None,
            "exec_s_last": exec_s,
        }

    # --- Path 3: Atheris fallback ---
    atheris_covs = [int(x) for x in ATHERIS_COV_RE.findall(text)]
    atheris_fts = [int(x) for x in ATHERIS_FEAT_RE.findall(text)]
    if atheris_covs or atheris_fts:
        return {
            "cov_first": atheris_covs[0] if atheris_covs else None,
            "cov_last": atheris_covs[-1] if atheris_covs else None,
            "ft_first": atheris_fts[0] if atheris_fts else None,
            "ft_last": atheris_fts[-1] if atheris_fts else None,
            "exec_s_last": exec_s,
        }

    # --- No coverage data found ---
    return {
        "cov_first": None,
        "cov_last": None,
        "ft_first": None,
        "ft_last": None,
        "exec_s_last": exec_s,
    }
