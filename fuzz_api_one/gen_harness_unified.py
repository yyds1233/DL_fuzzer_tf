#!/usr/bin/env python3
"""
tf_generate_from_yaml_unified.py  –  Generate atheris-based fuzz harnesses
from TensorFlow YAML spec files (raw_ops AND high-level APIs).

==========================================================================
CHANGES VS. tf_generate_from_yaml.py (raw_ops only)
==========================================================================

1. API INVOCATION: _call_target_api now uses importlib to call ANY tf.*
   path — tf.raw_ops.Conv2D, tf.nn.conv2d, tf.math.reduce_sum, etc.
   It also handles param_mapping for edge-case re-routing.

2. SHAPE-CONTROL PARAMS: Detected via semantic_role="shape_control" in the
   YAML.  These params (e.g., `size` in tf.image.resize, `shape` in
   tf.reshape) are NOT random tensors — they are int lists/tuples whose
   VALUES represent target dimensions.  The harness converts them to the
   type the high-level API expects (Python list, 1-D tensor, or scalar).

3. INDEX-INPUT PARAMS: Detected via semantic_role="index_input".  These
   get clamped to valid ranges based on the primary tensor's shape.

4. DTYPE HANDLING: High-level API YAMLs may lack dtype_from_attr.
   The sampler falls back to dtype_choices → test_dtype_choices → float32.

5. OPTIONAL-TENSOR HANDLING: High-level APIs often accept `None` for
   optional params (unlike raw_ops which may reject None).  The harness
   now passes None correctly.

6. SPEC LITERAL: Includes api_category, resolve_info for diagnostics.

7. COMPOSITE OPS: If api_name is a composite op (tf.nn.dropout, etc.),
   the harness calls it directly — works because api_name IS the callable.

==========================================================================
USAGE  (identical CLI to original)
==========================================================================

  python tf_generate_from_yaml_unified.py --yaml biasadd.yaml --out harness.py
  python tf_generate_from_yaml_unified.py --yaml_dir ./final/ --out_dir ./harnesses/
  python tf_generate_from_yaml_unified.py --yaml_dir ./final/ --out_dir ./harnesses/ --per_rank
"""
from __future__ import annotations

import sys
import re
import argparse
import pprint
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ════════════════════════════════════════════════════════════════
# Template
# ════════════════════════════════════════════════════════════════

TEMPLATE = '''\
#!/usr/bin/env python3
"""
Auto-generated atheris fuzz harness for TensorFlow API.
API:      {api_name}
Op:       {op_name}
Category: {api_category}
Ranks:    {test_ranks}
Strategy: {resolve_strategy}
"""
import os
import sys
import importlib
import hashlib
import random
import atheris

# Must instrument before importing TF
with atheris.instrument_imports():
    import tensorflow as tf
    import numpy as np
    import math

from utils.tf_param_sampler_unified import (
    gen_config_for_api,
    mutate_cfg,
    make_constraint_func,
)

# ============================================================
# Spec & constraints from YAML
# ============================================================

SPEC = {spec_literal}

CONSTRAINTS = {constraints_literal}


# ============================================================
# Env-configurable knobs
# ============================================================

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _seed_from_bytes(data: bytes) -> int:
    h = hashlib.sha1(data).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


# Profile knobs
SEED_TRIES = _env_int("SEED_TRIES", 8)
MUT_STEPS_MAX = _env_int("MUT_STEPS_MAX", 10)
MUT_ATTEMPTS = _env_int("MUT_ATTEMPTS", 6)
P_TYPE_MUT = _env_float("P_TYPE_MUT", 0.35)
P_SHAPE_MUT = _env_float("P_SHAPE_MUT", 0.10)
P_RANK_MUT = _env_float("P_RANK_MUT", 0.15)
P_LAYOUT_MUT = _env_float("P_LAYOUT_MUT", 0.10)


# ============================================================
# Constraint function
# ============================================================

_constraint_func = make_constraint_func(SPEC)


def constraint_func(cfg):
    return _constraint_func(cfg)


# ============================================================
# Valid config generation
# ============================================================

def gen_valid_config(spec, fdp, max_tries=None):
    if max_tries is None:
        max_tries = SEED_TRIES
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


# ============================================================
# API invocation
# ============================================================

def _call_target_api(cfg):
    """
    Call the TF API (raw_ops or high-level) with sampled parameters.

    Resolves api_name via importlib — works for any tf.* path:
      tf.raw_ops.Conv2D, tf.nn.conv2d, tf.math.reduce_sum, tf.reshape, ...
    """
    api_name = SPEC.get("api_name", "")
    if not api_name:
        raise RuntimeError("SPEC missing api_name")

    # Resolve the callable
    try:
        mod_name, func_name = api_name.rsplit(".", 1)
    except ValueError:
        raise RuntimeError(f"Invalid api_name: {{api_name!r}}")

    mod = importlib.import_module(mod_name)
    target = getattr(mod, func_name)

    # Build kwargs — only include actual API params, skip internal keys
    call_kwargs = {{}}
    params = SPEC.get("params", {{}})
    for pname in params:
        if pname not in cfg:
            continue
        if pname == "name":
            continue

        val = cfg[pname]
        p_spec = params[pname]
        kind = p_spec.get("kind", "")
        semantic_role = p_spec.get("semantic_role", "")

        # Skip None for optional params
        if val is None and kind in ("tensor_optional", "string_optional"):
            continue

        # shape_control params: high-level APIs often want a Python list,
        # not a tf.Tensor.  Convert if the value is a tensor.
        if semantic_role == "shape_control" and hasattr(val, "numpy"):
            val = val.numpy().tolist()
            # If it's a list of one element, some APIs want a scalar
            if isinstance(val, list) and len(val) == 1:
                val = val  # keep as list — safer

        # dtype_enum params: ensure it's a tf.DType
        if kind == "dtype_enum" and isinstance(val, str):
            try:
                val = tf.dtypes.as_dtype(val)
            except Exception:
                val = tf.float32

        call_kwargs[pname] = val

    return target(**call_kwargs)


# ============================================================
# Fuzz target
# ============================================================

@atheris.instrument_func
def TestOneInput(data: bytes):
    seed = _seed_from_bytes(data)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    tf.random.set_seed(seed)

    fdp = atheris.FuzzedDataProvider(data)

    # Generate initial config
    cfg = gen_valid_config(SPEC, fdp)
    if cfg is None:
        return

    # Mutate
    n_params = len(SPEC.get("params", {{}}))
    upper = max(1, min(MUT_STEPS_MAX, n_params))
    steps = fdp.ConsumeIntInRange(1, upper)

    cfg = mutate_cfg(
        SPEC, cfg, fdp,
        constraint_func=constraint_func,
        steps=steps,
        max_attempts_per_step=max(1, MUT_ATTEMPTS),
        p_type_mut=max(0.0, min(1.0, P_TYPE_MUT)),
        p_shape_mut=max(0.0, min(1.0, P_SHAPE_MUT)),
        p_rank_mut=max(0.0, min(1.0, P_RANK_MUT)),
        p_layout_mut=max(0.0, min(1.0, P_LAYOUT_MUT)),
    )

    # Call the API
    try:
        _ = _call_target_api(cfg)
    except (
        tf.errors.InvalidArgumentError,
        tf.errors.UnimplementedError,
        tf.errors.InternalError,
        tf.errors.ResourceExhaustedError,
        ValueError,
        TypeError,
        RuntimeError,
        AssertionError,
        IndexError,
        NotImplementedError,
    ):
        # Expected errors from invalid inputs — not real bugs
        return
    except Exception:
        # Catch-all for unexpected but non-crash errors
        return


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
'''


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def safe_name(s: Any, max_len: int = 120) -> str:
    if s is None:
        return "null"
    s = str(s).strip()
    if not s:
        return "empty"
    s = s.replace("::", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "empty"
    return s[:max_len]


def load_yaml_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML spec must be a mapping/dict: {path}")
    return data


def make_spec_literal(spec: Dict[str, Any]) -> str:
    """
    Convert spec to Python dict literal string, keeping only
    fields that the harness + sampler actually need at runtime.
    """
    keep_keys = {
        "api_name",
        "category",
        "api_category",         # NEW: nn / math / core / image / ...
        "primary_param",
        "op_family",
        "test_ranks",
        "test_dtype_choices",
        "layout_variants",
        "shape_vars",
        "params",
        "constraints",
        "rank_hints",
    }

    spec_copy = {}
    for k in keep_keys:
        if k in spec:
            spec_copy[k] = spec[k]

    # Embed minimal resolve_info for diagnostics
    ri = spec.get("resolve_info") or {}
    if ri:
        spec_copy["_resolve"] = {
            "strategy": ri.get("strategy", "unknown"),
            "is_raw_ops": ri.get("is_raw_ops", False),
            "raw_op_name": ri.get("raw_op_name"),
        }

    # Clean params: keep only sampler-relevant fields
    if "params" in spec_copy:
        cleaned_params = {}
        for pname, p_spec in spec_copy["params"].items():
            if not isinstance(p_spec, dict):
                continue
            cp = {}
            for field in (
                "kind", "origin", "role", "semantic_role",
                "dtype_choices", "dtype_from_attr",
                "shape_spec", "shape_spec_by_rank",
                "shape_spec_by_rank_and_layout",
                "values", "default", "range", "len_range",
                "constraints_by_rank",
            ):
                if field in p_spec:
                    cp[field] = p_spec[field]
            cleaned_params[pname] = cp
        spec_copy["params"] = cleaned_params

    return pprint.pformat(spec_copy, width=100, sort_dicts=False)


def make_constraints_literal(spec: Dict[str, Any]) -> str:
    constraints = spec.get("constraints", []) or []
    return pprint.pformat(constraints, width=100, sort_dicts=False)


def get_test_ranks(spec: Dict[str, Any]) -> List[int]:
    ranks = spec.get("test_ranks")
    if isinstance(ranks, list):
        return [int(r) for r in ranks if isinstance(r, int)]
    rh = spec.get("rank_hints") or {}
    cands = rh.get("rank_candidates") or []
    return [int(r) for r in cands if isinstance(r, int)]


def build_out_path(
    yaml_file: Path,
    spec: Dict[str, Any],
    rank: Optional[int],
    out_dir: Optional[Path],
) -> Path:
    api_name = spec.get("api_name", yaml_file.stem)

    base = f"fuzz_{safe_name(api_name)}"
    if isinstance(rank, int):
        base += f"__rank{rank}"
    filename = base + ".py"

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return yaml_file.with_name(filename)


# ════════════════════════════════════════════════════════════════
# Generation
# ════════════════════════════════════════════════════════════════

def generate_one(
    yaml_file: Path,
    spec: Dict[str, Any],
    out_file: Path,
    active_rank: Optional[int] = None,
):
    """Generate a single harness file."""
    api_name = spec.get("api_name", "unknown")
    tf_block = spec.get("tf") or {}
    op_name = tf_block.get("op_name") or api_name.split(".")[-1]
    test_ranks = get_test_ranks(spec)
    api_category = spec.get("api_category", "unknown")
    resolve_info = spec.get("resolve_info") or {}
    resolve_strategy = resolve_info.get("strategy", "unknown")

    if active_rank is not None:
        spec_for_gen = dict(spec)
        spec_for_gen["test_ranks"] = [active_rank]
    else:
        spec_for_gen = spec

    spec_literal = make_spec_literal(spec_for_gen)
    constraints_literal = make_constraints_literal(spec_for_gen)

    code = TEMPLATE.format(
        api_name=api_name,
        op_name=op_name,
        test_ranks=test_ranks if active_rank is None else [active_rank],
        api_category=api_category,
        resolve_strategy=resolve_strategy,
        spec_literal=spec_literal,
        constraints_literal=constraints_literal,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(code, encoding="utf-8")
    rank_str = f"rank={active_rank}" if active_rank is not None else "all ranks"
    print(f"[+] Generated {out_file} ({rank_str}) [{api_category}]")


def generate_from_yaml(
    yaml_path: str,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    single: bool = False,
    per_rank: bool = False,
):
    yaml_file = Path(yaml_path)
    spec = load_yaml_spec(yaml_file)
    ranks = get_test_ranks(spec)
    out_dir_path = Path(out_dir).resolve() if out_dir else None

    if out_path is not None:
        out_file = Path(out_path)
        generate_one(yaml_file, spec, out_file)
        return

    if per_rank and ranks:
        for r in ranks:
            out_file = build_out_path(yaml_file, spec, r, out_dir_path)
            generate_one(yaml_file, spec, out_file, active_rank=r)
        return

    out_file = build_out_path(yaml_file, spec, None, out_dir_path)
    generate_one(yaml_file, spec, out_file)


# ════════════════════════════════════════════════════════════════
# Batch generation
# ════════════════════════════════════════════════════════════════

def generate_batch(
    yaml_dir: str,
    out_dir: str,
    single: bool = False,
    per_rank: bool = False,
):
    yaml_dir_path = Path(yaml_dir).resolve()
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(yaml_dir_path.glob("*.yaml"))
    if not yaml_files:
        print(f"[!] No YAML files found in {yaml_dir_path}")
        return

    count = 0
    errors = 0
    stats = {"raw_ops": 0, "resolved_hl": 0, "unresolved_hl": 0}

    for yf in yaml_files:
        try:
            spec = load_yaml_spec(yf)
            ranks = get_test_ranks(spec)

            # Track stats
            ri = spec.get("resolve_info") or {}
            if ri.get("is_raw_ops"):
                stats["raw_ops"] += 1
            elif ri.get("raw_op_name"):
                stats["resolved_hl"] += 1
            else:
                stats["unresolved_hl"] += 1

            if per_rank and ranks:
                for r in ranks:
                    out_file = build_out_path(yf, spec, r, out_dir_path)
                    generate_one(yf, spec, out_file, active_rank=r)
                    count += 1
            else:
                out_file = build_out_path(yf, spec, None, out_dir_path)
                generate_one(yf, spec, out_file)
                count += 1
        except Exception as e:
            print(f"[!] Error processing {yf}: {e}")
            errors += 1

    print(f"\n=== Generation Summary ===")
    print(f"  YAML files processed: {len(yaml_files)}")
    print(f"  Harnesses generated:  {count}")
    print(f"  Errors:               {errors}")
    print(f"  raw_ops:              {stats['raw_ops']}")
    print(f"  resolved high-level:  {stats['resolved_hl']}")
    print(f"  unresolved (composite): {stats['unresolved_hl']}")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate atheris fuzz harnesses from TF YAML specs "
                    "(raw_ops + high-level APIs)."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--yaml", help="Single YAML spec file")
    g.add_argument("--yaml_dir", help="Directory of YAML spec files")

    ap.add_argument("--out", default=None,
                    help="Output .py path (single file, explicit)")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for generated harnesses")
    ap.add_argument("--single", action="store_true",
                    help="Generate one harness with all ranks (default)")
    ap.add_argument("--per_rank", action="store_true",
                    help="Generate one harness per test_rank")
    args = ap.parse_args()

    if args.yaml:
        generate_from_yaml(
            yaml_path=args.yaml,
            out_path=args.out,
            out_dir=args.out_dir,
            single=args.single,
            per_rank=args.per_rank,
        )
    else:
        if not args.out_dir:
            raise SystemExit("--out_dir is required with --yaml_dir")
        generate_batch(
            yaml_dir=args.yaml_dir,
            out_dir=args.out_dir,
            single=args.single,
            per_rank=args.per_rank,
        )


if __name__ == "__main__":
    main()
