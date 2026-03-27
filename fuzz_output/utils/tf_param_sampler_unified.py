#!/usr/bin/env python3
"""
tf_param_sampler_unified.py  –  TensorFlow parameter sampler & mutator
for atheris-based fuzz harnesses.  Supports BOTH raw_ops AND high-level APIs.

==========================================================================
CHANGES VS. tf_param_sampler.py (raw_ops only)
==========================================================================

1. SEMANTIC-ROLE-AWARE SAMPLING
   The unified YAML enriches every param with `semantic_role`:
     data_tensor, weight_tensor, aux_tensor, index_input,
     shape_control, fixed_arity_list, layout_attr, scalar_attr, ...

   The sampler now uses these roles to produce *semantically valid* values:

   - shape_control (e.g. `shape` in Reshape, `size` in tf.image.resize):
     Generated as a Python list of positive ints (NOT a random tensor),
     whose length matches the target rank.  Values drawn from shape_vars
     so they stay small and OOM-safe.

   - index_input (e.g. `indices` in Gather, `perm` in Transpose):
     Generated as int tensors whose VALUES are valid indices into the
     primary tensor's corresponding dimension.

   - weight_tensor (e.g. `filters` in Conv2D):
     Gets its own shape_spec (typically from shape_spec_by_rank), with
     its own dtype matching the primary tensor via dtype_from_attr.

   - fixed_arity_list (e.g. `strides`, `dilations`):
     Length is clamped to len_range; elements are ≥ 1 (not 0).

2. HIGH-LEVEL API DTYPE RESOLUTION
   raw_ops use `dtype_from_attr: T` to share a single dtype across params.
   High-level APIs may instead have explicit `dtype_choices` per param, or
   no dtype annotation at all (inferred from input tensor at runtime).

   Resolution order (per param):
     dtype_from_attr → dtype_choices → test_dtype_choices → "float32"

3. FLOAT_LIST KIND  (new)
   Some high-level APIs have float-list attrs (e.g., `scales`).
   Added `sample_float_list` and corresponding mutation.

4. SHAPE-CONTROL MUTATION
   Mutating a shape_control param re-draws a dimension list (not a tensor).

5. INDEX CLAMPING
   After any mutation that touches the primary tensor's shape, index_input
   params are re-clamped to stay within valid bounds.

6. COMPOSITE-OP COMPATIBILITY
   For composite ops (dropout, l2_normalize, ...) that have no OpDef,
   params are classified by the LLM.  The sampler handles these identically
   since it works purely from the YAML spec — it never reads the OpDef.

==========================================================================
ENV KNOBS  (same as original, plus new ones)
==========================================================================
  P_RANK_OUTLIER=0.05     try rank outside test_ranks
  P_RANK_EXTREME=0.01     try rank 0 or rank 6+
  P_LAYOUT_FLIP=0.15      flip layout during mutation
  P_SHAPE_CTRL_MUT=0.10   mutate shape_control param values
  P_INDEX_RECLAMP=1.0     re-clamp indices after shape mutation (1.0=always)
  ALLOW_EMPTY=0            allow 0-length dimensions
  P_EMPTY_DIM=0.01
"""
from __future__ import annotations

import math
import os
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import atheris

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    tf = None  # type: ignore
    np = None  # type: ignore


# ════════════════════════════════════════════════════════════════
# Env helpers
# ════════════════════════════════════════════════════════════════

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


# ════════════════════════════════════════════════════════════════
# TF dtype mapping
# ════════════════════════════════════════════════════════════════

_TF_DTYPE_MAP = {
    "float32": "float32", "float64": "float64",
    "float16": "float16", "bfloat16": "bfloat16",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "bool": "bool", "complex64": "complex64", "complex128": "complex128",
    "string": "string",
}


def resolve_tf_dtype(dtype_name: str):
    mapped = _TF_DTYPE_MAP.get(dtype_name, dtype_name)
    return tf.dtypes.as_dtype(mapped)


def is_float_dtype(d: str) -> bool:
    return d in ("float32", "float64", "float16", "bfloat16")


def is_complex_dtype(d: str) -> bool:
    return d in ("complex64", "complex128")


def is_int_dtype(d: str) -> bool:
    return d in ("int8", "int16", "int32", "int64",
                 "uint8", "uint16", "uint32", "uint64")


# ════════════════════════════════════════════════════════════════
# 1. Rank selection
# ════════════════════════════════════════════════════════════════

def pick_rank(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> int:
    test_ranks = spec.get("test_ranks") or []
    if not test_ranks:
        primary = spec.get("primary_param")
        params = spec.get("params") or {}
        p = params.get(primary) if primary else None
        if isinstance(p, dict):
            sbr = p.get("shape_spec_by_rank") or {}
            if sbr:
                test_ranks = sorted(int(k) for k in sbr.keys())
        if not test_ranks:
            test_ranks = [2, 3, 4]

    p_extreme = _env_float("P_RANK_EXTREME", 0.01)
    p_outlier = _env_float("P_RANK_OUTLIER", 0.05)
    roll = fdp.ConsumeIntInRange(0, 999)

    if roll < int(p_extreme * 1000):
        extreme = [r for r in [0, 6, 7, 8] if r not in test_ranks]
        if extreme:
            return fdp.PickValueInList(extreme)

    if roll < int((p_extreme + p_outlier) * 1000):
        mn, mx = min(test_ranks), max(test_ranks)
        outlier = []
        if mn - 1 >= 1:
            outlier.append(mn - 1)
        if mx + 1 <= 8:
            outlier.append(mx + 1)
        outlier = [r for r in outlier if r not in test_ranks]
        if outlier:
            return fdp.PickValueInList(outlier)

    return fdp.PickValueInList(test_ranks)


# ════════════════════════════════════════════════════════════════
# 2. Layout selection
# ════════════════════════════════════════════════════════════════

def pick_layout(
    spec: Dict[str, Any],
    rank: int,
    fdp: atheris.FuzzedDataProvider,
) -> Optional[str]:
    layout_variants = spec.get("layout_variants") or {}
    if not layout_variants:
        return None
    applicable = []
    for name, info in layout_variants.items():
        if isinstance(info, dict) and rank in (info.get("applies_to_ranks") or []):
            applicable.append(name)
    if not applicable:
        return None
    return fdp.PickValueInList(applicable)


# ════════════════════════════════════════════════════════════════
# 3. Dtype selection  (UNIFIED)
# ════════════════════════════════════════════════════════════════

def pick_dtype(spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> str:
    """Pick a global dtype from test_dtype_choices."""
    choices = spec.get("test_dtype_choices") or ["float32"]
    return fdp.PickValueInList(choices)


def resolve_param_dtype(
    p_spec: Dict[str, Any],
    global_dtype: str,
    fdp: atheris.FuzzedDataProvider,
) -> str:
    """
    Resolve dtype for a single param.

    Priority:
    1. dtype_from_attr → use global_dtype (shared type attr)
    2. dtype_choices → pick from param's own choices
    3. fall back to global_dtype
    """
    if p_spec.get("dtype_from_attr"):
        return global_dtype
    dc = p_spec.get("dtype_choices")
    if dc and isinstance(dc, list):
        return fdp.PickValueInList(dc)
    return global_dtype


# ════════════════════════════════════════════════════════════════
# 4. Shape vars sampling
# ════════════════════════════════════════════════════════════════

def gen_shape_vars(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, int]:
    shape_vars = {}
    allow_empty = _env_bool("ALLOW_EMPTY", False)
    p_empty = _env_float("P_EMPTY_DIM", 0.0)

    for name, rng in (spec.get("shape_vars") or {}).items():
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = int(rng[0]), int(rng[1])
            val = fdp.ConsumeIntInRange(lo, hi)
            if allow_empty and p_empty > 0:
                n = name.lower()
                skip = n in ("n", "c") or n.startswith("n_") or n.startswith("c_")
                if not skip and any(k in n for k in ("h", "w", "d", "l", "len", "seq")):
                    if fdp.ConsumeIntInRange(0, 999) < int(p_empty * 1000):
                        val = 0
            shape_vars[name] = val
        else:
            shape_vars[name] = 1
    return shape_vars


# ════════════════════════════════════════════════════════════════
# 5. Shape resolution
# ════════════════════════════════════════════════════════════════

def resolve_shape(
    shape_spec: List[Any],
    shape_vars: Dict[str, int],
) -> Tuple[int, ...]:
    dims = []
    for dim in shape_spec:
        if isinstance(dim, str):
            dims.append(int(shape_vars.get(dim, 1)))
        else:
            dims.append(int(dim))
    return tuple(dims)


def get_shape_spec_for_param(
    pname: str,
    p_spec: Dict[str, Any],
    spec: Dict[str, Any],
    rank: int,
    layout: Optional[str],
) -> List[Any]:
    if layout:
        sbrl = p_spec.get("shape_spec_by_rank_and_layout") or {}
        rk = str(rank)
        if rk in sbrl and isinstance(sbrl[rk], dict) and layout in sbrl[rk]:
            return list(sbrl[rk][layout])

    sbr = p_spec.get("shape_spec_by_rank") or {}
    rk = str(rank)
    if rk in sbr:
        return list(sbr[rk])

    ss = p_spec.get("shape_spec") or []
    return list(ss)


def build_shape_spec_for_outlier_rank(
    pname: str,
    p_spec: Dict[str, Any],
    spec: Dict[str, Any],
    rank: int,
) -> List[str]:
    sbr = p_spec.get("shape_spec_by_rank") or {}
    if not sbr:
        return [f"D{i+1}" for i in range(rank)]

    known = sorted(int(k) for k in sbr.keys())
    nearest = min(known, key=lambda k: abs(k - rank))
    base = list(sbr[str(nearest)])

    if rank == len(base):
        return base
    elif rank < len(base):
        return base[:rank]
    else:
        extra = rank - len(base)
        return base + [f"D_ext{i+1}" for i in range(extra)]


# ════════════════════════════════════════════════════════════════
# 6. TF tensor creation
# ════════════════════════════════════════════════════════════════

def create_tf_tensor(
    shape: Tuple[int, ...],
    dtype_str: str,
    fdp: atheris.FuzzedDataProvider,
):
    tf_dtype = resolve_tf_dtype(dtype_str)

    if is_float_dtype(dtype_str):
        arr = np.random.randn(*shape).astype(np.float64)
        return tf.constant(arr, dtype=tf_dtype)

    if is_complex_dtype(dtype_str):
        real = np.random.randn(*shape).astype(np.float64)
        imag = np.random.randn(*shape).astype(np.float64)
        return tf.constant(real + 1j * imag, dtype=tf_dtype)

    if is_int_dtype(dtype_str):
        arr = np.random.randint(-10, 11, size=shape)
        return tf.constant(arr, dtype=tf_dtype)

    if dtype_str == "bool":
        arr = np.random.random(shape) > 0.5
        return tf.constant(arr, dtype=tf.bool)

    return tf.zeros(shape, dtype=tf_dtype)


def create_index_tensor(
    shape: Tuple[int, ...],
    max_val: int,
    dtype_str: str,
    fdp: atheris.FuzzedDataProvider,
):
    """
    Create an int tensor whose values are valid indices in [0, max_val).
    Used for index_input params (indices, perm, segment_ids).
    """
    tf_dtype = resolve_tf_dtype(dtype_str)
    if max_val <= 0:
        max_val = 1
    arr = np.random.randint(0, max_val, size=shape)
    return tf.constant(arr, dtype=tf_dtype)


def create_shape_control_value(
    rank: int,
    shape_vars: Dict[str, int],
    fdp: atheris.FuzzedDataProvider,
) -> List[int]:
    """
    Create a shape_control value: a Python list of positive ints.
    Used for params like `shape` in tf.reshape, `size` in tf.image.resize.

    The length of the list = the target rank of the OUTPUT, not the input.
    Values are drawn from shape_vars to stay OOM-safe.
    """
    vals = []
    sv_values = list(shape_vars.values()) or [1, 2, 4, 8]
    for i in range(rank):
        val = fdp.PickValueInList(sv_values) if sv_values else fdp.ConsumeIntInRange(1, 16)
        vals.append(max(1, val))
    return vals


def mutate_tf_tensor_content(tensor, fdp: atheris.FuzzedDataProvider):
    shape = tuple(tensor.shape)
    dtype_str = tensor.dtype.name
    new_t = create_tf_tensor(shape, dtype_str, fdp)

    if is_float_dtype(dtype_str) and tensor.shape.num_elements() > 0:
        roll = fdp.ConsumeIntInRange(0, 99)
        if roll == 0:
            arr = new_t.numpy().copy()
            arr.flat[0] = float("nan")
            new_t = tf.constant(arr, dtype=tensor.dtype)
        elif roll == 1:
            arr = new_t.numpy().copy()
            arr.flat[-1] = float("inf")
            new_t = tf.constant(arr, dtype=tensor.dtype)

    return new_t


# ════════════════════════════════════════════════════════════════
# 7. Per-kind sampling
# ════════════════════════════════════════════════════════════════

def sample_int(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> int:
    if "values" in p_spec:
        return int(fdp.PickValueInList(list(p_spec["values"])))
    lo, hi = p_spec.get("range", [0, 10])
    return fdp.ConsumeIntInRange(int(lo), int(hi))


def sample_float(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> float:
    if "values" in p_spec:
        return float(fdp.PickValueInList(list(p_spec["values"])))
    lo, hi = p_spec.get("range", [-1.0, 1.0])
    steps = int(p_spec.get("steps", 1000))
    idx = fdp.ConsumeIntInRange(0, steps)
    return float(lo) + (float(hi) - float(lo)) * (idx / steps)


def sample_bool(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> bool:
    return fdp.ConsumeBool()


def sample_enum(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    values = p_spec.get("values") or []
    if not values:
        return None
    return fdp.PickValueInList(list(values))


def sample_int_list(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> List[int]:
    len_lo, len_hi = p_spec.get("len_range", [1, 4])
    length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
    lo, hi = p_spec.get("range", [0, 10])
    return [fdp.ConsumeIntInRange(int(lo), int(hi)) for _ in range(length)]


def sample_float_list(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> List[float]:
    """NEW: Sample a float_list param."""
    len_lo, len_hi = p_spec.get("len_range", [1, 4])
    length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
    lo, hi = p_spec.get("range", [-1.0, 1.0])
    result = []
    for _ in range(length):
        steps = 1000
        idx = fdp.ConsumeIntInRange(0, steps)
        result.append(float(lo) + (float(hi) - float(lo)) * (idx / steps))
    return result


def sample_string_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    return p_spec.get("default", None)


def sample_dtype_enum(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    values = p_spec.get("values") or ["float32"]
    chosen = fdp.PickValueInList(list(values))
    try:
        return resolve_tf_dtype(chosen)
    except Exception:
        return resolve_tf_dtype("float32")


def sample_tensor(
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    dtype_str: str,
    rank: int,
    layout: Optional[str],
    spec: Dict[str, Any],
    pname: str,
) -> Any:
    """
    Sample a TF tensor.  Dispatches by semantic_role for special handling.
    """
    semantic_role = p_spec.get("semantic_role", "")

    # ── shape_control: return a Python list of positive ints ─────
    if semantic_role == "shape_control":
        shape_spec = get_shape_spec_for_param(pname, p_spec, spec, rank, layout)
        if shape_spec and shape_spec[0] != "TODO_SHAPE":
            resolved = resolve_shape(shape_spec, shape_vars)
            # shape_control values should be the OUTPUT dimensions
            return tf.constant(list(resolved), dtype=tf.int32)
        else:
            # Fallback: generate a target shape of length = rank
            vals = create_shape_control_value(rank, shape_vars, fdp)
            return tf.constant(vals, dtype=tf.int32)

    # ── index_input: int tensor with valid index values ──────────
    if semantic_role == "index_input":
        shape_spec = get_shape_spec_for_param(pname, p_spec, spec, rank, layout)
        if not shape_spec or shape_spec[0] == "TODO_SHAPE":
            shape_spec = build_shape_spec_for_outlier_rank(pname, p_spec, spec, rank)
        for dim in shape_spec:
            if isinstance(dim, str) and dim not in shape_vars:
                shape_vars[dim] = fdp.ConsumeIntInRange(1, 8)
        shape = resolve_shape(shape_spec, shape_vars)

        # max_val: use the primary tensor's first-axis size as a heuristic
        primary = spec.get("primary_param")
        primary_spec = (spec.get("params") or {}).get(primary) if primary else None
        max_val = 8
        if isinstance(primary_spec, dict):
            p_shape_spec = get_shape_spec_for_param(
                primary, primary_spec, spec, rank, layout
            )
            if p_shape_spec:
                first_dim = p_shape_spec[0] if p_shape_spec else "D1"
                if isinstance(first_dim, str):
                    max_val = shape_vars.get(first_dim, 8)
                else:
                    max_val = int(first_dim)

        idx_dtype = "int32"
        dc = p_spec.get("dtype_choices") or []
        if dc:
            idx_dtype = dc[0]
        elif is_int_dtype(dtype_str):
            idx_dtype = dtype_str

        return create_index_tensor(shape, max_val, idx_dtype, fdp)

    # ── default tensor sampling ──────────────────────────────────
    shape_spec = get_shape_spec_for_param(pname, p_spec, spec, rank, layout)
    if not shape_spec or (len(shape_spec) == 1 and shape_spec[0] == "TODO_SHAPE"):
        shape_spec = build_shape_spec_for_outlier_rank(pname, p_spec, spec, rank)

    for dim in shape_spec:
        if isinstance(dim, str) and dim not in shape_vars:
            shape_vars[dim] = fdp.ConsumeIntInRange(1, 8)

    shape = resolve_shape(shape_spec, shape_vars)
    return create_tf_tensor(shape, dtype_str, fdp)


def sample_tensor_optional(
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    dtype_str: str,
    rank: int,
    layout: Optional[str],
    spec: Dict[str, Any],
    pname: str,
) -> Any:
    if fdp.ConsumeBool():
        return None
    return sample_tensor(p_spec, fdp, shape_vars, dtype_str, rank, layout, spec, pname)


# ════════════════════════════════════════════════════════════════
# 8. Top-level config generation  (UNIFIED)
# ════════════════════════════════════════════════════════════════

def gen_config_for_api(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, Any]:
    """
    Generate a complete parameter configuration.
    Works for both raw_ops and high-level APIs.
    """
    cfg: Dict[str, Any] = {}

    # 1. Pick rank
    rank = pick_rank(spec, fdp)
    cfg["_rank"] = rank

    # 2. Pick layout
    layout = pick_layout(spec, rank, fdp)
    cfg["_layout"] = layout

    # 3. Pick global dtype
    dtype_str = pick_dtype(spec, fdp)
    cfg["_dtype_str"] = dtype_str

    # 4. Sample shape_vars
    shape_vars = gen_shape_vars(spec, fdp)
    _ensure_outlier_rank_vars(spec, rank, shape_vars, fdp)
    cfg["_shape_vars"] = shape_vars

    # 5. Sample each param
    params = spec.get("params") or {}
    for pname, p_spec in params.items():
        if not isinstance(p_spec, dict):
            continue
        kind = p_spec.get("kind", "")
        semantic_role = p_spec.get("semantic_role", "")

        # Resolve this param's dtype
        param_dtype = resolve_param_dtype(p_spec, dtype_str, fdp)

        # Layout-controlled params: set to chosen layout string
        if pname == "data_format" and layout:
            cfg[pname] = layout
            continue

        # Sample by kind
        if kind == "tensor":
            cfg[pname] = sample_tensor(
                p_spec, fdp, shape_vars, param_dtype,
                rank, layout, spec, pname,
            )

        elif kind == "tensor_optional":
            cfg[pname] = sample_tensor_optional(
                p_spec, fdp, shape_vars, param_dtype,
                rank, layout, spec, pname,
            )

        elif kind == "tensor_list":
            len_lo, len_hi = p_spec.get("len_range", [1, 4])
            length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
            cfg[pname] = [
                sample_tensor(
                    p_spec, fdp, shape_vars, param_dtype,
                    rank, layout, spec, pname,
                )
                for _ in range(length)
            ]

        elif kind == "int":
            cfg[pname] = sample_int(p_spec, fdp)

        elif kind == "float":
            cfg[pname] = sample_float(p_spec, fdp)

        elif kind == "bool":
            cfg[pname] = sample_bool(p_spec, fdp)

        elif kind == "enum":
            cfg[pname] = sample_enum(p_spec, fdp)

        elif kind == "int_list":
            cfg[pname] = sample_int_list(p_spec, fdp)

        elif kind == "float_list":
            cfg[pname] = sample_float_list(p_spec, fdp)

        elif kind == "string_optional":
            cfg[pname] = sample_string_optional(p_spec, fdp)

        elif kind == "dtype_enum":
            cfg[pname] = sample_dtype_enum(p_spec, fdp)

        # else: skip unknown kinds (name, meta, etc.)

    # 6. Remove 'name' param
    cfg.pop("name", None)

    return cfg


def _ensure_outlier_rank_vars(
    spec: Dict[str, Any],
    rank: int,
    shape_vars: Dict[str, int],
    fdp: atheris.FuzzedDataProvider,
) -> None:
    primary = spec.get("primary_param")
    params = spec.get("params") or {}
    p = params.get(primary) if primary else None
    if not isinstance(p, dict):
        return
    sbr = p.get("shape_spec_by_rank") or {}
    if str(rank) in sbr:
        return
    synth = build_shape_spec_for_outlier_rank(primary, p, spec, rank)
    for dim in synth:
        if isinstance(dim, str) and dim not in shape_vars:
            shape_vars[dim] = fdp.ConsumeIntInRange(1, 16)


# ════════════════════════════════════════════════════════════════
# 9. Constraint checking
# ════════════════════════════════════════════════════════════════

def make_constraint_func(
    spec: Dict[str, Any],
) -> Callable[[Dict[str, Any]], bool]:
    top_constraints = spec.get("constraints") or []
    if not isinstance(top_constraints, list):
        top_constraints = []

    primary = spec.get("primary_param")
    params = spec.get("params") or {}
    per_rank: Dict[int, List[str]] = {}
    if primary and primary in params:
        p = params[primary]
        if isinstance(p, dict):
            cbr = p.get("constraints_by_rank") or {}
            for rk, clist in cbr.items():
                try:
                    per_rank[int(rk)] = list(clist) if isinstance(clist, list) else []
                except (ValueError, TypeError):
                    pass

    def constraint_func(cfg: Dict[str, Any]) -> bool:
        sv = cfg.get("_shape_vars", {})
        rank = cfg.get("_rank")

        locs = dict(sv)
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            locs[k] = v
        locs["tf"] = tf
        locs["math"] = math

        for expr in top_constraints:
            try:
                if not eval(expr, {"__builtins__": {}}, locs):
                    return False
            except Exception:
                return False

        if isinstance(rank, int) and rank in per_rank:
            for expr in per_rank[rank]:
                try:
                    if not eval(expr, {"__builtins__": {}}, locs):
                        return False
                except Exception:
                    return False

        return True

    return constraint_func


# ════════════════════════════════════════════════════════════════
# 10. Mutation  (UNIFIED — adds shape_control + index re-clamp)
# ════════════════════════════════════════════════════════════════

def _pick_other(fdp, vals, cur):
    vals = list(vals)
    if len(vals) <= 1:
        return cur
    for _ in range(4):
        v = fdp.PickValueInList(vals)
        if v != cur:
            return v
    return fdp.PickValueInList(vals)


def _mutate_int_value(p_spec, fdp, cur: int) -> int:
    if "values" in p_spec and p_spec["values"]:
        cand = _pick_other(fdp, p_spec["values"], cur)
        return int(cand) if not isinstance(cand, (list, tuple)) else int(cand[0]) if cand else cur
    lo, hi = p_spec.get("range", [0, 10])
    delta = fdp.ConsumeIntInRange(-3, 3)
    return max(int(lo), min(int(hi), int(cur) + delta))


def _mutate_float_value(p_spec, fdp, cur: float) -> float:
    specials = [0.0, 1.0, -1.0, 1e-12, 1e-6, 1e-3, 1e3]
    if fdp.ConsumeBool():
        v = float(fdp.PickValueInList(specials))
        return v if v != cur else -v
    delta = fdp.ConsumeIntInRange(-1000, 1000) / 1000.0
    return float(cur + delta)


def _deps_for_shape_vars(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    deps: Dict[str, List[str]] = {k: [] for k in (spec.get("shape_vars") or {}).keys()}
    for pname, p_spec in (spec.get("params") or {}).items():
        kind = p_spec.get("kind", "")
        if kind in ("tensor", "tensor_optional", "tensor_list"):
            for dim in (p_spec.get("shape_spec") or []):
                if isinstance(dim, str) and dim in deps and pname not in deps[dim]:
                    deps[dim].append(pname)
            for _rk, ss in (p_spec.get("shape_spec_by_rank") or {}).items():
                if isinstance(ss, list):
                    for dim in ss:
                        if isinstance(dim, str) and dim in deps and pname not in deps[dim]:
                            deps[dim].append(pname)
    return deps


def _resample_tensor_param(spec, pname, cfg, fdp):
    p_spec = (spec.get("params") or {}).get(pname) or {}
    shape_vars = cfg.get("_shape_vars", {})
    rank = cfg.get("_rank", 2)
    layout = cfg.get("_layout")
    dtype_str = cfg.get("_dtype_str", "float32")
    param_dtype = resolve_param_dtype(p_spec, dtype_str, fdp)
    return sample_tensor(p_spec, fdp, shape_vars, param_dtype, rank, layout, spec, pname)


def _reclamp_index_params(spec, cfg, fdp):
    """
    After shape mutation, re-clamp all index_input params to valid bounds.
    """
    p_reclamp = _env_float("P_INDEX_RECLAMP", 1.0)
    if fdp.ConsumeIntInRange(0, 999) >= int(p_reclamp * 1000):
        return

    params = spec.get("params") or {}
    for pname, p_spec in params.items():
        if not isinstance(p_spec, dict):
            continue
        if p_spec.get("semantic_role") != "index_input":
            continue
        if pname not in cfg or cfg[pname] is None:
            continue
        # Regenerate the index tensor with valid bounds
        cfg[pname] = _resample_tensor_param(spec, pname, cfg, fdp)


def mutate_rank(
    spec: Dict[str, Any],
    cfg: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, Any]:
    trial = deepcopy(cfg)
    test_ranks = spec.get("test_ranks") or [2, 3, 4]
    cur_rank = trial.get("_rank", 2)

    roll = fdp.ConsumeIntInRange(0, 99)

    if roll < 50:
        idx = -1
        for i, r in enumerate(test_ranks):
            if r == cur_rank:
                idx = i
                break
        if idx >= 0:
            cands = []
            if idx > 0:
                cands.append(test_ranks[idx - 1])
            if idx < len(test_ranks) - 1:
                cands.append(test_ranks[idx + 1])
            trial["_rank"] = fdp.PickValueInList(cands) if cands else fdp.PickValueInList(test_ranks)
        else:
            trial["_rank"] = fdp.PickValueInList(test_ranks)

    elif roll < 80:
        trial["_rank"] = _pick_other(fdp, test_ranks, cur_rank)

    elif roll < 95:
        mn, mx = min(test_ranks), max(test_ranks)
        cands = []
        if mn - 1 >= 1:
            cands.append(mn - 1)
        if mx + 1 <= 8:
            cands.append(mx + 1)
        if cands:
            trial["_rank"] = fdp.PickValueInList(cands)
    else:
        trial["_rank"] = fdp.PickValueInList([0, 6, 7])

    new_rank = trial["_rank"]
    trial["_layout"] = pick_layout(spec, new_rank, fdp)

    if trial["_layout"] and "data_format" in trial:
        trial["data_format"] = trial["_layout"]

    _ensure_outlier_rank_vars(spec, new_rank, trial.get("_shape_vars", {}), fdp)

    # Regenerate all tensor params
    params = spec.get("params") or {}
    for pname, p_spec in params.items():
        if not isinstance(p_spec, dict):
            continue
        kind = p_spec.get("kind", "")
        if kind == "tensor":
            trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)
        elif kind == "tensor_optional" and trial.get(pname) is not None:
            trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)

    return trial


def mutate_cfg(
    spec: Dict[str, Any],
    cfg: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    constraint_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    steps: int = 2,
    max_attempts_per_step: int = 6,
    p_type_mut: float = 0.35,
    p_shape_mut: float = 0.10,
    p_rank_mut: float = 0.15,
    p_layout_mut: float = 0.10,
):
    """
    Multi-step mutation for TF configs.
    Supports raw_ops and high-level API params identically.
    """
    if cfg is None:
        return None

    params_dict = spec.get("params") or {}
    param_names = list(params_dict.keys())
    if not param_names:
        return cfg

    deps = _deps_for_shape_vars(spec)
    cur = cfg
    steps = max(1, int(steps))

    p_layout_flip = _env_float("P_LAYOUT_FLIP", p_layout_mut)
    p_shape_ctrl = _env_float("P_SHAPE_CTRL_MUT", 0.10)

    for _step in range(steps):
        ok = False
        for _attempt in range(max_attempts_per_step):
            trial = deepcopy(cur)
            roll = fdp.ConsumeIntInRange(0, 999)

            threshold = 0

            # ── RANK MUTATION ────────────────────────────────────
            threshold += int(p_rank_mut * 1000)
            if roll < threshold:
                trial = mutate_rank(spec, trial, fdp)

            # ── LAYOUT MUTATION ──────────────────────────────────
            elif roll < threshold + int(p_layout_flip * 1000):
                threshold += int(p_layout_flip * 1000)
                layout_variants = spec.get("layout_variants") or {}
                cur_layout = trial.get("_layout")
                r = trial.get("_rank", 2)
                applicable = [
                    ln for ln, li in layout_variants.items()
                    if isinstance(li, dict) and r in (li.get("applies_to_ranks") or [])
                ]
                if applicable and cur_layout:
                    new_layout = _pick_other(fdp, applicable, cur_layout)
                    trial["_layout"] = new_layout
                    if "data_format" in trial:
                        trial["data_format"] = new_layout
                    primary = spec.get("primary_param")
                    if primary and primary in trial:
                        trial[primary] = _resample_tensor_param(spec, primary, trial, fdp)

            # ── SHAPE VAR MUTATION ───────────────────────────────
            elif roll < threshold + int(p_layout_flip * 1000) + int(p_shape_mut * 1000):
                sv = trial.get("_shape_vars", {})
                if isinstance(sv, dict) and sv:
                    var = fdp.PickValueInList(list(sv.keys()))
                    rng = (spec.get("shape_vars") or {}).get(var, [1, 16])
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        lo, hi = int(rng[0]), int(rng[1])
                        old_v = int(sv.get(var, lo))
                        new_v = fdp.ConsumeIntInRange(lo, hi)
                        if new_v == old_v and hi > lo:
                            new_v = lo if old_v != lo else hi
                        sv[var] = new_v
                        for pname in deps.get(var, []):
                            pk = params_dict.get(pname, {}).get("kind", "")
                            if pk == "tensor_optional" and trial.get(pname) is None:
                                continue
                            if pk in ("tensor", "tensor_optional"):
                                trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)
                    # Re-clamp index params after shape change
                    _reclamp_index_params(spec, trial, fdp)

            # ── PARAM VALUE/TYPE MUTATION ────────────────────────
            else:
                mutable = [
                    p for p in param_names
                    if params_dict.get(p, {}).get("kind", "")
                    not in ("string_optional", "dtype_enum", "")
                ]
                if not mutable:
                    mutable = param_names
                pname = fdp.PickValueInList(mutable)
                p_spec = params_dict.get(pname, {})
                kind = p_spec.get("kind", "")
                semantic_role = p_spec.get("semantic_role", "")
                val = trial.get(pname)

                do_type = fdp.ConsumeIntInRange(0, 999) < int(p_type_mut * 1000)

                if do_type:
                    # ── TYPE MUTATION ────────────────────────────
                    if kind == "tensor_optional":
                        if val is None:
                            trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)
                        else:
                            trial[pname] = None

                    elif kind == "tensor" and hasattr(val, 'dtype'):
                        test_dtypes = spec.get("test_dtype_choices") or ["float32"]
                        cur_dtype = trial.get("_dtype_str", "float32")
                        new_dtype = _pick_other(fdp, test_dtypes, cur_dtype)
                        trial["_dtype_str"] = new_dtype
                        dfa = p_spec.get("dtype_from_attr")
                        for pn, ps in params_dict.items():
                            if not isinstance(ps, dict):
                                continue
                            if dfa and ps.get("dtype_from_attr") == dfa:
                                if ps.get("kind") == "tensor" and pn in trial:
                                    trial[pn] = _resample_tensor_param(spec, pn, trial, fdp)
                                elif ps.get("kind") == "tensor_optional" and trial.get(pn) is not None:
                                    trial[pn] = _resample_tensor_param(spec, pn, trial, fdp)

                    elif kind == "bool":
                        trial[pname] = not bool(val) if val is not None else True

                else:
                    # ── VALUE MUTATION ───────────────────────────
                    if kind == "int":
                        trial[pname] = _mutate_int_value(
                            p_spec, fdp, int(val) if val is not None else 0
                        )

                    elif kind == "float":
                        trial[pname] = _mutate_float_value(
                            p_spec, fdp, float(val) if val is not None else 0.0
                        )

                    elif kind == "bool":
                        trial[pname] = not bool(val) if val is not None else True

                    elif kind == "enum":
                        if p_spec.get("values"):
                            trial[pname] = _pick_other(fdp, p_spec["values"], val)

                    elif kind == "int_list":
                        lst = list(val) if isinstance(val, list) else []
                        if lst:
                            idx = fdp.ConsumeIntInRange(0, len(lst) - 1)
                            lst[idx] = _mutate_int_value(p_spec, fdp, int(lst[idx]))
                            trial[pname] = lst
                        else:
                            trial[pname] = sample_int_list(p_spec, fdp)

                    elif kind == "float_list":
                        lst = list(val) if isinstance(val, list) else []
                        if lst:
                            idx = fdp.ConsumeIntInRange(0, len(lst) - 1)
                            lst[idx] = _mutate_float_value(p_spec, fdp, float(lst[idx]))
                            trial[pname] = lst
                        else:
                            trial[pname] = sample_float_list(p_spec, fdp)

                    elif kind in ("tensor", "tensor_optional"):
                        if semantic_role == "shape_control" and val is not None:
                            # Mutate one dimension in the shape_control value
                            if hasattr(val, 'numpy'):
                                arr = val.numpy().tolist()
                            elif isinstance(val, (list, tuple)):
                                arr = list(val)
                            else:
                                arr = [1]
                            if arr:
                                idx = fdp.ConsumeIntInRange(0, len(arr) - 1)
                                sv = trial.get("_shape_vars", {})
                                sv_vals = list(sv.values()) or [1, 2, 4, 8, 16]
                                arr[idx] = max(1, fdp.PickValueInList(sv_vals))
                                trial[pname] = tf.constant(arr, dtype=tf.int32)
                        elif val is not None and hasattr(val, 'shape'):
                            trial[pname] = mutate_tf_tensor_content(val, fdp)

            # Check constraints
            if constraint_func is None or constraint_func(trial):
                cur = trial
                ok = True
                break

        if not ok:
            continue

    return cur
