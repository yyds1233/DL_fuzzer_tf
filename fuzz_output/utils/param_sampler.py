#!/usr/bin/env python3
"""
tf_param_sampler.py  –  TensorFlow raw_ops parameter sampler & mutator
for atheris-based fuzz harnesses.

==========================================================================
KEY DIFFERENCES FROM PYTORCH param_sampler.py
==========================================================================

1. RANK-AWARE SAMPLING
   - Reads `test_ranks` from YAML spec to pick a concrete rank per fuzz input.
   - Uses `shape_spec_by_rank[rank]` for the primary param's shape.
   - Falls back to `shape_spec` when no per-rank spec exists.

2. LAYOUT-AWARE SAMPLING
   - Reads `layout_variants` to decide whether to pick NHWC/NCHW.
   - Uses `shape_spec_by_rank_and_layout[rank][layout]` when available.
   - Sets `data_format` attr to match the chosen layout.

3. dtype_from_attr RESOLUTION
   - TF raw_ops params declare `dtype_from_attr: T` meaning "same type as attr T".
   - Sampler picks one dtype from `test_dtype_choices` and applies it to ALL
     params sharing the same type attr.

4. TF TENSOR CREATION
   - Uses `tf.constant`, `tf.random.normal`, `tf.random.uniform` etc.
   - Maps dtype strings to `tf.dtypes.*`.

5. RANK MUTATION
   - Core mutation: pick an adjacent rank from test_ranks.
   - Boundary exploration: small probability to try rank ± 1 outside test_ranks.
   - Out-of-range exploration: very small probability to try rank 0 or rank 6+.

6. INT_LIST FIXED ARITY
   - When an int_list has `len_range: [4, 4]`, mutation keeps the length fixed
     and only mutates individual elements.

==========================================================================
ENV KNOBS (same pattern as PyTorch version)
==========================================================================
  P_RANK_OUTLIER=0.05     probability of trying a rank outside test_ranks
  P_RANK_EXTREME=0.01     probability of trying rank 0 or rank 6+
  P_LAYOUT_FLIP=0.15      probability of flipping layout during mutation
  P_NONCONTIG=0.0         (TF tensors are normally contiguous, this is less relevant)
  ALLOW_EMPTY=0            enable 0-length dimension exploration
  P_EMPTY_DIM=0.01
"""
from __future__ import annotations

import math
import os
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import atheris

# TF import — lazy to allow syntax checking without TF installed
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
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
    "complex64": "complex64",
    "complex128": "complex128",
    "string": "string",
}


def resolve_tf_dtype(dtype_name: str):
    """Map dtype string to tf.DType."""
    mapped = _TF_DTYPE_MAP.get(dtype_name, dtype_name)
    return tf.dtypes.as_dtype(mapped)


def is_float_dtype(dtype_name: str) -> bool:
    return dtype_name in ("float32", "float64", "float16", "bfloat16")


def is_complex_dtype(dtype_name: str) -> bool:
    return dtype_name in ("complex64", "complex128")


def is_int_dtype(dtype_name: str) -> bool:
    return dtype_name in ("int8", "int16", "int32", "int64",
                          "uint8", "uint16", "uint32", "uint64")


# ════════════════════════════════════════════════════════════════
# 1. Rank selection
# ════════════════════════════════════════════════════════════════

def pick_rank(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> int:
    """
    Pick a concrete rank for this fuzz iteration.

    Strategy:
    - Most of the time: pick from test_ranks uniformly.
    - P_RANK_OUTLIER chance: pick an adjacent rank (test_ranks boundary ± 1).
    - P_RANK_EXTREME chance: pick rank 0 or rank 6-8.
    """
    test_ranks = spec.get("test_ranks") or []
    if not test_ranks:
        # Fallback: derive from shape_spec_by_rank or default
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

    # Extreme: rank 0 or rank 6-8
    if roll < int(p_extreme * 1000):
        extreme_ranks = [0, 6, 7, 8]
        # Filter out any that are already in test_ranks
        extreme_ranks = [r for r in extreme_ranks if r not in test_ranks]
        if extreme_ranks:
            return fdp.PickValueInList(extreme_ranks)

    # Outlier: boundary ± 1
    if roll < int((p_extreme + p_outlier) * 1000):
        min_r = min(test_ranks)
        max_r = max(test_ranks)
        outlier_ranks = []
        if min_r - 1 >= 1:
            outlier_ranks.append(min_r - 1)
        if max_r + 1 <= 8:
            outlier_ranks.append(max_r + 1)
        # Filter out existing
        outlier_ranks = [r for r in outlier_ranks if r not in test_ranks]
        if outlier_ranks:
            return fdp.PickValueInList(outlier_ranks)

    # Normal: pick from test_ranks
    return fdp.PickValueInList(test_ranks)


# ════════════════════════════════════════════════════════════════
# 2. Layout selection
# ════════════════════════════════════════════════════════════════

def pick_layout(
    spec: Dict[str, Any],
    rank: int,
    fdp: atheris.FuzzedDataProvider,
) -> Optional[str]:
    """
    Pick a layout variant for the given rank, or None if no layout applies.
    """
    layout_variants = spec.get("layout_variants") or {}
    if not layout_variants:
        return None

    applicable = []
    for layout_name, layout_info in layout_variants.items():
        if not isinstance(layout_info, dict):
            continue
        applies_to = layout_info.get("applies_to_ranks") or []
        if rank in applies_to:
            applicable.append(layout_name)

    if not applicable:
        return None

    return fdp.PickValueInList(applicable)


# ════════════════════════════════════════════════════════════════
# 3. Dtype selection
# ════════════════════════════════════════════════════════════════

def pick_dtype(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> str:
    """Pick a dtype string from test_dtype_choices."""
    choices = spec.get("test_dtype_choices") or ["float32"]
    return fdp.PickValueInList(choices)


# ════════════════════════════════════════════════════════════════
# 4. Shape vars sampling
# ════════════════════════════════════════════════════════════════

def gen_shape_vars(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, int]:
    """Sample shape_vars from their declared ranges."""
    shape_vars = {}
    for name, rng in (spec.get("shape_vars") or {}).items():
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = int(rng[0]), int(rng[1])
            val = fdp.ConsumeIntInRange(lo, hi)
            # Optional empty-dim exploration
            if _env_bool("ALLOW_EMPTY", False):
                p = _env_float("P_EMPTY_DIM", 0.0)
                n = name.lower()
                is_batch_or_channel = n in ("n", "c") or n.startswith("n_") or n.startswith("c_")
                if not is_batch_or_channel and p > 0:
                    if any(k in n for k in ("h", "w", "d", "l", "len", "seq")):
                        if fdp.ConsumeIntInRange(0, 999) < int(p * 1000):
                            val = 0
            shape_vars[name] = val
        else:
            shape_vars[name] = 1  # safe fallback
    return shape_vars


# ════════════════════════════════════════════════════════════════
# 5. Shape resolution
# ════════════════════════════════════════════════════════════════

def resolve_shape(
    shape_spec: List[Any],
    shape_vars: Dict[str, int],
) -> Tuple[int, ...]:
    """Convert shape_spec (list of var names / ints) to concrete tuple."""
    dims = []
    for dim in shape_spec:
        if isinstance(dim, str):
            if dim in shape_vars:
                dims.append(int(shape_vars[dim]))
            else:
                dims.append(1)  # undefined var fallback
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
    """
    Get the appropriate shape_spec for a parameter given rank & layout.

    Resolution order:
    1. shape_spec_by_rank_and_layout[rank][layout] (if layout is set)
    2. shape_spec_by_rank[rank]
    3. shape_spec (static fallback)
    """
    # Try rank-and-layout specific
    if layout:
        sbrl = p_spec.get("shape_spec_by_rank_and_layout") or {}
        rank_key = str(rank)
        if rank_key in sbrl and isinstance(sbrl[rank_key], dict):
            if layout in sbrl[rank_key]:
                return list(sbrl[rank_key][layout])

    # Try rank-specific
    sbr = p_spec.get("shape_spec_by_rank") or {}
    rank_key = str(rank)
    if rank_key in sbr:
        return list(sbr[rank_key])

    # Fallback to static shape_spec
    ss = p_spec.get("shape_spec") or []
    return list(ss)


def build_shape_spec_for_outlier_rank(
    pname: str,
    p_spec: Dict[str, Any],
    spec: Dict[str, Any],
    rank: int,
) -> List[str]:
    """
    Build a synthetic shape_spec for a rank that's NOT in shape_spec_by_rank.
    Used when rank mutation produces an outlier rank.

    Strategy: extend/truncate the nearest known rank's spec.
    """
    sbr = p_spec.get("shape_spec_by_rank") or {}
    if not sbr:
        # Pure fallback: generic dims
        dim_names = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
        return dim_names[:rank]

    # Find nearest known rank
    known_ranks = sorted(int(k) for k in sbr.keys())
    nearest = min(known_ranks, key=lambda k: abs(k - rank))
    base_spec = list(sbr[str(nearest)])

    if rank == len(base_spec):
        return base_spec
    elif rank < len(base_spec):
        # Truncate: keep first `rank` dims
        return base_spec[:rank]
    else:
        # Extend: repeat last dim pattern
        extra_needed = rank - len(base_spec)
        extra_dims = []
        for i in range(extra_needed):
            extra_dims.append(f"D_ext{i+1}")
        return base_spec + extra_dims


# ════════════════════════════════════════════════════════════════
# 6. TF tensor creation
# ════════════════════════════════════════════════════════════════

def create_tf_tensor(
    shape: Tuple[int, ...],
    dtype_str: str,
    fdp: atheris.FuzzedDataProvider,
):
    """Create a TF tensor with random content."""
    tf_dtype = resolve_tf_dtype(dtype_str)

    if is_float_dtype(dtype_str):
        # Use numpy for generation, then convert
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

    # Fallback
    return tf.zeros(shape, dtype=tf_dtype)


def mutate_tf_tensor_content(
    tensor,
    fdp: atheris.FuzzedDataProvider,
):
    """Create a new tensor with same shape & dtype but different content."""
    shape = tuple(tensor.shape)
    dtype_str = tensor.dtype.name

    new_t = create_tf_tensor(shape, dtype_str, fdp)

    # Small probability: inject special values
    if is_float_dtype(dtype_str) and tensor.shape.num_elements() > 0:
        if fdp.ConsumeIntInRange(0, 99) == 0:
            arr = new_t.numpy().copy()
            arr.flat[0] = float("nan")
            new_t = tf.constant(arr, dtype=tensor.dtype)
        elif fdp.ConsumeIntInRange(0, 99) == 0:
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
    lo, hi = float(lo), float(hi)
    steps = int(p_spec.get("steps", 1000))
    idx = fdp.ConsumeIntInRange(0, steps)
    return lo + (hi - lo) * (idx / steps)


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


def sample_string_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    return p_spec.get("default", None)


def sample_dtype_enum(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    """Sample a TF dtype enum value."""
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
    Sample a TF tensor, using rank-aware shape resolution.
    """
    # Get appropriate shape_spec
    shape_spec = get_shape_spec_for_param(pname, p_spec, spec, rank, layout)

    if not shape_spec or (len(shape_spec) == 1 and shape_spec[0] == "TODO_SHAPE"):
        # Emergency fallback
        shape_spec = build_shape_spec_for_outlier_rank(pname, p_spec, spec, rank)

    # Ensure shape_vars has all needed vars (add missing ones as 1)
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
# 8. Top-level config generation
# ════════════════════════════════════════════════════════════════

def gen_config_for_api(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, Any]:
    """
    Generate a complete parameter configuration for one TF raw_ops API call.

    Returns cfg dict with:
    - Each param name → sampled value
    - _shape_vars: the sampled shape vars
    - _rank: the chosen rank
    - _layout: the chosen layout (or None)
    - _dtype_str: the chosen dtype string
    """
    cfg: Dict[str, Any] = {}

    # 1. Pick rank
    rank = pick_rank(spec, fdp)
    cfg["_rank"] = rank

    # 2. Pick layout
    layout = pick_layout(spec, rank, fdp)
    cfg["_layout"] = layout

    # 3. Pick dtype for shared type attrs
    dtype_str = pick_dtype(spec, fdp)
    cfg["_dtype_str"] = dtype_str

    # 4. Sample shape_vars
    shape_vars = gen_shape_vars(spec, fdp)
    # Ensure extended vars exist for outlier ranks
    _ensure_outlier_rank_vars(spec, rank, shape_vars, fdp)
    cfg["_shape_vars"] = shape_vars

    # 5. Sample each param
    params = spec.get("params") or {}
    for pname, p_spec in params.items():
        if not isinstance(p_spec, dict):
            continue
        kind = p_spec.get("kind", "")

        # Determine this param's dtype
        param_dtype = dtype_str
        dfa = p_spec.get("dtype_from_attr")
        if dfa:
            # All params sharing the same type attr get the same dtype
            param_dtype = dtype_str
        elif "dtype_choices" in p_spec:
            param_dtype = fdp.PickValueInList(p_spec["dtype_choices"])

        # Handle layout-controlled params
        if pname == "data_format" and layout:
            cfg[pname] = layout
            continue

        # Sample by kind
        if kind in ("tensor", "tensor_optional"):
            if kind == "tensor":
                val = sample_tensor(p_spec, fdp, shape_vars, param_dtype,
                                    rank, layout, spec, pname)
            else:
                val = sample_tensor_optional(p_spec, fdp, shape_vars, param_dtype,
                                             rank, layout, spec, pname)
            cfg[pname] = val

        elif kind == "tensor_list":
            len_lo, len_hi = p_spec.get("len_range", [1, 4])
            length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
            elems = []
            for _ in range(length):
                elems.append(sample_tensor(p_spec, fdp, shape_vars, param_dtype,
                                           rank, layout, spec, pname))
            cfg[pname] = elems

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
        elif kind == "string_optional":
            cfg[pname] = sample_string_optional(p_spec, fdp)
        elif kind == "dtype_enum":
            cfg[pname] = sample_dtype_enum(p_spec, fdp)
        else:
            # Skip unknown kinds (name, meta, etc.)
            pass

    # 6. Remove 'name' param (TF ops don't need it for testing)
    cfg.pop("name", None)

    return cfg


def _ensure_outlier_rank_vars(
    spec: Dict[str, Any],
    rank: int,
    shape_vars: Dict[str, int],
    fdp: atheris.FuzzedDataProvider,
) -> None:
    """
    If rank is an outlier (not in test_ranks), we may need extra shape vars.
    Pre-generate them so resolve_shape won't fail.
    """
    primary = spec.get("primary_param")
    params = spec.get("params") or {}
    p = params.get(primary) if primary else None
    if not isinstance(p, dict):
        return

    sbr = p.get("shape_spec_by_rank") or {}
    rank_key = str(rank)
    if rank_key in sbr:
        return  # We have a spec for this rank

    # Build synthetic spec and ensure vars exist
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
    """
    Build a constraint checker from YAML spec.
    Handles both top-level constraints and per-rank constraints_by_rank.
    """
    top_constraints = spec.get("constraints") or []
    if not isinstance(top_constraints, list):
        top_constraints = []

    # Per-rank constraints: from primary_param.constraints_by_rank
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
        shape_vars = cfg.get("_shape_vars", {})
        rank = cfg.get("_rank")

        # Build local namespace for eval
        locs = dict(shape_vars)
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            locs[k] = v

        # Inject tf and math
        locs["tf"] = tf
        locs["math"] = math

        # Check top-level constraints
        for expr in top_constraints:
            try:
                if not eval(expr, {"__builtins__": {}}, locs):
                    return False
            except Exception:
                return False

        # Check per-rank constraints
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
# 10. Mutation
# ════════════════════════════════════════════════════════════════

def _pick_other(fdp, vals, cur):
    """Pick a value from vals that's different from cur (best effort)."""
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
        if isinstance(cand, (list, tuple)):
            return int(cand[0]) if cand else cur
        return int(cand)
    lo, hi = p_spec.get("range", [0, 10])
    delta = fdp.ConsumeIntInRange(-3, 3)
    v = int(cur) + delta
    return max(int(lo), min(int(hi), v))


def _mutate_float_value(p_spec, fdp, cur: float) -> float:
    specials = [0.0, 1.0, -1.0, 1e-12, 1e-6, 1e-3, 1e3]
    if fdp.ConsumeBool():
        v = float(fdp.PickValueInList(specials))
        return v if v != cur else -v
    delta = fdp.ConsumeIntInRange(-1000, 1000) / 1000.0
    return float(cur + delta)


def _deps_for_shape_vars(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Map shape_var name → list of param names that use it in their shape_spec."""
    deps: Dict[str, List[str]] = {k: [] for k in (spec.get("shape_vars") or {}).keys()}
    for pname, p_spec in (spec.get("params") or {}).items():
        kind = p_spec.get("kind", "")
        if kind in ("tensor", "tensor_optional", "tensor_list"):
            # Check all possible shape_specs
            for dim in (p_spec.get("shape_spec") or []):
                if isinstance(dim, str) and dim in deps:
                    deps[dim].append(pname)
            # Also check shape_spec_by_rank entries
            for _rk, ss in (p_spec.get("shape_spec_by_rank") or {}).items():
                if isinstance(ss, list):
                    for dim in ss:
                        if isinstance(dim, str) and dim in deps:
                            if pname not in deps[dim]:
                                deps[dim].append(pname)
    return deps


def _resample_tensor_param(spec, pname, cfg, fdp):
    """Re-create a tensor param using current cfg state."""
    p_spec = (spec.get("params") or {}).get(pname) or {}
    shape_vars = cfg.get("_shape_vars", {})
    rank = cfg.get("_rank", 2)
    layout = cfg.get("_layout")
    dtype_str = cfg.get("_dtype_str", "float32")
    return sample_tensor(p_spec, fdp, shape_vars, dtype_str, rank, layout, spec, pname)


def mutate_rank(
    spec: Dict[str, Any],
    cfg: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, Any]:
    """
    Rank mutation: change the rank and regenerate all tensor params accordingly.

    Strategies:
    - 50%: pick adjacent rank in test_ranks
    - 30%: pick random rank from test_ranks
    - 15%: pick boundary ± 1
    - 5%: pick extreme (rank 0 or 6+)
    """
    trial = deepcopy(cfg)
    test_ranks = spec.get("test_ranks") or [2, 3, 4]
    cur_rank = trial.get("_rank", 2)

    roll = fdp.ConsumeIntInRange(0, 99)

    if roll < 50:
        # Adjacent in test_ranks
        idx = -1
        for i, r in enumerate(test_ranks):
            if r == cur_rank:
                idx = i
                break
        if idx >= 0:
            candidates = []
            if idx > 0:
                candidates.append(test_ranks[idx - 1])
            if idx < len(test_ranks) - 1:
                candidates.append(test_ranks[idx + 1])
            if candidates:
                trial["_rank"] = fdp.PickValueInList(candidates)
            else:
                trial["_rank"] = fdp.PickValueInList(test_ranks)
        else:
            trial["_rank"] = fdp.PickValueInList(test_ranks)

    elif roll < 80:
        # Random from test_ranks
        trial["_rank"] = _pick_other(fdp, test_ranks, cur_rank)

    elif roll < 95:
        # Boundary ± 1
        mn, mx = min(test_ranks), max(test_ranks)
        candidates = []
        if mn - 1 >= 1:
            candidates.append(mn - 1)
        if mx + 1 <= 8:
            candidates.append(mx + 1)
        if candidates:
            trial["_rank"] = fdp.PickValueInList(candidates)

    else:
        # Extreme
        trial["_rank"] = fdp.PickValueInList([0, 6, 7])

    new_rank = trial["_rank"]

    # Re-pick layout for new rank
    trial["_layout"] = pick_layout(spec, new_rank, fdp)

    # Update data_format if layout changed
    if trial["_layout"] and "data_format" in trial:
        trial["data_format"] = trial["_layout"]

    # Ensure shape vars exist for new rank
    _ensure_outlier_rank_vars(spec, new_rank, trial.get("_shape_vars", {}), fdp)

    # Regenerate all tensor params
    params = spec.get("params") or {}
    for pname, p_spec in params.items():
        if not isinstance(p_spec, dict):
            continue
        kind = p_spec.get("kind", "")
        if kind == "tensor":
            trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)
        elif kind == "tensor_optional":
            if trial.get(pname) is not None:
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
    Multi-step mutation for TF raw_ops configs.

    Extra mutation types vs PyTorch:
    - Rank mutation: changes rank → regenerates all tensors
    - Layout mutation: flips layout → regenerates primary tensor
    """
    if cfg is None:
        return None

    params = list((spec.get("params") or {}).keys())
    if not params:
        return cfg

    deps = _deps_for_shape_vars(spec)
    cur = cfg
    steps = max(1, int(steps))

    p_layout_flip = _env_float("P_LAYOUT_FLIP", p_layout_mut)

    for _step in range(steps):
        ok = False
        for _attempt in range(max_attempts_per_step):
            trial = deepcopy(cur)

            # Decide mutation type
            roll = fdp.ConsumeIntInRange(0, 999)

            # ---- RANK MUTATION ----
            if roll < int(p_rank_mut * 1000):
                trial = mutate_rank(spec, trial, fdp)

            # ---- LAYOUT MUTATION ----
            elif roll < int((p_rank_mut + p_layout_flip) * 1000):
                layout_variants = spec.get("layout_variants") or {}
                cur_layout = trial.get("_layout")
                rank = trial.get("_rank", 2)
                applicable = []
                for ln, li in layout_variants.items():
                    if isinstance(li, dict) and rank in (li.get("applies_to_ranks") or []):
                        applicable.append(ln)
                if applicable and cur_layout:
                    new_layout = _pick_other(fdp, applicable, cur_layout)
                    trial["_layout"] = new_layout
                    if "data_format" in trial:
                        trial["data_format"] = new_layout
                    # Regenerate primary tensor with new layout
                    primary = spec.get("primary_param")
                    if primary and primary in trial:
                        trial[primary] = _resample_tensor_param(spec, primary, trial, fdp)

            # ---- SHAPE VAR MUTATION ----
            elif roll < int((p_rank_mut + p_layout_flip + p_shape_mut) * 1000):
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
                        # Resample dependent tensors
                        for pname in deps.get(var, []):
                            pk = (spec.get("params") or {}).get(pname, {}).get("kind", "")
                            if pk == "tensor_optional" and trial.get(pname) is None:
                                continue
                            if pk in ("tensor", "tensor_optional"):
                                trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)

            # ---- PARAM VALUE/TYPE MUTATION ----
            else:
                # Pick a mutable param (skip meta params)
                mutable = [p for p in params
                           if (spec.get("params") or {}).get(p, {}).get("kind", "")
                           not in ("string_optional", "dtype_enum", "")]
                if not mutable:
                    mutable = params
                pname = fdp.PickValueInList(mutable)
                p_spec = (spec.get("params") or {}).get(pname, {})
                kind = p_spec.get("kind", "")
                val = trial.get(pname)

                do_type = fdp.ConsumeIntInRange(0, 999) < int(p_type_mut * 1000)

                if do_type:
                    # TYPE MUTATION
                    if kind == "tensor_optional":
                        if val is None:
                            trial[pname] = _resample_tensor_param(spec, pname, trial, fdp)
                        else:
                            trial[pname] = None

                    elif kind == "tensor" and hasattr(val, 'dtype'):
                        # Dtype mutation for tensor
                        test_dtypes = spec.get("test_dtype_choices") or ["float32"]
                        cur_dtype = trial.get("_dtype_str", "float32")
                        new_dtype = _pick_other(fdp, test_dtypes, cur_dtype)
                        trial["_dtype_str"] = new_dtype
                        # Regenerate this tensor and all same-type-attr tensors
                        for pn, ps in (spec.get("params") or {}).items():
                            if isinstance(ps, dict) and ps.get("dtype_from_attr") == p_spec.get("dtype_from_attr"):
                                if ps.get("kind") == "tensor" and pn in trial:
                                    trial[pn] = _resample_tensor_param(spec, pn, trial, fdp)
                                elif ps.get("kind") == "tensor_optional" and trial.get(pn) is not None:
                                    trial[pn] = _resample_tensor_param(spec, pn, trial, fdp)

                    elif kind == "bool":
                        trial[pname] = not bool(val) if val is not None else True

                else:
                    # VALUE MUTATION
                    if kind == "int":
                        trial[pname] = _mutate_int_value(p_spec, fdp, int(val) if val is not None else 0)

                    elif kind == "float":
                        trial[pname] = _mutate_float_value(p_spec, fdp, float(val) if val is not None else 0.0)

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

                    elif kind in ("tensor", "tensor_optional"):
                        if val is not None and hasattr(val, 'shape'):
                            trial[pname] = mutate_tf_tensor_content(val, fdp)

            # Check constraints
            if constraint_func is None or constraint_func(trial):
                cur = trial
                ok = True
                break

        # If this step failed, keep previous cfg
        if not ok:
            continue

    return cur
