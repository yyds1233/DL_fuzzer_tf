#!/usr/bin/env python3
"""
Auto-generated atheris fuzz harness for TensorFlow API.
API:      tf.image.resize
Op:       ResizeBilinear
Category: image
Ranks:    [3, 4]
Strategy: static_mapping
"""
import os
import sys
import importlib
import hashlib
import random
import atheris

# Must instrument before importing TF
# with atheris.instrument_imports():
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

SPEC = {'rank_hints': {'marker': '__RANK_FROM_DOC__',
                'status': 'assigned',
                'rank_candidates': [3, 4],
                'rank_any': False,
                'rank_min': None,
                'rank_max': 4},
 'test_dtype_choices': ['float32', 'float64', 'int32', 'int64'],
 'category': 'ResizeBilinear',
 'op_family': 'image_resize_hl',
 'layout_variants': {},
 'params': {'images': {'kind': 'tensor',
                       'origin': 'input',
                       'role': 'primary',
                       'semantic_role': 'data_tensor',
                       'dtype_from_attr': 'T',
                       'shape_spec': ['H', 'W', 'C'],
                       'shape_spec_by_rank': {'3': ['H', 'W', 'C'], '4': ['N', 'H', 'W', 'C']}},
            'size': {'kind': 'tensor',
                     'origin': 'input',
                     'role': 'attr',
                     'semantic_role': 'shape_control',
                     'dtype_choices': ['int32'],
                     'shape_spec': ['new_H', 'new_W']},
            'align_corners': {'kind': 'bool',
                              'origin': 'attr',
                              'role': 'attr',
                              'semantic_role': 'scalar_attr',
                              'default': False},
            'half_pixel_centers': {'kind': 'bool',
                                   'origin': 'attr',
                                   'role': 'attr',
                                   'semantic_role': 'scalar_attr',
                                   'default': False}},
 'api_category': 'image',
 'primary_param': 'images',
 'shape_vars': {'H': [1, 32],
                'W': [1, 32],
                'C': [1, 64],
                'N': [1, 8],
                'new_H': [1, 16],
                'new_W': [1, 16]},
 'test_ranks': [3, 4],
 'api_name': 'tf.image.resize',
 'constraints': ['images.ndim in (3, 4)'],
 '_resolve': {'strategy': 'static_mapping', 'is_raw_ops': False, 'raw_op_name': 'ResizeBilinear'}}

CONSTRAINTS = ['images.ndim in (3, 4)']


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
        raise RuntimeError(f"Invalid api_name: {api_name!r}")

    mod = importlib.import_module(mod_name)
    target = getattr(mod, func_name)

    # Build kwargs — only include actual API params, skip internal keys
    call_kwargs = {}
    params = SPEC.get("params", {})
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
    n_params = len(SPEC.get("params", {}))
    upper = max(1, min(MUT_STEPS_MAX, n_params))
    steps = fdp.ConsumeIntInRange(1, upper)

    # cfg = mutate_cfg(
    #     SPEC, cfg, fdp,
    #     constraint_func=constraint_func,
    #     steps=steps,
    #     max_attempts_per_step=max(1, MUT_ATTEMPTS),
    #     p_type_mut=max(0.0, min(1.0, P_TYPE_MUT)),
    #     p_shape_mut=max(0.0, min(1.0, P_SHAPE_MUT)),
    #     p_rank_mut=max(0.0, min(1.0, P_RANK_MUT)),
    #     p_layout_mut=max(0.0, min(1.0, P_LAYOUT_MUT)),
    # )

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
