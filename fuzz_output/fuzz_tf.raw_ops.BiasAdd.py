#!/usr/bin/env python3
"""
Auto-generated atheris fuzz harness for TensorFlow raw_ops.
API: tf.raw_ops.BiasAdd
Op:  BiasAdd
Generated test_ranks: [1, 2, 3, 4, 5]
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

from utils.param_sampler import (
    gen_config_for_api,
    mutate_cfg,
    make_constraint_func,
)

# ============================================================
# Spec & constraints from YAML
# ============================================================

SPEC = {'category': 'BiasAdd',
 'shape_vars': {'C': [1, 64], 'N': [1, 8], 'L': [1, 32], 'H': [1, 16], 'W': [1, 16], 'D': [1, 16]},
 'layout_variants': {'NHWC': {'applies_to_ranks': [4],
                              'notes': 'Channel dimension is last; bias matches last dimension.'},
                     'NCHW': {'applies_to_ranks': [4],
                              'notes': 'Channel dimension is second (third-to-last overall for '
                                       '4-D); bias matches that dimension.'}},
 'constraints': ['value.ndim in (1, 2, 3, 4, 5)',
                 'value.dtype == bias.dtype',
                 'value.ndim == 4 or bias.shape[0] == value.shape[-1]',
                 "data_format in ('NHWC', 'NCHW')"],
 'op_family': 'bias_add',
 'rank_hints': {'marker': '__RANK_FROM_DOC__',
                'status': 'assigned',
                'rank_candidates': ['__RANK_TODO__'],
                'rank_any': True,
                'rank_min': None,
                'rank_max': None},
 'test_ranks': [1, 2, 3, 4, 5],
 'test_dtype_choices': ['float32', 'float64', 'int32', 'int64'],
 'api_name': 'tf.raw_ops.BiasAdd',
 'params': {'value': {'kind': 'tensor',
                      'origin': 'input',
                      'role': 'primary',
                      'semantic_role': 'data_tensor',
                      'dtype_from_attr': 'T',
                      'shape_spec': ['C'],
                      'shape_spec_by_rank': {'1': ['C'],
                                             '2': ['N', 'C'],
                                             '3': ['N', 'L', 'C'],
                                             '4': ['N', 'H', 'W', 'C'],
                                             '5': ['N', 'D', 'H', 'W', 'C']},
                      'shape_spec_by_rank_and_layout': {'4': {'NHWC': ['N', 'H', 'W', 'C'],
                                                              'NCHW': ['N', 'C', 'H', 'W']}},
                      'constraints_by_rank': {'4': ["(data_format == 'NHWC' and bias.shape[0] == "
                                                    "value.shape[-1]) or (data_format == 'NCHW' "
                                                    'and bias.shape[0] == value.shape[1])']}},
            'bias': {'kind': 'tensor',
                     'origin': 'input',
                     'role': 'aux',
                     'semantic_role': 'weight_tensor',
                     'dtype_from_attr': 'T',
                     'shape_spec': ['C']},
            'data_format': {'kind': 'enum',
                            'origin': 'attr',
                            'role': 'attr',
                            'semantic_role': 'layout_attr',
                            'values': ['NHWC', 'NCHW'],
                            'default': 'NHWC'}},
 'primary_param': 'value'}

CONSTRAINTS = ['value.ndim in (1, 2, 3, 4, 5)',
 'value.dtype == bias.dtype',
 'value.ndim == 4 or bias.shape[0] == value.shape[-1]',
 "data_format in ('NHWC', 'NCHW')"]


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
    """Call the tf.raw_ops.* API with sampled parameters."""
    api_name = SPEC.get("api_name", "")
    if not api_name:
        raise RuntimeError("SPEC missing api_name")

    # Resolve the function
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
        if pname in cfg and pname != "name":
            val = cfg[pname]
            # Skip None for optional params (TF doesn't accept None for most)
            kind = params[pname].get("kind", "")
            if val is None and kind in ("tensor_optional", "string_optional"):
                continue
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
        ValueError,
        TypeError,
        RuntimeError,
        AssertionError,
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
