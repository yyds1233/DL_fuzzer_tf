#!/usr/bin/env python3
"""
tf_api_resolver.py  –  Resolve high-level TF APIs to underlying raw_ops.

For a given high-level API (e.g., tf.nn.conv2d), this module:
  1. Identifies the underlying raw_op(s) (e.g., Conv2D)
  2. Builds a parameter mapping (high-level param → raw_op param)
  3. Extracts the raw_op's OpDef for downstream use

Resolution strategies (tried in order):
  T1: Static mapping table (hand-curated, highest confidence)
  T2: Name heuristics (e.g., conv2d → Conv2D)
  T3: Source code inspection (grep for gen_*_ops / raw_ops calls)
  T4: LLM fallback (ask LLM to identify the underlying raw_op)

==========================================================================
USAGE
==========================================================================

  from tf_api_resolver import TFApiResolver

  resolver = TFApiResolver()  # optional: pass llm_client for T4
  result = resolver.resolve("tf.nn.conv2d")

  result.raw_op_name       # "Conv2D"
  result.param_mapping     # {"input": "input", "filters": "filter", ...}
  result.confidence        # 0.95
  result.strategy          # "static_mapping"
  result.api_category      # "nn"

==========================================================================
STANDALONE CLI
==========================================================================

  python tf_api_resolver.py --api_list apis.txt --out_dir ./resolved
  python tf_api_resolver.py --api "tf.nn.conv2d"
  python tf_api_resolver.py --api_list apis.txt --llm_base_url http://localhost:11434/v1 --llm_model qwen2.5:72b
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ══════════════════════════════════════════════════════════════════
# Resolution result
# ══════════════════════════════════════════════════════════════════

@dataclass
class ResolveResult:
    """Result of resolving a high-level API to its underlying raw_op."""
    api_name: str                                    # e.g. "tf.nn.conv2d"
    raw_op_name: Optional[str] = None                # e.g. "Conv2D"
    raw_op_names: List[str] = field(default_factory=list)  # if multiple ops
    param_mapping: Dict[str, str] = field(default_factory=dict)  # high→raw
    inverse_mapping: Dict[str, str] = field(default_factory=dict)  # raw→high
    confidence: float = 0.0
    strategy: str = "none"                           # static/heuristic/source/llm
    api_category: str = "unknown"                    # nn/math/linalg/image/signal/...
    api_module: str = ""                             # e.g. "tf.nn"
    is_raw_ops: bool = False
    is_class: bool = False
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════
# T1: Static mapping table
# ══════════════════════════════════════════════════════════════════

# Format: "tf.module.func" → {
#   "raw_op": "OpName",
#   "params": { "high_level_param": "raw_op_param", ... },
#   "category": "nn" | "math" | ...
# }
#
# Only params that differ in name need to be listed.
# Identical-name params are auto-mapped.

STATIC_MAPPING: Dict[str, Dict[str, Any]] = {
    # ── tf.nn ────────────────────────────────────────────────────
    "tf.nn.conv2d": {
        "raw_op": "Conv2D",
        "params": {"filters": "filter"},
        "category": "nn",
    },
    "tf.nn.conv1d": {
        "raw_op": "Conv2D",  # TF internally expands conv1d → conv2d
        "params": {"filters": "filter"},
        "category": "nn",
        "notes": ["conv1d internally expands to Conv2D via reshape"],
    },
    "tf.nn.conv3d": {
        "raw_op": "Conv3D",
        "params": {"filters": "filter"},
        "category": "nn",
    },
    "tf.nn.conv2d_transpose": {
        "raw_op": "Conv2DBackpropInput",
        "params": {"filters": "filter", "output_shape": "input_sizes"},
        "category": "nn",
    },
    "tf.nn.conv3d_transpose": {
        "raw_op": "Conv3DBackpropInputV2",
        "params": {"filters": "filter", "output_shape": "input_sizes"},
        "category": "nn",
    },
    "tf.nn.depthwise_conv2d": {
        "raw_op": "DepthwiseConv2dNative",
        "params": {"filters": "filter"},  # TF 2.x uses 'filters' in high-level
        "category": "nn",
    },
    "tf.nn.max_pool": {
        "raw_op": "MaxPoolV2",
        "params": {},
        "category": "nn",
    },
    "tf.nn.max_pool2d": {
        "raw_op": "MaxPoolV2",
        "params": {},
        "category": "nn",
    },
    "tf.nn.avg_pool": {
        "raw_op": "AvgPool",
        "params": {"value": "value"},
        "category": "nn",
    },
    "tf.nn.avg_pool2d": {
        "raw_op": "AvgPool",
        "params": {},
        "category": "nn",
    },
    "tf.nn.max_pool3d": {
        "raw_op": "MaxPool3D",
        "params": {},
        "category": "nn",
    },
    "tf.nn.avg_pool3d": {
        "raw_op": "AvgPool3D",
        "params": {},
        "category": "nn",
    },
    "tf.nn.relu": {
        "raw_op": "Relu",
        "params": {},
        "category": "nn",
    },
    "tf.nn.relu6": {
        "raw_op": "Relu6",
        "params": {},
        "category": "nn",
    },
    "tf.nn.leaky_relu": {
        "raw_op": "LeakyRelu",
        "params": {},
        "category": "nn",
    },
    "tf.nn.elu": {
        "raw_op": "Elu",
        "params": {},
        "category": "nn",
    },
    "tf.nn.selu": {
        "raw_op": "Selu",
        "params": {},
        "category": "nn",
    },
    "tf.nn.softmax": {
        "raw_op": "Softmax",
        "params": {},
        "category": "nn",
    },
    "tf.nn.log_softmax": {
        "raw_op": "LogSoftmax",
        "params": {},
        "category": "nn",
    },
    "tf.nn.sigmoid": {
        "raw_op": "Sigmoid",
        "params": {},
        "category": "nn",
    },
    "tf.nn.tanh": {
        "raw_op": "Tanh",
        "params": {},
        "category": "nn",
    },
    "tf.nn.bias_add": {
        "raw_op": "BiasAdd",
        "params": {},
        "category": "nn",
    },
    "tf.nn.batch_normalization": {
        "raw_op": "FusedBatchNormV3",
        "params": {"x": "x", "mean": "mean", "variance": "variance",
                   "offset": "offset", "scale": "scale"},
        "category": "nn",
    },
    "tf.nn.dropout": {
        "raw_op": None,  # Composite: uses random + mul
        "params": {},
        "category": "nn",
        "notes": ["dropout is composite: random_uniform + floor + div + mul"],
    },
    "tf.nn.l2_normalize": {
        "raw_op": "L2Loss",
        "params": {},
        "category": "nn",
        "notes": ["l2_normalize is composite"],
    },
    "tf.nn.embedding_lookup": {
        "raw_op": "GatherV2",
        "params": {"params": "params", "ids": "indices"},
        "category": "nn",
    },
    "tf.nn.softmax_cross_entropy_with_logits": {
        "raw_op": "SoftmaxCrossEntropyWithLogits",
        "params": {},
        "category": "nn",
    },
    "tf.nn.sparse_softmax_cross_entropy_with_logits": {
        "raw_op": "SparseSoftmaxCrossEntropyWithLogits",
        "params": {},
        "category": "nn",
    },

    # ── tf.math ──────────────────────────────────────────────────
    "tf.math.add": {
        "raw_op": "AddV2",
        "params": {},
        "category": "math",
    },
    "tf.math.subtract": {
        "raw_op": "Sub",
        "params": {},
        "category": "math",
    },
    "tf.math.multiply": {
        "raw_op": "Mul",
        "params": {},
        "category": "math",
    },
    "tf.math.divide": {
        "raw_op": "RealDiv",
        "params": {},
        "category": "math",
    },
    "tf.math.reduce_sum": {
        "raw_op": "Sum",
        "params": {"input_tensor": "input", "axis": "reduction_indices"},
        "category": "math",
    },
    "tf.math.reduce_mean": {
        "raw_op": "Mean",
        "params": {"input_tensor": "input", "axis": "reduction_indices"},
        "category": "math",
    },
    "tf.math.reduce_max": {
        "raw_op": "Max",
        "params": {"input_tensor": "input", "axis": "reduction_indices"},
        "category": "math",
    },
    "tf.math.reduce_min": {
        "raw_op": "Min",
        "params": {"input_tensor": "input", "axis": "reduction_indices"},
        "category": "math",
    },
    "tf.math.reduce_prod": {
        "raw_op": "Prod",
        "params": {"input_tensor": "input", "axis": "reduction_indices"},
        "category": "math",
    },
    "tf.math.abs": {
        "raw_op": "Abs",
        "params": {},
        "category": "math",
    },
    "tf.math.negative": {
        "raw_op": "Neg",
        "params": {},
        "category": "math",
    },
    "tf.math.exp": {
        "raw_op": "Exp",
        "params": {},
        "category": "math",
    },
    "tf.math.log": {
        "raw_op": "Log",
        "params": {},
        "category": "math",
    },
    "tf.math.sqrt": {
        "raw_op": "Sqrt",
        "params": {},
        "category": "math",
    },
    "tf.math.square": {
        "raw_op": "Square",
        "params": {},
        "category": "math",
    },
    "tf.math.pow": {
        "raw_op": "Pow",
        "params": {},
        "category": "math",
    },
    "tf.math.matmul": {
        "raw_op": "BatchMatMulV2",
        "params": {"a": "x", "b": "y"},
        "category": "math",
    },
    "tf.math.argmax": {
        "raw_op": "ArgMax",
        "params": {"input": "input", "axis": "dimension"},
        "category": "math",
    },
    "tf.math.argmin": {
        "raw_op": "ArgMin",
        "params": {"input": "input", "axis": "dimension"},
        "category": "math",
    },
    "tf.math.cumsum": {
        "raw_op": "Cumsum",
        "params": {},
        "category": "math",
    },
    "tf.math.top_k": {
        "raw_op": "TopKV2",
        "params": {},
        "category": "math",
    },
    "tf.math.ceil": {
        "raw_op": "Ceil",
        "params": {},
        "category": "math",
    },
    "tf.math.floor": {
        "raw_op": "Floor",
        "params": {},
        "category": "math",
    },
    "tf.math.round": {
        "raw_op": "Round",
        "params": {},
        "category": "math",
    },
    "tf.math.sign": {
        "raw_op": "Sign",
        "params": {},
        "category": "math",
    },
    "tf.math.maximum": {
        "raw_op": "Maximum",
        "params": {},
        "category": "math",
    },
    "tf.math.minimum": {
        "raw_op": "Minimum",
        "params": {},
        "category": "math",
    },
    "tf.math.logical_and": {
        "raw_op": "LogicalAnd",
        "params": {},
        "category": "math",
    },
    "tf.math.logical_or": {
        "raw_op": "LogicalOr",
        "params": {},
        "category": "math",
    },
    "tf.math.logical_not": {
        "raw_op": "LogicalNot",
        "params": {},
        "category": "math",
    },
    "tf.math.equal": {
        "raw_op": "Equal",
        "params": {},
        "category": "math",
    },
    "tf.math.not_equal": {
        "raw_op": "NotEqual",
        "params": {},
        "category": "math",
    },
    "tf.math.greater": {
        "raw_op": "Greater",
        "params": {},
        "category": "math",
    },
    "tf.math.less": {
        "raw_op": "Less",
        "params": {},
        "category": "math",
    },

    # ── tf.linalg ────────────────────────────────────────────────
    "tf.linalg.matmul": {
        "raw_op": "BatchMatMulV2",
        "params": {"a": "x", "b": "y"},
        "category": "linalg",
    },
    "tf.linalg.inv": {
        "raw_op": "MatrixInverse",
        "params": {"input": "input"},
        "category": "linalg",
    },
    "tf.linalg.det": {
        "raw_op": "MatrixDeterminant",
        "params": {"input": "input"},
        "category": "linalg",
    },
    "tf.linalg.diag": {
        "raw_op": "MatrixDiagV3",
        "params": {"diagonal": "diagonal"},
        "category": "linalg",
    },
    "tf.linalg.band_part": {
        "raw_op": "MatrixBandPart",
        "params": {},
        "category": "linalg",
    },
    "tf.linalg.norm": {
        "raw_op": None,  # composite
        "params": {},
        "category": "linalg",
        "notes": ["linalg.norm is composite"],
    },

    # ── tf (top-level) ──────────────────────────────────────────
    "tf.matmul": {
        "raw_op": "BatchMatMulV2",
        "params": {"a": "x", "b": "y"},
        "category": "math",
    },
    "tf.reshape": {
        "raw_op": "Reshape",
        "params": {"tensor": "tensor", "shape": "shape"},
        "category": "core",
    },
    "tf.transpose": {
        "raw_op": "Transpose",
        "params": {"a": "x", "perm": "perm"},
        "category": "core",
    },
    "tf.concat": {
        "raw_op": "ConcatV2",
        "params": {"values": "values", "axis": "axis"},
        "category": "core",
    },
    "tf.split": {
        "raw_op": "SplitV",
        "params": {"value": "value", "num_or_size_splits": "size_splits",
                   "axis": "split_dim"},
        "category": "core",
    },
    "tf.gather": {
        "raw_op": "GatherV2",
        "params": {},
        "category": "core",
    },
    "tf.gather_nd": {
        "raw_op": "GatherNd",
        "params": {},
        "category": "core",
    },
    "tf.scatter_nd": {
        "raw_op": "ScatterNd",
        "params": {},
        "category": "core",
    },
    "tf.one_hot": {
        "raw_op": "OneHot",
        "params": {},
        "category": "core",
    },
    "tf.where": {
        "raw_op": "SelectV2",
        "params": {"condition": "condition"},
        "category": "core",
    },
    "tf.cast": {
        "raw_op": "Cast",
        "params": {"x": "x", "dtype": "DstT"},
        "category": "core",
    },
    "tf.expand_dims": {
        "raw_op": "ExpandDims",
        "params": {"input": "input", "axis": "dim"},
        "category": "core",
    },
    "tf.squeeze": {
        "raw_op": "Squeeze",
        "params": {"input": "input", "axis": "squeeze_dims"},
        "category": "core",
    },
    "tf.tile": {
        "raw_op": "Tile",
        "params": {},
        "category": "core",
    },
    "tf.pad": {
        "raw_op": "PadV2",
        "params": {"tensor": "input", "paddings": "paddings"},
        "category": "core",
    },
    "tf.slice": {
        "raw_op": "Slice",
        "params": {},
        "category": "core",
    },
    "tf.stack": {
        "raw_op": "Pack",
        "params": {"values": "values", "axis": "axis"},
        "category": "core",
    },
    "tf.unstack": {
        "raw_op": "Unpack",
        "params": {"value": "value", "axis": "axis"},
        "category": "core",
    },
    "tf.clip_by_value": {
        "raw_op": "ClipByValue",
        "params": {"t": "t", "clip_value_min": "clip_value_min",
                   "clip_value_max": "clip_value_max"},
        "category": "core",
    },
    "tf.identity": {
        "raw_op": "Identity",
        "params": {},
        "category": "core",
    },
    "tf.zeros_like": {
        "raw_op": "ZerosLike",
        "params": {},
        "category": "core",
    },
    "tf.ones_like": {
        "raw_op": "OnesLike",
        "params": {},
        "category": "core",
    },
    "tf.fill": {
        "raw_op": "Fill",
        "params": {"dims": "dims", "value": "value"},
        "category": "core",
    },
    "tf.shape": {
        "raw_op": "Shape",
        "params": {},
        "category": "core",
    },
    "tf.rank": {
        "raw_op": "Rank",
        "params": {},
        "category": "core",
    },
    "tf.size": {
        "raw_op": "Size",
        "params": {},
        "category": "core",
    },
    "tf.sort": {
        "raw_op": None,  # composite
        "params": {},
        "category": "core",
        "notes": ["tf.sort is composite (TopKV2 + Reverse)"],
    },

    # ── tf.image ─────────────────────────────────────────────────
    "tf.image.resize": {
        "raw_op": "ResizeBilinear",
        "params": {"images": "images", "size": "size"},
        "category": "image",
        "notes": ["resize dispatches to different raw_ops based on method arg"],
    },
    "tf.image.resize_with_crop_or_pad": {
        "raw_op": None,  # composite
        "params": {},
        "category": "image",
    },
    "tf.image.flip_left_right": {
        "raw_op": "ReverseV2",
        "params": {},
        "category": "image",
    },
    "tf.image.flip_up_down": {
        "raw_op": "ReverseV2",
        "params": {},
        "category": "image",
    },
    "tf.image.rot90": {
        "raw_op": None,  # composite
        "params": {},
        "category": "image",
    },
    "tf.image.rgb_to_grayscale": {
        "raw_op": None,  # composite
        "params": {},
        "category": "image",
    },
    "tf.image.per_image_standardization": {
        "raw_op": None,
        "params": {},
        "category": "image",
    },

    # ── tf.signal ────────────────────────────────────────────────
    "tf.signal.fft": {
        "raw_op": "FFT",
        "params": {},
        "category": "signal",
    },
    "tf.signal.ifft": {
        "raw_op": "IFFT",
        "params": {},
        "category": "signal",
    },
    "tf.signal.rfft": {
        "raw_op": "RFFT",
        "params": {},
        "category": "signal",
    },
    "tf.signal.stft": {
        "raw_op": None,  # composite
        "params": {},
        "category": "signal",
        "notes": ["stft is composite (window + rfft)"],
    },
}


# ══════════════════════════════════════════════════════════════════
# T2: Name heuristics
# ══════════════════════════════════════════════════════════════════

# Common python_name → OpName transformations
_NAME_TRANSFORMS: List[Tuple[re.Pattern, str]] = [
    # conv2d → Conv2D, conv3d → Conv3D, conv1d → Conv1D
    (re.compile(r"^conv(\d)d$"), r"Conv\1D"),
    # depthwise_conv2d → DepthwiseConv2dNative
    (re.compile(r"^depthwise_conv2d$"), "DepthwiseConv2dNative"),
    # max_pool → MaxPool, avg_pool → AvgPool, max_pool2d → MaxPool
    (re.compile(r"^(max|avg)_pool(\dd)?$"), lambda m: f"{m.group(1).title()}Pool"),
    # batch_normalization → FusedBatchNormV3
    (re.compile(r"^batch_normalization$"), "FusedBatchNormV3"),
    # l2_normalize → L2Loss (approximate)
    (re.compile(r"^l2_normalize$"), "L2Loss"),
]

# Mapping of python func names to likely OpNames (simple CamelCase transform)
def _snake_to_camel(name: str) -> str:
    """relu → Relu, reduce_sum → ReduceSum, batch_matmul → BatchMatmul"""
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


def _try_name_heuristic(func_name: str) -> Optional[str]:
    """Try to guess the raw_op name from the Python function name."""
    # Try explicit transforms first
    for pattern, replacement in _NAME_TRANSFORMS:
        m = pattern.match(func_name)
        if m:
            if callable(replacement):
                return replacement(m)
            return m.expand(replacement)

    # Generic CamelCase
    return _snake_to_camel(func_name)


# ══════════════════════════════════════════════════════════════════
# T3: Source code inspection
# ══════════════════════════════════════════════════════════════════

# Patterns to find raw_op references in source code
_RAW_OP_PATTERNS = [
    re.compile(r"gen_(\w+)_ops\.(\w+)\("),            # gen_nn_ops.Conv2D(
    re.compile(r"raw_ops\.(\w+)\("),                    # raw_ops.Conv2D(
    re.compile(r"_op_def_lib\.apply_op\(['\"](\w+)['\"]"),  # older TF style
    re.compile(r"op_def_registry\.get\(['\"](\w+)['\"]"),
]


def _inspect_source_for_raw_ops(func) -> List[str]:
    """
    Inspect the source code of a function to find raw_op references.
    Returns list of raw_op names found.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    found = []
    for pat in _RAW_OP_PATTERNS:
        for m in pat.finditer(source):
            # gen_nn_ops.Conv2D → "Conv2D"
            # raw_ops.Conv2D → "Conv2D"
            name = m.group(m.lastindex)
            if name and name[0].isupper():
                found.append(name)
            elif m.lastindex >= 2:
                found.append(m.group(2))

    return list(dict.fromkeys(found))  # dedupe preserving order


# ══════════════════════════════════════════════════════════════════
# T4: LLM fallback
# ══════════════════════════════════════════════════════════════════

_LLM_RESOLVE_SYSTEM = """\
You are a TensorFlow internals expert. Given a high-level TF API name and its \
Python signature, identify the underlying tf.raw_ops operation(s) and parameter \
mapping.

Respond ONLY with a JSON object (no markdown fences):
{
  "raw_op_name": "<primary raw_op name, e.g. Conv2D, or null if composite>",
  "raw_op_names": ["<list of all raw_ops used>"],
  "param_mapping": {
    "<high_level_param>": "<raw_op_param>"
  },
  "is_composite": <true if uses multiple raw_ops>,
  "category": "<nn|math|linalg|image|signal|core|other>",
  "confidence": <0.0-1.0>,
  "notes": ["<any relevant notes>"]
}

RULES:
- param_mapping maps high-level API param names to the PRIMARY raw_op's param names
- Only include params that exist in both APIs (skip 'name', Python-only flags)
- If the API is purely composite with no single primary raw_op, set raw_op_name=null
- Common mappings: filters→filter, input_tensor→input, axis→reduction_indices
"""


def _llm_resolve(
    api_name: str,
    signature_str: str,
    llm_client: Any,
) -> Optional[Dict[str, Any]]:
    """Ask LLM to resolve the API. Returns parsed dict or None."""
    user_prompt = f"API: {api_name}\nPython signature: {signature_str}\n"

    try:
        raw = llm_client.chat(_LLM_RESOLVE_SYSTEM, user_prompt)
    except Exception as e:
        print(f"  [LLM-resolve] error for {api_name}: {e}")
        return None

    if not raw:
        return None

    # Parse JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object
        brace_start = cleaned.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
    return None


# ══════════════════════════════════════════════════════════════════
# Auto parameter mapping via name matching
# ══════════════════════════════════════════════════════════════════

def _auto_map_params(
    high_level_params: List[str],
    raw_op_input_args: List[str],
    raw_op_attrs: List[str],
    explicit_mapping: Dict[str, str],
) -> Dict[str, str]:
    """
    Build a complete parameter mapping by combining explicit mappings
    with automatic same-name matching.
    """
    mapping = dict(explicit_mapping)
    all_raw_names = set(raw_op_input_args) | set(raw_op_attrs)
    mapped_raw = set(mapping.values())

    for hp in high_level_params:
        if hp in mapping:
            continue
        if hp == "name":
            continue
        # Exact match
        if hp in all_raw_names and hp not in mapped_raw:
            mapping[hp] = hp
            mapped_raw.add(hp)
            continue
        # Lowercase match
        hp_low = hp.lower()
        for rn in all_raw_names:
            if rn.lower() == hp_low and rn not in mapped_raw:
                mapping[hp] = rn
                mapped_raw.add(rn)
                break

    return mapping


# ══════════════════════════════════════════════════════════════════
# API category detection
# ══════════════════════════════════════════════════════════════════

_MODULE_CATEGORY_MAP = {
    "tf.raw_ops": "raw_ops",
    "tf.nn": "nn",
    "tf.math": "math",
    "tf.linalg": "linalg",
    "tf.image": "image",
    "tf.signal": "signal",
    "tf.io": "io",
    "tf.strings": "strings",
    "tf.dtypes": "dtypes",
    "tf.bitwise": "bitwise",
    "tf.random": "random",
    "tf.keras": "keras",
    "tf.keras.layers": "keras_layers",
    "tf.keras.losses": "keras_losses",
    "tf.keras.activations": "keras_activations",
    "tf.distribute": "distribute",
    "tf.data": "data",
}


def _detect_category(api_name: str) -> Tuple[str, str]:
    """Returns (category, module_path)."""
    parts = api_name.split(".")
    # Try progressively shorter module prefixes
    for i in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:i])
        if module in _MODULE_CATEGORY_MAP:
            return _MODULE_CATEGORY_MAP[module], module

    # Fallback: if starts with tf., use "core"
    if api_name.startswith("tf."):
        module = ".".join(parts[:-1])
        return "core", module

    return "unknown", ""


# ══════════════════════════════════════════════════════════════════
# Main resolver class
# ══════════════════════════════════════════════════════════════════

class TFApiResolver:
    """
    Resolves high-level TF APIs to their underlying raw_ops.
    """

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    def resolve(self, api_name: str) -> ResolveResult:
        """Resolve a single API name."""
        result = ResolveResult(api_name=api_name)
        category, module = _detect_category(api_name)
        result.api_category = category
        result.api_module = module

        # Check if it's already a raw_ops API
        if "raw_ops" in api_name:
            result.is_raw_ops = True
            result.raw_op_name = api_name.split(".")[-1]
            result.raw_op_names = [result.raw_op_name]
            result.confidence = 1.0
            result.strategy = "raw_ops_direct"
            return result

        # Resolve the Python object
        func_obj = None
        sig_str = ""
        high_level_params = []
        try:
            func_obj = self._resolve_obj(api_name)
            result.is_class = inspect.isclass(func_obj)
            try:
                sig = inspect.signature(func_obj)
                sig_str = str(sig)
                high_level_params = [
                    p.name for p in sig.parameters.values()
                    if p.name != "self"
                ]
            except (ValueError, TypeError):
                pass
        except Exception as e:
            result.error = f"Could not resolve: {e}"
            result.notes.append(f"import error: {e}")

        # T1: Static mapping
        static = STATIC_MAPPING.get(api_name)
        if static:
            result.raw_op_name = static.get("raw_op")
            result.raw_op_names = [static["raw_op"]] if static.get("raw_op") else []
            result.param_mapping = dict(static.get("params") or {})
            result.confidence = 0.95
            result.strategy = "static_mapping"
            result.notes.extend(static.get("notes") or [])
            if static.get("category"):
                result.api_category = static["category"]

            # Try to complete the param mapping with OpDef info
            self._complete_mapping_from_opdef(result, high_level_params)
            return result

        # T2: Name heuristics
        func_name = api_name.split(".")[-1]
        guessed_op = _try_name_heuristic(func_name)
        if guessed_op:
            # Verify it exists in the registry
            try:
                from tensorflow.python.framework import op_def_registry
                opdef = op_def_registry.get(guessed_op)
                if opdef is not None:
                    result.raw_op_name = guessed_op
                    result.raw_op_names = [guessed_op]
                    result.confidence = 0.7
                    result.strategy = "name_heuristic"
                    self._complete_mapping_from_opdef(result, high_level_params)
                    return result
            except Exception:
                pass

            # Try with V2 suffix (common in TF)
            for suffix in ["V2", "V3", ""]:
                try:
                    from tensorflow.python.framework import op_def_registry
                    opdef = op_def_registry.get(guessed_op + suffix)
                    if opdef is not None:
                        result.raw_op_name = guessed_op + suffix
                        result.raw_op_names = [guessed_op + suffix]
                        result.confidence = 0.65
                        result.strategy = "name_heuristic_versioned"
                        self._complete_mapping_from_opdef(result, high_level_params)
                        return result
                except Exception:
                    pass

        # T3: Source code inspection
        if func_obj is not None:
            found_ops = _inspect_source_for_raw_ops(func_obj)
            if found_ops:
                result.raw_op_name = found_ops[0]
                result.raw_op_names = found_ops
                result.confidence = 0.6
                result.strategy = "source_inspection"
                if len(found_ops) > 1:
                    result.notes.append(
                        f"Multiple raw_ops found in source: {found_ops}; "
                        f"using first as primary"
                    )
                self._complete_mapping_from_opdef(result, high_level_params)
                return result

        # T4: LLM fallback
        if self.llm_client is not None and sig_str:
            llm_result = _llm_resolve(api_name, sig_str, self.llm_client)
            if llm_result and isinstance(llm_result, dict):
                raw_op = llm_result.get("raw_op_name")
                if raw_op:
                    result.raw_op_name = raw_op
                    result.raw_op_names = llm_result.get("raw_op_names") or [raw_op]
                else:
                    result.raw_op_names = llm_result.get("raw_op_names") or []
                result.param_mapping = llm_result.get("param_mapping") or {}
                result.confidence = float(llm_result.get("confidence", 0.4))
                result.strategy = "llm"
                result.notes.extend(llm_result.get("notes") or [])
                if llm_result.get("category"):
                    result.api_category = llm_result["category"]

                # Verify the raw_op exists
                if result.raw_op_name:
                    try:
                        from tensorflow.python.framework import op_def_registry
                        if op_def_registry.get(result.raw_op_name) is None:
                            result.notes.append(
                                f"LLM-suggested raw_op {result.raw_op_name} not in registry"
                            )
                            result.confidence *= 0.5
                    except Exception:
                        pass

                self._complete_mapping_from_opdef(result, high_level_params)
                return result

        # No resolution found
        result.strategy = "unresolved"
        result.confidence = 0.0
        result.notes.append("Could not resolve to any raw_op")
        return result

    def _resolve_obj(self, qualname: str):
        """Import and resolve a Python object from its qualified name."""
        qualname = qualname.strip()
        if qualname.endswith("()"):
            qualname = qualname[:-2]
        module_name, _, attr_name = qualname.rpartition(".")
        if not module_name:
            raise ValueError(f"Invalid qualified name: {qualname}")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def _complete_mapping_from_opdef(
        self,
        result: ResolveResult,
        high_level_params: List[str],
    ) -> None:
        """
        Complete the param mapping using OpDef info.
        Also build the inverse mapping.
        """
        if not result.raw_op_name:
            return

        try:
            from tensorflow.python.framework import op_def_registry
            opdef = op_def_registry.get(result.raw_op_name)
            if opdef is None:
                return

            input_arg_names = [a.name for a in opdef.input_arg]
            attr_names = [a.name for a in opdef.attr]

            result.param_mapping = _auto_map_params(
                high_level_params=high_level_params,
                raw_op_input_args=input_arg_names,
                raw_op_attrs=attr_names,
                explicit_mapping=result.param_mapping,
            )
        except Exception as e:
            result.notes.append(f"OpDef lookup failed: {e}")

        # Build inverse mapping
        result.inverse_mapping = {v: k for k, v in result.param_mapping.items()}


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Resolve high-level TF APIs to underlying raw_ops."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--api", help="single API name to resolve")
    g.add_argument("--api_list", help="txt/json file of API names")
    ap.add_argument("--out_dir", default="./tf_resolved",
                    help="output directory for resolution results")
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--llm_api_key", default=None)
    args = ap.parse_args()

    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        raise SystemExit(f"TensorFlow import failed: {e}")

    # Build LLM client if requested
    llm_client = None
    if args.llm_base_url:
        from llm_doc_rank_extractor import LLMClient
        llm_client = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key or "",
        )
        print(f"[i] LLM enabled: {args.llm_base_url}")

    resolver = TFApiResolver(llm_client=llm_client)

    # Build API list
    if args.api:
        apis = [args.api]
    else:
        from tf_schema_common import load_api_list
        apis = load_api_list(args.api_list)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "resolved": 0, "unresolved": 0}

    for api_name in apis:
        result = resolver.resolve(api_name)
        stats["total"] += 1
        if result.raw_op_name:
            stats["resolved"] += 1
        else:
            stats["unresolved"] += 1

        safe = api_name.replace(".", "_")
        out_path = out_dir / f"{safe}_resolved.json"
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[{'+'if result.raw_op_name else '!'}] {api_name} → "
            f"raw_op={result.raw_op_name} strategy={result.strategy} "
            f"conf={result.confidence:.2f}"
        )

    print(f"\n[done] resolved={stats['resolved']}, "
          f"unresolved={stats['unresolved']}, total={stats['total']}")


if __name__ == "__main__":
    main()
