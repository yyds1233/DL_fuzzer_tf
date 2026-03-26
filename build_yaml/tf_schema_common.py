#!/usr/bin/env python3
"""
tf_schema_common.py  –  shared constants, helpers, and tiny utilities
for the TF raw_ops schema → YAML pipeline (Part 1 only).

Changes vs. v2
---------------
1. Extended role taxonomy: added SEMANTIC_ROLE_* constants for fine-grained
   parameter classification (data_tensor, weight_tensor, shape_control,
   index_input, fixed_arity_list, layout_attr, etc.).
2. Added `classify_param_semantic_role()` – returns fine-grained role string.
3. Added OP_FAMILY_RULES – per-op-family constraints (e.g., Conv2D strides
   must be length 4).
4. All original exports preserved for backward compatibility.
"""
from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# ── sentinel markers ──────────────────────────────────────────────
RANK_MISS_MARKER = "__RANK_TODO__"
ENUM_TODO_MARKER = "__ENUM_TODO__"

# ── default ranges ────────────────────────────────────────────────
DEFAULT_TENSOR_DTYPES = ["float32", "float64"]
DEFAULT_TENSOR_DTYPES_OPT = ["float32", "float64"]
DEFAULT_INT_RANGE = [-1, 8]
DEFAULT_DIM_RANGE = [-4, 4]
DEFAULT_FLOAT_RANGE = [-1.0, 1.0]
DEFAULT_EPS_RANGE = [1e-12, 1e-1]
DEFAULT_PROB_RANGE = [0.0, 1.0]
DEFAULT_INT_LIST_LEN_RANGE = [1, 3]
DEFAULT_INT_LIST_RANGE = [0, 4]

# ══════════════════════════════════════════════════════════════════
# COARSE parameter-role vocabulary (backward compatible)
# ══════════════════════════════════════════════════════════════════
PRIMARY_INPUT_NAMES: set = {
    "input", "x", "a", "features", "images", "image", "value",
    "tensor", "logits", "data", "params", "operand",
    "lhs", "y",
}

AUX_INPUT_NAMES: set = {
    "filter", "filters", "bias", "b", "weight", "weights",
    "scale", "offset", "mean", "variance",
    "indices", "index", "segment_ids", "updates",
    "labels", "targets", "mask",
}

ATTR_LIKE_NAMES: set = {
    "strides", "dilations", "ksize", "padding", "data_format",
    "explicit_paddings", "perm", "shape", "axis", "dim",
    "keep_dims", "keepdims", "transpose_a", "transpose_b",
    "name", "out_type", "Tidx", "T", "use_cudnn_on_gpu",
}


def classify_param_role(param_name: str) -> str:
    """Classify a parameter name into one of: primary / aux / attr / unknown."""
    low = param_name.lower()
    if low in {n.lower() for n in PRIMARY_INPUT_NAMES}:
        return "primary"
    if low in {n.lower() for n in AUX_INPUT_NAMES}:
        return "aux"
    if low in {n.lower() for n in ATTR_LIKE_NAMES}:
        return "attr"
    return "unknown"


# ══════════════════════════════════════════════════════════════════
# FINE-GRAINED semantic role taxonomy (NEW)
# ══════════════════════════════════════════════════════════════════

# Semantic roles – used by param_role_enricher.py
SEMANTIC_ROLE_DATA_TENSOR = "data_tensor"       # primary data input (input, x, features)
SEMANTIC_ROLE_WEIGHT_TENSOR = "weight_tensor"    # learned parameter (filter, weight, bias)
SEMANTIC_ROLE_AUX_TENSOR = "aux_tensor"          # auxiliary tensor (mean, variance, scale, offset)
SEMANTIC_ROLE_INDEX_INPUT = "index_input"        # indices, segment_ids, perm
SEMANTIC_ROLE_SHAPE_CONTROL = "shape_control"    # shape param that controls output geometry
SEMANTIC_ROLE_FIXED_ARITY_LIST = "fixed_arity_list"  # int list with op-determined length (strides, dilations)
SEMANTIC_ROLE_LAYOUT_ATTR = "layout_attr"        # data_format, padding
SEMANTIC_ROLE_SCALAR_ATTR = "scalar_attr"        # bool / int / float knob
SEMANTIC_ROLE_DTYPE_ATTR = "dtype_attr"          # T, out_type, Tidx
SEMANTIC_ROLE_META = "meta"                      # name, internal bookkeeping

# Mapping: param_name (lowercase) → semantic role
_SEMANTIC_ROLE_MAP: Dict[str, str] = {
    # data tensors
    "input": SEMANTIC_ROLE_DATA_TENSOR,
    "x": SEMANTIC_ROLE_DATA_TENSOR,
    "a": SEMANTIC_ROLE_DATA_TENSOR,
    "features": SEMANTIC_ROLE_DATA_TENSOR,
    "images": SEMANTIC_ROLE_DATA_TENSOR,
    "image": SEMANTIC_ROLE_DATA_TENSOR,
    "value": SEMANTIC_ROLE_DATA_TENSOR,
    "tensor": SEMANTIC_ROLE_DATA_TENSOR,
    "logits": SEMANTIC_ROLE_DATA_TENSOR,
    "data": SEMANTIC_ROLE_DATA_TENSOR,
    "params": SEMANTIC_ROLE_DATA_TENSOR,
    "operand": SEMANTIC_ROLE_DATA_TENSOR,
    "lhs": SEMANTIC_ROLE_DATA_TENSOR,
    "y": SEMANTIC_ROLE_DATA_TENSOR,
    # weight / bias tensors
    "filter": SEMANTIC_ROLE_WEIGHT_TENSOR,
    "filters": SEMANTIC_ROLE_WEIGHT_TENSOR,
    "weight": SEMANTIC_ROLE_WEIGHT_TENSOR,
    "weights": SEMANTIC_ROLE_WEIGHT_TENSOR,
    "bias": SEMANTIC_ROLE_WEIGHT_TENSOR,
    "b": SEMANTIC_ROLE_WEIGHT_TENSOR,
    # aux tensors
    "scale": SEMANTIC_ROLE_AUX_TENSOR,
    "offset": SEMANTIC_ROLE_AUX_TENSOR,
    "mean": SEMANTIC_ROLE_AUX_TENSOR,
    "variance": SEMANTIC_ROLE_AUX_TENSOR,
    "labels": SEMANTIC_ROLE_AUX_TENSOR,
    "targets": SEMANTIC_ROLE_AUX_TENSOR,
    "mask": SEMANTIC_ROLE_AUX_TENSOR,
    "updates": SEMANTIC_ROLE_AUX_TENSOR,
    # index / permutation inputs
    "indices": SEMANTIC_ROLE_INDEX_INPUT,
    "index": SEMANTIC_ROLE_INDEX_INPUT,
    "segment_ids": SEMANTIC_ROLE_INDEX_INPUT,
    "perm": SEMANTIC_ROLE_INDEX_INPUT,
    # shape-control inputs
    "shape": SEMANTIC_ROLE_SHAPE_CONTROL,
    # fixed-arity int lists
    "strides": SEMANTIC_ROLE_FIXED_ARITY_LIST,
    "dilations": SEMANTIC_ROLE_FIXED_ARITY_LIST,
    "ksize": SEMANTIC_ROLE_FIXED_ARITY_LIST,
    "explicit_paddings": SEMANTIC_ROLE_FIXED_ARITY_LIST,
    # layout attrs
    "padding": SEMANTIC_ROLE_LAYOUT_ATTR,
    "data_format": SEMANTIC_ROLE_LAYOUT_ATTR,
    # scalar attrs
    "axis": SEMANTIC_ROLE_SCALAR_ATTR,
    "dim": SEMANTIC_ROLE_SCALAR_ATTR,
    "keep_dims": SEMANTIC_ROLE_SCALAR_ATTR,
    "keepdims": SEMANTIC_ROLE_SCALAR_ATTR,
    "transpose_a": SEMANTIC_ROLE_SCALAR_ATTR,
    "transpose_b": SEMANTIC_ROLE_SCALAR_ATTR,
    "use_cudnn_on_gpu": SEMANTIC_ROLE_SCALAR_ATTR,
    # dtype attrs
    "t": SEMANTIC_ROLE_DTYPE_ATTR,
    "out_type": SEMANTIC_ROLE_DTYPE_ATTR,
    "tidx": SEMANTIC_ROLE_DTYPE_ATTR,
    # meta
    "name": SEMANTIC_ROLE_META,
}


def classify_param_semantic_role(param_name: str, op_name: str = "") -> str:
    """
    Fine-grained semantic role classification.
    Uses name heuristics + optional op_name context.
    """
    low = param_name.lower()

    # Exact match first
    if low in _SEMANTIC_ROLE_MAP:
        return _SEMANTIC_ROLE_MAP[low]

    # Heuristic patterns
    if low.endswith("_indices") or low.endswith("_index"):
        return SEMANTIC_ROLE_INDEX_INPUT
    if low.endswith("_shape") or low == "output_shape":
        return SEMANTIC_ROLE_SHAPE_CONTROL
    if low.endswith("_format"):
        return SEMANTIC_ROLE_LAYOUT_ATTR
    if low.startswith("num_") or low.endswith("_size"):
        return SEMANTIC_ROLE_SCALAR_ATTR

    # Fallback to coarse role
    coarse = classify_param_role(param_name)
    role_map = {
        "primary": SEMANTIC_ROLE_DATA_TENSOR,
        "aux": SEMANTIC_ROLE_AUX_TENSOR,
        "attr": SEMANTIC_ROLE_SCALAR_ATTR,
    }
    return role_map.get(coarse, SEMANTIC_ROLE_SCALAR_ATTR)


# ══════════════════════════════════════════════════════════════════
# OP FAMILY RULES (NEW)
# ══════════════════════════════════════════════════════════════════
# Family-level constraints that normalize_yaml_skeleton or
# param_role_enricher can apply automatically.
#
# Structure:
#   family_key → {
#       "match": list of op name substrings (any match → this family),
#       "primary_rank": int or None,
#       "fixed_arity_params": { param_name: required_length },
#       "shape_control_params": [param_name, ...],
#       "index_input_params": [param_name, ...],
#       "weight_rank": int or None,
#   }

OP_FAMILY_RULES: Dict[str, Dict[str, Any]] = {
    "conv2d": {
        "match": ["Conv2D"],
        "primary_rank": 4,
        "weight_rank": 4,
        "fixed_arity_params": {
            "strides": 4,
            "dilations": 4,
            "explicit_paddings": 8,
        },
    },
    "conv3d": {
        "match": ["Conv3D"],
        "primary_rank": 5,
        "weight_rank": 5,
        "fixed_arity_params": {
            "strides": 5,
            "dilations": 5,
        },
    },
    "conv1d": {
        "match": ["Conv1D"],
        "primary_rank": 3,
        "weight_rank": 3,
        "fixed_arity_params": {
            "strides": 3,
            "dilations": 3,
        },
    },
    "depthwise_conv2d": {
        "match": ["DepthwiseConv2d", "DepthwiseConv2D"],
        "primary_rank": 4,
        "weight_rank": 4,
        "fixed_arity_params": {
            "strides": 4,
            "dilations": 4,
        },
    },
    "pool2d": {
        "match": ["MaxPool", "AvgPool", "MaxPoolV2"],
        "primary_rank": 4,
        "fixed_arity_params": {
            "ksize": 4,
            "strides": 4,
        },
    },
    "pool3d": {
        "match": ["MaxPool3D", "AvgPool3D"],
        "primary_rank": 5,
        "fixed_arity_params": {
            "ksize": 5,
            "strides": 5,
        },
    },
    "matmul": {
        "match": ["MatMul", "BatchMatMul", "BatchMatMulV2", "BatchMatMulV3"],
        "primary_rank": 2,
        "weight_rank": 2,
    },
    "bias_add": {
        "match": ["BiasAdd", "BiasAddV1"],
        "primary_rank": None,  # any rank
        "primary_rank_any": True,
    },
    "batch_norm": {
        "match": ["FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"],
        "primary_rank": 4,
    },
    "reshape": {
        "match": ["Reshape"],
        "primary_rank": None,
        "primary_rank_any": True,
        "shape_control_params": ["shape"],
    },
    "transpose": {
        "match": ["Transpose", "ConjugateTranspose"],
        "primary_rank": None,
        "primary_rank_any": True,
        "index_input_params": ["perm"],
    },
    "gather": {
        "match": ["Gather", "GatherV2", "GatherNd"],
        "primary_rank": None,
        "primary_rank_any": True,
        "index_input_params": ["indices"],
    },
    "scatter": {
        "match": ["ScatterNd", "TensorScatterUpdate"],
        "primary_rank": None,
        "primary_rank_any": True,
        "index_input_params": ["indices"],
        "shape_control_params": ["shape"],
    },
    "reduce": {
        "match": ["Sum", "Mean", "Prod", "Max", "Min", "All", "Any",
                   "ReduceSum", "ReduceMean", "ReduceProd", "ReduceMax", "ReduceMin"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "concat": {
        "match": ["Concat", "ConcatV2"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "split": {
        "match": ["Split", "SplitV"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "softmax": {
        "match": ["Softmax", "LogSoftmax"],
        "primary_rank": None,
        "primary_rank_min": 1,
    },
    "one_hot": {
        "match": ["OneHot"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
}


def find_op_family(op_name: str) -> Optional[Dict[str, Any]]:
    """Find the family rule for a given op_name, or None."""
    for _family_key, rule in OP_FAMILY_RULES.items():
        for pattern in rule.get("match", []):
            if op_name == pattern:
                return rule
    # Partial match fallback (e.g., "Conv2DBackpropInput" matches "Conv2D")
    for _family_key, rule in OP_FAMILY_RULES.items():
        for pattern in rule.get("match", []):
            if pattern in op_name:
                return rule
    return None


# ── name helpers ──────────────────────────────────────────────────

def safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "default"
    return (
        s.replace("::", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _name_lower(name: str) -> str:
    return (name or "").lower()


# ── file / data IO ───────────────────────────────────────────────

def load_api_list(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("json must be a list of api strings")
        return [str(x).strip() for x in data if str(x).strip()]
    if suffix in (".pkl", ".pickle"):
        with p.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError("pickle must be a list of api strings")
        return [str(x).strip() for x in data if str(x).strip()]
    raise ValueError(f"Unsupported api list file: {path}")


def iter_files(src: Path, suffix: str) -> List[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(src.glob(f"*{suffix}"))
    return []


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def decode_bytes_maybe(x: Any) -> Any:
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return repr(x)
    return x


def listify_unique(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ── default-value helpers ────────────────────────────────────────

def parse_default_repr(default_repr: Optional[str]) -> Any:
    if default_repr is None:
        return None
    try:
        return ast.literal_eval(default_repr)
    except Exception:
        return default_repr


# ── dtype / range heuristics ─────────────────────────────────────

def tensor_dtype_choices_for_param_name(arg_name: str, optional: bool) -> List[str]:
    n = _name_lower(arg_name)
    if any(tok in n for tok in ("indices", "index")) or (n.startswith("ind") or "_ind" in n):
        return ["int64"]
    if "mask" in n:
        return ["bool"]
    return DEFAULT_TENSOR_DTYPES_OPT if optional else DEFAULT_TENSOR_DTYPES


def int_range_for_param_name(arg_name: str) -> List[int]:
    n = _name_lower(arg_name)
    if "dim" in n or "axis" in n:
        return DEFAULT_DIM_RANGE[:]
    return DEFAULT_INT_RANGE[:]


def float_range_for_param_name(arg_name: str) -> List[float]:
    n = _name_lower(arg_name)
    if "eps" in n:
        return DEFAULT_EPS_RANGE[:]
    if n == "p" or n.endswith("_p") or "prob" in n:
        return DEFAULT_PROB_RANGE[:]
    return DEFAULT_FLOAT_RANGE[:]


# ── rank-hint placeholders ───────────────────────────────────────

def stable_rank_hints_placeholder() -> Dict[str, Any]:
    return {
        "marker": RANK_MISS_MARKER,
        "status": "missing",
        "rank_candidates": [RANK_MISS_MARKER],
        "rank_any": None,
        "rank_min": None,
        "rank_max": None,
    }


def stable_param_rank_placeholder() -> Dict[str, Any]:
    return {
        "rank": None,
        "rank_min": None,
        "rank_max": None,
        "rank_any": False,
        "source": "none",
    }


def normalize_rank_hints(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = stable_rank_hints_placeholder()
    if not isinstance(raw, dict):
        return base
    ranks = raw.get("rank_candidates") or raw.get("fixed_ranks")
    norm_ranks: List[int] = []
    if isinstance(ranks, list):
        for x in ranks:
            try:
                norm_ranks.append(int(x))
            except Exception:
                continue
    base["marker"] = raw.get("marker") or base["marker"]
    status = raw.get("status")
    if status not in ("missing", "unassigned", "assigned"):
        status = "unassigned" if norm_ranks else "missing"
    base["status"] = status
    base["rank_candidates"] = sorted(set(norm_ranks)) if norm_ranks else base["rank_candidates"]
    base["rank_any"] = raw.get("rank_any", base["rank_any"])
    base["rank_min"] = raw.get("rank_min", base["rank_min"])
    base["rank_max"] = raw.get("rank_max", base["rank_max"])
    return base
