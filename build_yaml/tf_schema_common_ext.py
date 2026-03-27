#!/usr/bin/env python3
"""
tf_schema_common_ext.py  –  Extensions to tf_schema_common.py for
high-level API support.

Import this AFTER tf_schema_common to add:
  - Extended parameter name sets for high-level APIs
  - Module-aware op family detection
  - High-level API specific heuristics

==========================================================================
USAGE
==========================================================================

  from tf_schema_common import *
  from tf_schema_common_ext import *  # adds high-level support

Or in tf_schema_common.py, add at the bottom:
  try:
      from tf_schema_common_ext import *
  except ImportError:
      pass
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from tf_schema_common import (
    OP_FAMILY_RULES,
    PRIMARY_INPUT_NAMES,
    AUX_INPUT_NAMES,
    ATTR_LIKE_NAMES,
    _SEMANTIC_ROLE_MAP,
    SEMANTIC_ROLE_DATA_TENSOR,
    SEMANTIC_ROLE_WEIGHT_TENSOR,
    SEMANTIC_ROLE_AUX_TENSOR,
    SEMANTIC_ROLE_INDEX_INPUT,
    SEMANTIC_ROLE_SHAPE_CONTROL,
    SEMANTIC_ROLE_FIXED_ARITY_LIST,
    SEMANTIC_ROLE_LAYOUT_ATTR,
    SEMANTIC_ROLE_SCALAR_ATTR,
    SEMANTIC_ROLE_DTYPE_ATTR,
    SEMANTIC_ROLE_META,
)


# ══════════════════════════════════════════════════════════════════
# Extended parameter name sets for high-level APIs
# ══════════════════════════════════════════════════════════════════

# Additional primary input names found in high-level APIs
EXTENDED_PRIMARY_INPUT_NAMES: Set[str] = PRIMARY_INPUT_NAMES | {
    "input_tensor",
    "inputs",
    "input_data",
    "predictions",
    "condition",
    "source",
    "query",  # attention APIs
}

# Additional aux names
EXTENDED_AUX_INPUT_NAMES: Set[str] = AUX_INPUT_NAMES | {
    "key", "value",  # attention APIs (note: "value" is also primary)
    "labels", "targets",
    "sample_weight", "class_weight",
    "initial_state", "sequence_length",
    "reference", "template",
}

# Additional attr names
EXTENDED_ATTR_LIKE_NAMES: Set[str] = ATTR_LIKE_NAMES | {
    "training", "trainable",
    "units", "filters", "kernel_size",  # keras-style
    "activation", "use_bias",
    "kernel_initializer", "bias_initializer",
    "kernel_regularizer", "bias_regularizer",
    "method", "interpolation", "antialias",
    "preserve_aspect_ratio", "crop_to_bounding_box",
    "num_classes", "depth",  # one_hot
    "on_value", "off_value",
    "rate",  # dropout
    "seed",
    "center", "scale_param",  # batch_norm
    "epsilon", "momentum",
    "fused",
    "num_or_size_splits",
    "maxval", "minval",
    "reduction",
    "from_logits",
    "label_smoothing",
    "normalize",
}

# Extended semantic role map
EXTENDED_SEMANTIC_ROLE_MAP: Dict[str, str] = dict(_SEMANTIC_ROLE_MAP)
EXTENDED_SEMANTIC_ROLE_MAP.update({
    # High-level primary inputs
    "input_tensor": SEMANTIC_ROLE_DATA_TENSOR,
    "inputs": SEMANTIC_ROLE_DATA_TENSOR,
    "input_data": SEMANTIC_ROLE_DATA_TENSOR,
    "predictions": SEMANTIC_ROLE_DATA_TENSOR,
    "condition": SEMANTIC_ROLE_DATA_TENSOR,
    "source": SEMANTIC_ROLE_DATA_TENSOR,
    "query": SEMANTIC_ROLE_DATA_TENSOR,
    # Attention
    "key": SEMANTIC_ROLE_AUX_TENSOR,
    # Keras-style
    "sample_weight": SEMANTIC_ROLE_AUX_TENSOR,
    "class_weight": SEMANTIC_ROLE_AUX_TENSOR,
    "initial_state": SEMANTIC_ROLE_AUX_TENSOR,
    "sequence_length": SEMANTIC_ROLE_SCALAR_ATTR,
    # Image
    "size": SEMANTIC_ROLE_SHAPE_CONTROL,
    "target_height": SEMANTIC_ROLE_SCALAR_ATTR,
    "target_width": SEMANTIC_ROLE_SCALAR_ATTR,
    # Misc attrs
    "training": SEMANTIC_ROLE_SCALAR_ATTR,
    "rate": SEMANTIC_ROLE_SCALAR_ATTR,
    "seed": SEMANTIC_ROLE_SCALAR_ATTR,
    "method": SEMANTIC_ROLE_LAYOUT_ATTR,
    "interpolation": SEMANTIC_ROLE_LAYOUT_ATTR,
    "reduction": SEMANTIC_ROLE_LAYOUT_ATTR,
    "from_logits": SEMANTIC_ROLE_SCALAR_ATTR,
    "activation": SEMANTIC_ROLE_LAYOUT_ATTR,
})


# ══════════════════════════════════════════════════════════════════
# Extended op family rules for high-level API patterns
# ══════════════════════════════════════════════════════════════════

# Add patterns that match high-level API names
HIGHLEVEL_FAMILY_RULES: Dict[str, Dict[str, Any]] = {
    # These extend the existing OP_FAMILY_RULES with additional match patterns
    "conv2d_hl": {
        "match": ["conv2d", "Conv2d"],  # lowercase from tf.nn.conv2d
        "primary_rank": 4,
        "weight_rank": 4,
        "fixed_arity_params": {
            "strides": 4,
            "dilations": 4,
        },
    },
    "conv3d_hl": {
        "match": ["conv3d", "Conv3d"],
        "primary_rank": 5,
        "weight_rank": 5,
        "fixed_arity_params": {
            "strides": 5,
            "dilations": 5,
        },
    },
    "conv1d_hl": {
        "match": ["conv1d", "Conv1d"],
        "primary_rank": 3,
        "weight_rank": 3,
        "fixed_arity_params": {
            "strides": 3,
            "dilations": 3,
        },
    },
    "pool2d_hl": {
        "match": ["max_pool", "avg_pool", "max_pool2d", "avg_pool2d"],
        "primary_rank": 4,
        "fixed_arity_params": {
            "ksize": 4,
            "strides": 4,
        },
    },
    "pool3d_hl": {
        "match": ["max_pool3d", "avg_pool3d"],
        "primary_rank": 5,
        "fixed_arity_params": {
            "ksize": 5,
            "strides": 5,
        },
    },
    "batch_norm_hl": {
        "match": ["batch_normalization"],
        "primary_rank": 4,
    },
    "matmul_hl": {
        "match": ["matmul"],
        "primary_rank": 2,
        "weight_rank": 2,
    },
    "reduce_hl": {
        "match": ["reduce_sum", "reduce_mean", "reduce_prod",
                  "reduce_max", "reduce_min", "reduce_any", "reduce_all",
                  "reduce_logsumexp", "reduce_std", "reduce_variance"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "softmax_hl": {
        "match": ["softmax", "log_softmax"],
        "primary_rank": None,
        "primary_rank_min": 1,
    },
    "activation_hl": {
        "match": ["relu", "relu6", "leaky_relu", "elu", "selu",
                  "sigmoid", "tanh", "gelu", "swish", "softplus",
                  "softsign"],
        "primary_rank": None,
        "primary_rank_any": True,
        "primary_rank_min": 1,
    },
    "image_resize_hl": {
        "match": ["resize", "resize_with_crop_or_pad"],
        "primary_rank": 4,  # [batch, height, width, channels]
    },
    "reshape_hl": {
        "match": ["reshape"],
        "primary_rank": None,
        "primary_rank_any": True,
        "shape_control_params": ["shape"],
    },
    "transpose_hl": {
        "match": ["transpose"],
        "primary_rank": None,
        "primary_rank_any": True,
        "index_input_params": ["perm"],
    },
    "gather_hl": {
        "match": ["gather", "gather_nd", "embedding_lookup"],
        "primary_rank": None,
        "primary_rank_any": True,
        "index_input_params": ["indices", "ids"],
    },
    "concat_hl": {
        "match": ["concat"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "split_hl": {
        "match": ["split"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "dropout_hl": {
        "match": ["dropout"],
        "primary_rank": None,
        "primary_rank_any": True,
        "primary_rank_min": 1,
    },
    "one_hot_hl": {
        "match": ["one_hot"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "bias_add_hl": {
        "match": ["bias_add"],
        "primary_rank": None,
        "primary_rank_any": True,
    },
    "cross_entropy_hl": {
        "match": ["softmax_cross_entropy_with_logits",
                  "sparse_softmax_cross_entropy_with_logits",
                  "sigmoid_cross_entropy_with_logits"],
        "primary_rank": 2,
    },
}


def find_op_family_extended(
    api_name: str,
    op_name: Optional[str] = None,
) -> Optional[str]:
    """
    Find op family for both raw_ops names and high-level API names.

    Tries:
    1. Raw op name against OP_FAMILY_RULES (original)
    2. Function name against HIGHLEVEL_FAMILY_RULES
    3. Partial matches
    """
    func_name = api_name.split(".")[-1]

    # Try raw op name first (if provided)
    if op_name:
        for family_key, rule in OP_FAMILY_RULES.items():
            for pattern in rule.get("match", []):
                if op_name == pattern or pattern in op_name:
                    return family_key

    # Try function name against high-level rules
    for family_key, rule in HIGHLEVEL_FAMILY_RULES.items():
        for pattern in rule.get("match", []):
            if func_name == pattern:
                return family_key

    # Partial match on function name
    for family_key, rule in HIGHLEVEL_FAMILY_RULES.items():
        for pattern in rule.get("match", []):
            if pattern in func_name or func_name in pattern:
                return family_key

    return None


def get_family_rule(family_key: str) -> Optional[Dict[str, Any]]:
    """Get family rule dict from either standard or extended rules."""
    if family_key in OP_FAMILY_RULES:
        return OP_FAMILY_RULES[family_key]
    if family_key in HIGHLEVEL_FAMILY_RULES:
        return HIGHLEVEL_FAMILY_RULES[family_key]
    return None


# ══════════════════════════════════════════════════════════════════
# Utility: classify param with extended tables
# ══════════════════════════════════════════════════════════════════

def classify_param_semantic_role_extended(
    param_name: str,
    op_name: str = "",
    api_name: str = "",
) -> str:
    """
    Extended semantic role classification that considers high-level API
    parameter naming conventions.
    """
    low = param_name.lower()

    # Check extended map first
    if low in EXTENDED_SEMANTIC_ROLE_MAP:
        return EXTENDED_SEMANTIC_ROLE_MAP[low]

    # Heuristic patterns for high-level APIs
    if low.endswith("_tensor") or low.endswith("_input"):
        return SEMANTIC_ROLE_DATA_TENSOR
    if low.endswith("_weight") or low.endswith("_kernel"):
        return SEMANTIC_ROLE_WEIGHT_TENSOR
    if low.endswith("_bias"):
        return SEMANTIC_ROLE_WEIGHT_TENSOR
    if low.endswith("_indices") or low.endswith("_index"):
        return SEMANTIC_ROLE_INDEX_INPUT
    if low.endswith("_shape") or low == "output_shape" or low == "target_shape":
        return SEMANTIC_ROLE_SHAPE_CONTROL
    if low.endswith("_format") or low.endswith("_method"):
        return SEMANTIC_ROLE_LAYOUT_ATTR
    if low.startswith("num_") or low.endswith("_size") or low.endswith("_count"):
        return SEMANTIC_ROLE_SCALAR_ATTR
    if low.startswith("use_") or low.startswith("is_") or low.startswith("enable_"):
        return SEMANTIC_ROLE_SCALAR_ATTR

    # Fall back to base classification
    from tf_schema_common import classify_param_semantic_role
    return classify_param_semantic_role(param_name, op_name)
