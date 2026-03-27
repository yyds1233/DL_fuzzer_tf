#!/usr/bin/env python3
"""
tf_highlevel_param_classifier.py  –  LLM-assisted parameter classification
for high-level TF APIs that lack OpDef information.

==========================================================================
WHY THIS EXISTS
==========================================================================

When we can't resolve a high-level API to a raw_op (composite ops, or
unresolved APIs), we have only the Python signature to work with.
The existing classify_from_signature_param() in tf_schema2yaml.py uses
simple heuristics (default value type, name matching) which fails for
many high-level APIs because:

  - Parameter names are more descriptive ("input_tensor", "num_classes")
  - Default values may be None for tensor params (optional inputs)
  - Type annotations may provide useful info
  - Docstrings describe the parameter roles clearly

This module uses LLM to classify parameters when heuristics are
insufficient, producing the same param spec format as the skeleton builder.

==========================================================================
USAGE
==========================================================================

  from tf_highlevel_param_classifier import classify_params_with_llm

  # schema is the schema.json dict from export_tf_schema_unified.py
  param_specs = classify_params_with_llm(
      api_name="tf.image.resize",
      py_sig=schema["python_signature"],
      llm_client=llm_client,
  )
  # param_specs = {"images": {"origin": "input", "kind": "tensor", ...}, ...}
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tf_schema_common import (
    ENUM_TODO_MARKER,
    DEFAULT_INT_LIST_LEN_RANGE,
    DEFAULT_INT_LIST_RANGE,
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
    tensor_dtype_choices_for_param_name,
    int_range_for_param_name,
    float_range_for_param_name,
    parse_default_repr,
)


# ══════════════════════════════════════════════════════════════════
# LLM prompt for parameter classification
# ══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a TensorFlow API parameter classification expert.

Given a high-level TF API name and its Python signature parameters, classify
each parameter into the appropriate category for test generation.

For each parameter, determine:
1. Whether it's a TENSOR input or a non-tensor ATTRIBUTE
2. Its semantic role (data_tensor, weight_tensor, aux_tensor, etc.)
3. For tensor params: expected dtype(s) and a rough shape description
4. For attr params: the kind (bool, int, float, enum, int_list, etc.)

Respond ONLY with a JSON object (no markdown fences):
{
  "<param_name>": {
    "origin": "input" | "attr" | "kwarg",
    "kind": "tensor" | "tensor_optional" | "tensor_list" | "bool" | "int" | "float" | "enum" | "dtype_enum" | "int_list" | "float_list" | "string_optional",
    "semantic_role": "<one of: data_tensor, weight_tensor, aux_tensor, index_input, shape_control, fixed_arity_list, layout_attr, scalar_attr, dtype_attr, meta>",
    "dtype_choices": ["float32", "float64"],  // only for tensor kinds
    "shape_hint": "4-D for images NHWC" | "1-D bias" | null,  // brief description
    "default": <default value if known, else null>,
    "enum_values": ["SAME", "VALID"],  // only for enum kinds
    "range": [0, 8],  // only for int/float kinds
    "description": "<1-sentence summary>"
  },
  ...
}

RULES:
- Parameters with default=None that accept tensors → kind="tensor_optional"
- Parameters named "name" → origin="kwarg", kind="string_optional", semantic_role="meta"
- Parameters for data format (padding, data_format) → origin="attr", kind="enum"
- dtype parameters → origin="attr", kind="dtype_enum"
- axis/dim parameters → origin="attr", kind="int"
- strides/dilations → origin="attr", kind="int_list"
- Boolean flags → origin="attr", kind="bool"
- For tensor params, provide realistic dtype_choices based on the API semantics
- shape_hint should describe what shape is expected (e.g., "4-D NHWC image tensor")
"""


def _build_user_prompt(
    api_name: str,
    py_sig: Dict[str, Any],
    doc_snippet: str = "",
) -> str:
    """Build user prompt for parameter classification."""
    parts = [f"API: {api_name}\n"]

    sig_str = py_sig.get("signature_str", "")
    if sig_str:
        parts.append(f"Signature: {api_name}{sig_str}\n")

    params = py_sig.get("parameters") or []
    if params:
        parts.append("Parameters:")
        for p in params:
            name = p.get("name", "?")
            has_default = p.get("has_default", False)
            default = p.get("default")
            annotation = p.get("annotation")
            kind = p.get("kind", "POSITIONAL_OR_KEYWORD")

            line = f"  - {name}"
            if annotation:
                line += f" (annotation={annotation})"
            if has_default:
                line += f" (default={default})"
            line += f" [{kind}]"
            parts.append(line)

    if doc_snippet:
        parts.append(f"\nDocumentation snippet:\n{doc_snippet[:2000]}")

    parts.append("\nClassify ALL parameters. Return ONLY JSON.")
    return "\n".join(parts)


def _parse_llm_classification(
    llm_result: Dict[str, Any],
    py_sig: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Convert LLM classification output into param spec dicts compatible
    with the YAML skeleton builder.
    """
    param_specs: Dict[str, Dict[str, Any]] = {}

    for pname, pinfo in llm_result.items():
        if not isinstance(pinfo, dict):
            continue

        origin = pinfo.get("origin", "attr")
        kind = pinfo.get("kind", "enum")
        semantic_role = pinfo.get("semantic_role", SEMANTIC_ROLE_SCALAR_ATTR)
        default = pinfo.get("default")

        spec: Dict[str, Any] = {"origin": origin}

        if kind in ("tensor", "tensor_optional", "tensor_list"):
            spec["kind"] = kind
            dtypes = pinfo.get("dtype_choices")
            if isinstance(dtypes, list) and dtypes:
                spec["dtype_choices"] = dtypes
            else:
                spec["dtype_choices"] = tensor_dtype_choices_for_param_name(
                    pname, kind == "tensor_optional"
                )
            spec["shape_spec"] = ["TODO_SHAPE"]
            # Store shape_hint for downstream use
            shape_hint = pinfo.get("shape_hint")
            if shape_hint:
                spec["_shape_hint"] = shape_hint

        elif kind == "bool":
            spec["kind"] = "bool"
            if isinstance(default, bool):
                spec["default"] = default

        elif kind == "int":
            spec["kind"] = "int"
            rng = pinfo.get("range")
            if isinstance(rng, list) and len(rng) == 2:
                spec["range"] = rng
            else:
                spec["range"] = int_range_for_param_name(pname)
            if isinstance(default, int) and not isinstance(default, bool):
                spec["default"] = default

        elif kind == "float":
            spec["kind"] = "float"
            rng = pinfo.get("range")
            if isinstance(rng, list) and len(rng) == 2:
                spec["range"] = rng
            else:
                spec["range"] = float_range_for_param_name(pname)
            if isinstance(default, (int, float)):
                spec["default"] = float(default)

        elif kind == "enum":
            spec["kind"] = "enum"
            vals = pinfo.get("enum_values")
            if isinstance(vals, list) and vals:
                spec["values"] = vals
            else:
                spec["values"] = [ENUM_TODO_MARKER]
            if isinstance(default, str):
                if default not in spec["values"]:
                    spec["values"].append(default)
                spec["default"] = default

        elif kind == "dtype_enum":
            spec["kind"] = "dtype_enum"
            vals = pinfo.get("enum_values")
            if isinstance(vals, list) and vals:
                spec["values"] = vals
            else:
                spec["values"] = [ENUM_TODO_MARKER]
            if isinstance(default, str):
                spec["default"] = default

        elif kind == "int_list":
            spec["kind"] = "int_list"
            spec["len_range"] = DEFAULT_INT_LIST_LEN_RANGE[:]
            spec["range"] = DEFAULT_INT_LIST_RANGE[:]
            if isinstance(default, list) and all(isinstance(x, int) for x in default):
                L = len(default)
                spec["default"] = default
                spec["len_range"] = [L, L] if L > 0 else spec["len_range"]

        elif kind == "float_list":
            spec["kind"] = "float_list"
            spec["len_range"] = DEFAULT_INT_LIST_LEN_RANGE[:]
            spec["range"] = [-1.0, 1.0]

        elif kind == "string_optional":
            spec["kind"] = "string_optional"
            spec["default"] = default

        else:
            spec["kind"] = "enum"
            spec["values"] = [ENUM_TODO_MARKER]

        spec["semantic_role"] = semantic_role

        # Determine coarse role
        if semantic_role in (SEMANTIC_ROLE_DATA_TENSOR,):
            spec["role"] = "primary"
        elif semantic_role in (SEMANTIC_ROLE_WEIGHT_TENSOR, SEMANTIC_ROLE_AUX_TENSOR):
            spec["role"] = "aux"
        else:
            spec["role"] = "attr"

        param_specs[pname] = spec

    return param_specs


# ══════════════════════════════════════════════════════════════════
# Heuristic fallback classifier (no LLM)
# ══════════════════════════════════════════════════════════════════

# Extended name patterns for high-level APIs
_TENSOR_NAME_PATTERNS = {
    "input", "inputs", "x", "a", "b", "features", "images", "image",
    "value", "values", "tensor", "tensors", "logits", "data", "params",
    "operand", "lhs", "rhs", "y", "predictions", "labels", "targets",
    "input_tensor", "input_data", "condition",
    "filter", "filters", "weight", "weights", "bias", "kernel",
    "scale", "offset", "mean", "variance",
    "indices", "index", "segment_ids", "updates", "mask",
    "query", "key", "value",  # attention
}

_ATTR_NAME_PATTERNS = {
    "axis", "dim", "keepdims", "keep_dims", "name", "dtype", "out_type",
    "padding", "data_format", "strides", "dilations", "ksize",
    "rate", "seed", "training", "transpose_a", "transpose_b",
    "num_classes", "depth", "on_value", "off_value",
    "method", "antialias", "preserve_aspect_ratio",
    "epsilon", "momentum", "center", "fused",
    "use_bias", "activation", "units",
}


def classify_params_heuristic(
    api_name: str,
    py_sig: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Classify parameters using heuristics only (no LLM).
    Works as a fallback when LLM is unavailable.
    """
    params_list = py_sig.get("parameters") or []
    param_specs: Dict[str, Dict[str, Any]] = {}

    for p in params_list:
        name = p.get("name", "")
        if not name or name == "self":
            continue

        has_default = p.get("has_default", False)
        default_repr = p.get("default")
        default_value = parse_default_repr(default_repr)
        low = name.lower()

        # Meta
        if name == "name":
            param_specs[name] = {
                "origin": "kwarg",
                "kind": "string_optional",
                "default": None,
                "semantic_role": SEMANTIC_ROLE_META,
                "role": "attr",
            }
            continue

        # dtype params
        if low in ("dtype", "out_type", "output_type"):
            param_specs[name] = {
                "origin": "attr",
                "kind": "dtype_enum",
                "values": [ENUM_TODO_MARKER],
                "semantic_role": SEMANTIC_ROLE_DTYPE_ATTR,
                "role": "attr",
            }
            if isinstance(default_value, str):
                param_specs[name]["default"] = default_value
            continue

        # Boolean params
        if isinstance(default_value, bool):
            param_specs[name] = {
                "origin": "attr",
                "kind": "bool",
                "default": default_value,
                "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                "role": "attr",
            }
            continue

        # Int params
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            param_specs[name] = {
                "origin": "attr",
                "kind": "int",
                "range": int_range_for_param_name(name),
                "default": default_value,
                "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                "role": "attr",
            }
            continue

        # Float params
        if isinstance(default_value, float):
            param_specs[name] = {
                "origin": "attr",
                "kind": "float",
                "range": float_range_for_param_name(name),
                "default": default_value,
                "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                "role": "attr",
            }
            continue

        # String params (likely enum)
        if isinstance(default_value, str) and low not in ("name",):
            if low in ("padding", "data_format", "method", "mode",
                       "interpolation") or low.endswith("_format"):
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "enum",
                    "values": [default_value],
                    "default": default_value,
                    "semantic_role": SEMANTIC_ROLE_LAYOUT_ATTR,
                    "role": "attr",
                }
            else:
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "enum",
                    "values": [default_value],
                    "default": default_value,
                    "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                    "role": "attr",
                }
            continue

        # List params
        if isinstance(default_value, (list, tuple)):
            if all(isinstance(x, int) for x in default_value):
                L = len(default_value)
                sem = SEMANTIC_ROLE_FIXED_ARITY_LIST if low in (
                    "strides", "dilations", "ksize"
                ) else SEMANTIC_ROLE_SCALAR_ATTR
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "int_list",
                    "len_range": [L, L] if L > 0 else DEFAULT_INT_LIST_LEN_RANGE[:],
                    "range": DEFAULT_INT_LIST_RANGE[:],
                    "default": list(default_value),
                    "semantic_role": sem,
                    "role": "attr",
                }
                continue

        # Attr-like names
        if low in _ATTR_NAME_PATTERNS:
            if low in ("strides", "dilations", "ksize"):
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "int_list",
                    "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
                    "range": DEFAULT_INT_LIST_RANGE[:],
                    "semantic_role": SEMANTIC_ROLE_FIXED_ARITY_LIST,
                    "role": "attr",
                }
            elif low in ("padding", "data_format", "method"):
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "enum",
                    "values": [ENUM_TODO_MARKER],
                    "semantic_role": SEMANTIC_ROLE_LAYOUT_ATTR,
                    "role": "attr",
                }
            elif low in ("axis", "dim"):
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "int",
                    "range": [-4, 4],
                    "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                    "role": "attr",
                }
            elif low in ("training", "center", "fused", "use_bias"):
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "bool",
                    "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                    "role": "attr",
                }
                if isinstance(default_value, bool):
                    param_specs[name]["default"] = default_value
            else:
                param_specs[name] = {
                    "origin": "attr",
                    "kind": "enum",
                    "values": [ENUM_TODO_MARKER],
                    "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                    "role": "attr",
                }
            continue

        # Tensor-like names or no default → tensor
        if low in _TENSOR_NAME_PATTERNS or not has_default:
            is_optional = has_default and default_value is None
            # Determine semantic role
            if low in ("filter", "filters", "weight", "weights", "bias", "kernel"):
                sem = SEMANTIC_ROLE_WEIGHT_TENSOR
                role = "aux"
            elif low in ("scale", "offset", "mean", "variance", "labels",
                         "targets", "mask", "updates"):
                sem = SEMANTIC_ROLE_AUX_TENSOR
                role = "aux"
            elif low in ("indices", "index", "segment_ids", "perm"):
                sem = SEMANTIC_ROLE_INDEX_INPUT
                role = "aux"
            elif low in ("shape", "output_shape") or low.endswith("_shape"):
                sem = SEMANTIC_ROLE_SHAPE_CONTROL
                role = "attr"
            else:
                sem = SEMANTIC_ROLE_DATA_TENSOR
                role = "primary"

            param_specs[name] = {
                "origin": "input",
                "kind": "tensor_optional" if is_optional else "tensor",
                "dtype_choices": tensor_dtype_choices_for_param_name(name, is_optional),
                "shape_spec": ["TODO_SHAPE"],
                "semantic_role": sem,
                "role": role,
            }
            continue

        # Default: optional with None default → could be tensor or attr
        if has_default and default_value is None:
            # Ambiguous — likely an optional tensor
            param_specs[name] = {
                "origin": "input",
                "kind": "tensor_optional",
                "dtype_choices": tensor_dtype_choices_for_param_name(name, True),
                "shape_spec": ["TODO_SHAPE"],
                "semantic_role": SEMANTIC_ROLE_AUX_TENSOR,
                "role": "aux",
            }
        else:
            # Completely unknown
            param_specs[name] = {
                "origin": "attr",
                "kind": "enum",
                "values": [ENUM_TODO_MARKER],
                "semantic_role": SEMANTIC_ROLE_SCALAR_ATTR,
                "role": "attr",
            }

    return param_specs


# ══════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════

def classify_params_with_llm(
    api_name: str,
    py_sig: Dict[str, Any],
    llm_client: Any = None,
    doc_snippet: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Classify parameters using LLM (with heuristic fallback).

    Args:
        api_name: Full API name (e.g., "tf.nn.conv2d")
        py_sig: Python signature dict from export_tf_schema
        llm_client: LLMClient instance (optional)
        doc_snippet: Documentation text snippet (optional)

    Returns:
        Dict of param_name → param_spec
    """
    if llm_client is None:
        return classify_params_heuristic(api_name, py_sig)

    # Build and send LLM prompt
    user_prompt = _build_user_prompt(api_name, py_sig, doc_snippet)

    try:
        raw_response = llm_client.chat(_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"  [LLM-classify] error for {api_name}: {e}")
        return classify_params_heuristic(api_name, py_sig)

    if not raw_response:
        return classify_params_heuristic(api_name, py_sig)

    # Parse response
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        llm_result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object
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
                            llm_result = json.loads(cleaned[brace_start:i + 1])
                            break
                        except json.JSONDecodeError:
                            pass
            else:
                return classify_params_heuristic(api_name, py_sig)
        else:
            return classify_params_heuristic(api_name, py_sig)

    if not isinstance(llm_result, dict):
        return classify_params_heuristic(api_name, py_sig)

    # Parse LLM result
    specs = _parse_llm_classification(llm_result, py_sig)

    # Merge with heuristic for any params the LLM missed
    heuristic_specs = classify_params_heuristic(api_name, py_sig)
    for pname, hspec in heuristic_specs.items():
        if pname not in specs:
            specs[pname] = hspec

    return specs
