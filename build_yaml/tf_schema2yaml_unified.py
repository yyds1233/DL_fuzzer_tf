#!/usr/bin/env python3
"""
tf_schema2yaml_unified.py  –  Stage B: unified schema JSON → YAML skeleton.

Reworked goals:
1) Keep the original 3-layer fallback for building params.
2) Try harder to recover params/primary for unresolved high-level APIs.
3) Do NOT let unusable skeletons flow into Stage C.
4) Route unroutable skeletons into a dedicated empty_param directory with reports.

Key policy:
- Preferred: Stage B produces routable skeletons with non-empty params and valid primary_param.
- If recovery still fails, write YAML into --empty_param_dir and DO NOT write it to normal out_dir.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from tf_schema_common import (
    ENUM_TODO_MARKER,
    DEFAULT_INT_LIST_LEN_RANGE,
    DEFAULT_INT_LIST_RANGE,
    classify_param_role,
    dump_json,
    float_range_for_param_name,
    int_range_for_param_name,
    iter_files,
    load_json,
    normalize_rank_hints,
    parse_default_repr,
    safe_name,
    tensor_dtype_choices_for_param_name,
)

from tf_schema2yaml import (
    merge_signature_and_opdef as _original_merge,
    classify_from_signature_param,
    _attr_type_to_param_spec,
)

from tf_highlevel_param_classifier import (
    classify_params_with_llm,
    classify_params_heuristic,
)

GENERATOR_BLOCK = {
    "stage": "B-json-to-yaml-skeleton-unified-tf",
    "version": "2026-04-09-tf-v6",
}

_SKIP_TYPE_ATTRS = {
    "T", "Tparams", "Tindices", "Taxis", "SrcT", "DstT",
    "Tidx", "Tshape", "Tpaddings", "Tsegmentids",
    "out_type",
}

_TENSOR_KINDS = {"tensor", "tensor_optional", "tensor_list"}
_COMMON_TENSOR_INPUT_NAMES = {
    "input", "input_tensor", "inputs", "values", "x", "tensor", "images", "features", "data",
}
_COMMON_SCALAR_CONTROL_NAMES = {
    "axis", "axes", "dim", "dims", "reduction_indices", "keepdims", "keep_dims",
    "stable", "descending", "exclusive", "reverse",
}
_COMMON_STRING_ENUM_NAMES = {
    "direction": ["ASCENDING", "DESCENDING"],
    "mode": [ENUM_TODO_MARKER],
    "padding": [ENUM_TODO_MARKER],
    "data_format": [ENUM_TODO_MARKER],
}


def _is_tensor_like_kind(kind: Any) -> bool:
    return kind in _TENSOR_KINDS


# ══════════════════════════════════════════════════════════════════
# Basic helpers
# ══════════════════════════════════════════════════════════════════

def _build_tensor_spec_from_opdef(
    meta: Dict[str, Any],
    raw_name: str,
) -> Dict[str, Any]:
    type_name = meta.get("type_name")
    if type_name:
        return {
            "origin": "input",
            "kind": "tensor",
            "dtype_choices": [type_name],
            "shape_spec": ["TODO_SHAPE"],
            "_raw_op_param": raw_name,
        }
    elif meta.get("type_attr"):
        return {
            "origin": "input",
            "kind": "tensor",
            "dtype_from_attr": meta["type_attr"],
            "shape_spec": ["TODO_SHAPE"],
            "_raw_op_param": raw_name,
        }
    elif meta.get("type_list_attr"):
        return {
            "origin": "input",
            "kind": "tensor_list",
            "dtype_from_attr": meta["type_list_attr"],
            "shape_spec": ["TODO_SHAPE"],
            "_raw_op_param": raw_name,
        }
    else:
        return {
            "origin": "input",
            "kind": "tensor",
            "dtype_choices": ["float32", "float64"],
            "shape_spec": ["TODO_SHAPE"],
            "_raw_op_param": raw_name,
        }


def _normalize_default_to_value(raw_default: Any) -> Any:
    if raw_default is None:
        return None
    try:
        return parse_default_repr(raw_default)
    except Exception:
        return raw_default


# def _tensor_spec_for_name(name: str, optional: bool = False) -> Dict[str, Any]:
#     dtype_choices = tensor_dtype_choices_for_param_name(name, optional)
#     spec: Dict[str, Any] = {
#         "origin": "input",
#         "kind": "tensor_optional" if optional else "tensor",
#         "shape_spec": ["TODO_SHAPE"],
#     }
#     if dtype_choices:
#         spec["dtype_choices"] = dtype_choices
#     else:
#         spec["dtype_choices"] = ["float32", "float64"]
#     return spec


# ══════════════════════════════════════════════════════════════════
# Path 1: high-level with OpDef + Python signature
# ══════════════════════════════════════════════════════════════════

def merge_highlevel_with_opdef(schema: Dict[str, Any]) -> Dict[str, Any]:
    py_sig = schema.get("python_signature") or {}
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}
    param_mapping = tf_block.get("param_mapping") or {}
    inverse_mapping = tf_block.get("inverse_mapping") or {}

    py_params = py_sig.get("parameters") or []
    py_names = [p["name"] for p in py_params if isinstance(p, dict) and p.get("name")]

    input_args = op_def.get("input_args") or []
    attr_defs = op_def.get("attrs") or []

    input_map = {x["name"]: x for x in input_args if isinstance(x, dict) and x.get("name")}
    attr_map = {x["name"]: x for x in attr_defs if isinstance(x, dict) and x.get("name")}

    params: Dict[str, Any] = {}
    consumed_raw_inputs = set()
    consumed_raw_attrs = set()

    for py_name in py_names:
        if py_name == "name":
            params[py_name] = {"origin": "kwarg", "kind": "string_optional", "default": None}
            continue
        if py_name == "self":
            continue

        raw_name = param_mapping.get(py_name)
        if not raw_name:
            if py_name in input_map:
                raw_name = py_name
            elif py_name in attr_map:
                raw_name = py_name

        if raw_name and raw_name in input_map:
            params[py_name] = _build_tensor_spec_from_opdef(input_map[raw_name], raw_name)
            consumed_raw_inputs.add(raw_name)
            continue

        if raw_name and raw_name in attr_map:
            spec = _attr_type_to_param_spec(attr_map[raw_name])
            spec["_raw_op_param"] = raw_name
            params[py_name] = spec
            consumed_raw_attrs.add(raw_name)
            continue

        param_info = next((p for p in py_params if p.get("name") == py_name), None)
        if param_info is not None:
            params[py_name] = classify_from_signature_param(param_info)

    for raw_name, meta in input_map.items():
        if raw_name in consumed_raw_inputs:
            continue
        already = any(
            spec.get("_raw_op_param") == raw_name
            for spec in params.values() if isinstance(spec, dict)
        )
        if already:
            continue
        hl_name = inverse_mapping.get(raw_name, raw_name)
        if hl_name not in params:
            params[hl_name] = _build_tensor_spec_from_opdef(meta, raw_name)

    for raw_name, attr in attr_map.items():
        if raw_name in consumed_raw_attrs or raw_name in _SKIP_TYPE_ATTRS:
            continue
        already = any(
            spec.get("_raw_op_param") == raw_name
            for spec in params.values() if isinstance(spec, dict)
        )
        if already:
            continue
        hl_name = inverse_mapping.get(raw_name, raw_name)
        if hl_name not in params:
            params[hl_name] = _attr_type_to_param_spec(attr)
            params[hl_name]["_raw_op_param"] = raw_name

    return params


# ══════════════════════════════════════════════════════════════════
# Path 1b: fallback from OpDef when signature missing
# ══════════════════════════════════════════════════════════════════

def _fallback_from_opdef(schema: Dict[str, Any]) -> Dict[str, Any]:
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}
    inverse_mapping = tf_block.get("inverse_mapping") or {}

    input_args = op_def.get("input_args") or []
    attr_defs = op_def.get("attrs") or []

    params: Dict[str, Any] = {}

    for arg in input_args:
        if not isinstance(arg, dict) or not arg.get("name"):
            continue
        raw_name = arg["name"]
        hl_name = inverse_mapping.get(raw_name, raw_name)
        params[hl_name] = _build_tensor_spec_from_opdef(arg, raw_name)

    for attr in attr_defs:
        if not isinstance(attr, dict) or not attr.get("name"):
            continue
        raw_name = attr["name"]
        if raw_name in _SKIP_TYPE_ATTRS:
            continue
        hl_name = inverse_mapping.get(raw_name, raw_name)
        if hl_name not in params:
            params[hl_name] = _attr_type_to_param_spec(attr)
            params[hl_name]["_raw_op_param"] = raw_name

    return params


# ══════════════════════════════════════════════════════════════════
# Path 2: unresolved / no OpDef  — try harder
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# Unresolved high-level API recovery helpers
# ══════════════════════════════════════════════════════════════════

_TENSOR_LIKE_NAMES = {
    "input", "inputs", "x", "a", "b", "y",
    "values", "value", "tensor", "tensors",
    "features", "images", "image", "data",
    "logits", "params", "operand", "lhs", "rhs",
    "query", "key", "updates", "mask",
    "input_tensor", "input_data", "condition",
}

_ATTR_INT_NAMES = {
    "axis", "dim", "depth", "num_classes", "seed", "top_k",
    "k", "bins", "num_bins", "block_size",
}

_ATTR_BOOL_NAMES = {
    "stable", "keepdims", "keep_dims", "reverse", "exclusive",
    "training", "center", "fused", "sorted", "adjoint",
    "transpose_a", "transpose_b", "use_cudnn_on_gpu",
    "preserve_aspect_ratio", "antialias",
}

_ATTR_ENUM_NAMES = {
    "direction", "padding", "data_format", "mode", "method",
    "interpolation", "rounding_mode",
}

_ATTR_LIST_NAMES = {
    "shape", "perm", "strides", "dilations", "ksize",
    "size", "multiples", "paddings", "crops", "axes",
}

_DTYPE_NAMES = {
    "dtype", "out_type", "output_type", "tidx", "t",
}


def _tensor_spec_for_name(name: str, optional: bool = False) -> Dict[str, Any]:
    return {
        "origin": "input",
        "kind": "tensor_optional" if optional else "tensor",
        "dtype_choices": tensor_dtype_choices_for_param_name(name, optional),
        "shape_spec": ["TODO_SHAPE"],
    }


def _is_tensor_kind(kind: Any) -> bool:
    return kind in ("tensor", "tensor_optional", "tensor_list")


def _is_attr_kind(kind: Any) -> bool:
    return kind in (
        "bool", "int", "float", "enum", "dtype_enum",
        "int_list", "float_list", "string_optional",
    )


def _is_high_confidence_attr_name(name: str) -> bool:
    low = (name or "").lower()
    if low == "name":
        return True
    if low in _ATTR_INT_NAMES | _ATTR_BOOL_NAMES | _ATTR_ENUM_NAMES | _ATTR_LIST_NAMES | _DTYPE_NAMES:
        return True
    if low.endswith("_format"):
        return True
    return False


def _parse_default_maybe(default_value: Any) -> Any:
    # 兼容 default / default_repr / 直接传入已解析值
    if isinstance(default_value, str):
        try:
            return parse_default_repr(default_value)
        except Exception:
            return default_value
    return default_value


def _guess_param_spec_from_name(
    name: str,
    default_value: Any = None,
    has_explicit_default: bool = False,
) -> Dict[str, Any]:
    """
    High-confidence param recovery for unresolved high-level APIs.

    IMPORTANT ORDER:
    1) hard attr-name rules
    2) explicit default type rules
    3) tensor-like name rules
    4) conservative fallback
    """
    low = (name or "").lower()
    dv = _parse_default_maybe(default_value)

    # ----- meta -----
    if name == "name":
        return {
            "origin": "kwarg",
            "kind": "string_optional",
            "default": None,
            "semantic_role": "meta",
            "role": "attr",
        }

    # ----- hard attr-name rules first -----
    if low in _DTYPE_NAMES:
        spec = {
            "origin": "attr",
            "kind": "dtype_enum",
            "values": [ENUM_TODO_MARKER],
            "semantic_role": "dtype_attr",
            "role": "attr",
        }
        if isinstance(dv, str):
            spec["default"] = dv
        return spec

    if low in _ATTR_BOOL_NAMES:
        spec = {
            "origin": "attr",
            "kind": "bool",
            "semantic_role": "scalar_attr",
            "role": "attr",
        }
        if isinstance(dv, bool):
            spec["default"] = dv
        return spec

    if low in _ATTR_INT_NAMES:
        spec = {
            "origin": "attr",
            "kind": "int",
            "range": int_range_for_param_name(name),
            "semantic_role": "scalar_attr",
            "role": "attr",
        }
        if isinstance(dv, int) and not isinstance(dv, bool):
            spec["default"] = dv
        return spec

    if low in _ATTR_ENUM_NAMES or low.endswith("_format"):
        values = [ENUM_TODO_MARKER]
        if isinstance(dv, str):
            values = [dv]
        spec = {
            "origin": "attr",
            "kind": "enum",
            "values": values,
            "semantic_role": "layout_attr" if (low in ("padding", "data_format") or low.endswith("_format")) else "scalar_attr",
            "role": "attr",
        }
        if isinstance(dv, str):
            spec["default"] = dv
        return spec

    if low in _ATTR_LIST_NAMES:
        spec = {
            "origin": "attr",
            "kind": "int_list",
            "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
            "range": DEFAULT_INT_LIST_RANGE[:],
            "semantic_role": "fixed_arity_list" if low in ("strides", "dilations", "ksize") else "shape_control",
            "role": "attr",
        }
        if isinstance(dv, (list, tuple)) and all(isinstance(x, int) for x in dv):
            L = len(dv)
            spec["default"] = list(dv)
            spec["len_range"] = [L, L] if L > 0 else spec["len_range"]
        return spec

    # ----- explicit default type rules -----
    if isinstance(dv, bool):
        return {
            "origin": "attr",
            "kind": "bool",
            "default": dv,
            "semantic_role": "scalar_attr",
            "role": "attr",
        }

    if isinstance(dv, int) and not isinstance(dv, bool):
        return {
            "origin": "attr",
            "kind": "int",
            "range": int_range_for_param_name(name),
            "default": dv,
            "semantic_role": "scalar_attr",
            "role": "attr",
        }

    if isinstance(dv, float):
        return {
            "origin": "attr",
            "kind": "float",
            "range": float_range_for_param_name(name),
            "default": float(dv),
            "semantic_role": "scalar_attr",
            "role": "attr",
        }

    if isinstance(dv, str):
        return {
            "origin": "attr",
            "kind": "enum",
            "values": [dv],
            "default": dv,
            "semantic_role": "scalar_attr",
            "role": "attr",
        }

    if isinstance(dv, (list, tuple)):
        if all(isinstance(x, int) for x in dv):
            L = len(dv)
            return {
                "origin": "attr",
                "kind": "int_list",
                "len_range": [L, L] if L > 0 else DEFAULT_INT_LIST_LEN_RANGE[:],
                "range": DEFAULT_INT_LIST_RANGE[:],
                "default": list(dv),
                "semantic_role": "scalar_attr",
                "role": "attr",
            }
        if all(isinstance(x, (int, float)) for x in dv):
            L = len(dv)
            return {
                "origin": "attr",
                "kind": "float_list",
                "len_range": [L, L] if L > 0 else DEFAULT_INT_LIST_LEN_RANGE[:],
                "range": [-1.0, 1.0],
                "default": [float(x) for x in dv],
                "semantic_role": "scalar_attr",
                "role": "attr",
            }

    # ----- tensor-like names -----
    if low in _TENSOR_LIKE_NAMES:
        spec = _tensor_spec_for_name(name, optional=(has_explicit_default and dv is None))
        spec["semantic_role"] = "data_tensor"
        spec["role"] = "primary" if low in {
            "input", "inputs", "x", "values", "value", "features", "images",
            "image", "tensor", "input_tensor"
        } else "aux"
        return spec

    # names that usually indicate tensor inputs even if optional
    if low.endswith(("_tensor", "_input", "_inputs", "_values", "_image", "_images")):
        spec = _tensor_spec_for_name(name, optional=(has_explicit_default and dv is None))
        spec["semantic_role"] = "data_tensor"
        spec["role"] = "primary"
        return spec

    if low.endswith(("_indices", "_index")):
        spec = _tensor_spec_for_name(name, optional=(has_explicit_default and dv is None))
        spec["semantic_role"] = "index_input"
        spec["role"] = "aux"
        return spec

    # ----- explicit default None fallback -----
    # Only tensor-ify None defaults when the name looks tensor-like.
    if has_explicit_default and dv is None:
        if classify_param_role(name) == "primary":
            spec = _tensor_spec_for_name(name, optional=True)
            spec["semantic_role"] = "data_tensor"
            spec["role"] = "primary"
            return spec
        return {
            "origin": "attr",
            "kind": "enum",
            "values": [ENUM_TODO_MARKER],
            "default": None,
            "semantic_role": "scalar_attr",
            "role": "attr",
        }

    # ----- final conservative fallback -----
    # Required unknown params lean tensor; optional unknown params lean attr.
    if not has_explicit_default:
        spec = _tensor_spec_for_name(name, optional=False)
        spec["semantic_role"] = "data_tensor" if classify_param_role(name) == "primary" else "aux_tensor"
        spec["role"] = "primary" if classify_param_role(name) == "primary" else "aux"
        return spec

    return {
        "origin": "attr",
        "kind": "enum",
        "values": [ENUM_TODO_MARKER],
        "semantic_role": "scalar_attr",
        "role": "attr",
    }


def _split_signature_args(arg_text: str) -> List[str]:
    """
    Split 'a, axis=-1, direction=\"ASCENDING\", name=None'
    while respecting nested (), [], {}, and quoted strings.
    """
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    quote: Optional[str] = None
    escape = False

    for ch in arg_text:
        if quote is not None:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue

        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue

        if ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue

        if ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
            continue

        buf.append(ch)

    piece = "".join(buf).strip()
    if piece:
        out.append(piece)

    return out


def _extract_call_arg_text(api_name: str, doc_snippet: str) -> Optional[str]:
    """
    Find the first occurrence of:
      tf.argsort(...)
      argsort(...)
    and return the raw text between the outermost parentheses.
    """
    if not doc_snippet:
        return None

    candidates = []
    full_name = api_name.strip()
    short_name = full_name.split(".")[-1] if full_name else ""

    if full_name:
        candidates.append(full_name)
    if short_name and short_name not in candidates:
        candidates.append(short_name)

    for name in candidates:
        m = re.search(rf"{re.escape(name)}\s*\(", doc_snippet)
        if not m:
            continue

        start = m.end() - 1  # points at '('
        depth = 0
        quote: Optional[str] = None
        escape = False

        for i in range(start, len(doc_snippet)):
            ch = doc_snippet[i]

            if quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue

            if ch in ("'", '"'):
                quote = ch
                continue

            if ch == "(":
                depth += 1
                continue

            if ch == ")":
                depth -= 1
                if depth == 0:
                    return doc_snippet[start + 1:i]

    return None


def _recover_params_from_doc_signature(
    api_name: str,
    doc_snippet: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Recover params from a high-level API call signature embedded in docs.

    Example:
      tf.argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None)
    """
    params: Dict[str, Dict[str, Any]] = {}

    arg_text = _extract_call_arg_text(api_name, doc_snippet)
    if not arg_text:
        return params

    for entry in _split_signature_args(arg_text):
        if not entry or entry in {"/", "*"}:
            continue
        if entry.startswith("**") or entry.startswith("*"):
            continue

        if "=" in entry:
            pname, raw_default = entry.split("=", 1)
            pname = pname.strip()
            raw_default = raw_default.strip()
            if not pname:
                continue
            default_value = _parse_default_maybe(raw_default)
            params[pname] = _guess_param_spec_from_name(
                pname,
                default_value=default_value,
                has_explicit_default=True,
            )
        else:
            pname = entry.strip()
            if not pname:
                continue
            params[pname] = _guess_param_spec_from_name(
                pname,
                default_value=None,
                has_explicit_default=False,
            )

    return params


def _merge_recovered_params(
    base_params: Dict[str, Dict[str, Any]],
    recovered_params: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Merge doc-recovered params into classifier output.

    Rules:
    - fill missing params
    - keep existing high-confidence attr decisions
    - if doc recovery says a hard attr-name param is attr, override tensor-like mistakes
    """
    merged: Dict[str, Dict[str, Any]] = dict(base_params)

    for pname, rspec in recovered_params.items():
        if pname not in merged:
            merged[pname] = rspec
            continue

        espec = merged[pname]
        if not isinstance(espec, dict) or not isinstance(rspec, dict):
            merged[pname] = rspec
            continue

        ek = espec.get("kind")
        rk = rspec.get("kind")

        # If existing is already a good attr decision, keep it.
        if _is_attr_kind(ek):
            continue

        # If doc recovery gives a hard attr-name correction, take it.
        if _is_high_confidence_attr_name(pname) and _is_attr_kind(rk):
            merged[pname] = rspec
            continue

        # If existing is weak/unknown-ish and recovered is more concrete, take recovered.
        if ek in (None, "", "enum") and rk not in (None, "", "enum"):
            merged[pname] = rspec
            continue

        # Otherwise preserve existing classifier result.
        # (Especially for true tensor params already identified by classifier.)
        continue

    return merged


def build_params_without_opdef(
    schema: Dict[str, Any],
    llm_client: Any = None,
    doc_snippet: str = "",
) -> Dict[str, Any]:
    """
    Build params for unresolved / composite / no-OpDef APIs.

    Priority:
      1) LLM classifier if available
      2) heuristic classifier
      3) doc-signature recovery (merge/fill, not blind overwrite)
      4) python_signature direct recovery as a final weak fallback
    """
    api_name = schema.get("api_name", "")
    py_sig = schema.get("python_signature") or {}

    # 1) classifier pass
    if llm_client:
        try:
            params = classify_params_with_llm(
                api_name=api_name,
                py_sig=py_sig,
                llm_client=llm_client,
                doc_snippet=doc_snippet,
            )
        except Exception:
            params = classify_params_heuristic(api_name, py_sig)
    else:
        params = classify_params_heuristic(api_name, py_sig)

    if not isinstance(params, dict):
        params = {}

    # 2) doc-signature recovery: merge, do NOT blindly overwrite
    doc_params = _recover_params_from_doc_signature(api_name, doc_snippet)
    if doc_params:
        params = _merge_recovered_params(params, doc_params)

    # 3) final weak fallback from python_signature parameter list
    py_params = py_sig.get("parameters") or []
    for p in py_params:
        if not isinstance(p, dict):
            continue
        pname = p.get("name")
        if not pname or pname == "self":
            continue
        if pname in params:
            continue

        has_default = bool(p.get("has_default", False))
        default_value = p.get("default", p.get("default_repr"))
        params[pname] = _guess_param_spec_from_name(
            pname,
            default_value=default_value,
            has_explicit_default=has_default,
        )

    return params


# def _guess_param_spec_from_name(name: str, default_repr: Any = None) -> Dict[str, Any]:
#     lname = str(name).strip()
#     role = classify_param_role(lname)
#     default_value = _normalize_default_to_value(default_repr)

#     if lname == "name":
#         return {"origin": "kwarg", "kind": "string_optional", "default": None}

#     if role == "primary" or lname in _COMMON_TENSOR_INPUT_NAMES:
#         spec = _tensor_spec_for_name(lname, optional=(default_value is None and default_repr is not None))
#         spec["origin"] = "input"
#         return spec

#     if lname in {"axis", "axes", "dim", "dims", "reduction_indices"}:
#         return {
#             "origin": "kwarg",
#             "kind": "tensor_optional",
#             "dtype_choices": ["int32", "int64"],
#             "shape_spec": ["TODO_SHAPE"],
#         }

#     if lname in {"perm", "shape", "size", "multiples", "paddings", "begin", "end", "strides"}:
#         return {
#             "origin": "kwarg",
#             "kind": "tensor_optional",
#             "dtype_choices": ["int32", "int64"],
#             "shape_spec": ["TODO_SHAPE"],
#         }

#     if lname in _COMMON_STRING_ENUM_NAMES:
#         return {
#             "origin": "kwarg",
#             "kind": "enum",
#             "values": _COMMON_STRING_ENUM_NAMES[lname],
#             "default": default_value,
#         }

#     if lname in {"stable", "descending", "exclusive", "reverse", "keepdims", "keep_dims"}:
#         return {
#             "origin": "kwarg",
#             "kind": "bool",
#             "default": bool(default_value) if isinstance(default_value, bool) else False,
#         }

#     if re.search(r"dtype|type", lname, flags=re.I):
#         return {
#             "origin": "kwarg",
#             "kind": "dtype_enum",
#             "values": [ENUM_TODO_MARKER],
#             "default": default_value,
#         }

#     if re.search(r"direction|mode|padding|format|order", lname, flags=re.I):
#         return {
#             "origin": "kwarg",
#             "kind": "enum",
#             "values": [ENUM_TODO_MARKER],
#             "default": default_value,
#         }

#     if re.search(r"axis|dim|rank|depth|num|count|size|length|len", lname, flags=re.I):
#         rng = int_range_for_param_name(lname) or DEFAULT_INT_LIST_RANGE
#         return {
#             "origin": "kwarg",
#             "kind": "int",
#             "range": rng,
#             "default": default_value,
#         }

#     if re.search(r"alpha|beta|eps|epsilon|rate|stddev|mean|val", lname, flags=re.I):
#         rng = float_range_for_param_name(lname)
#         spec: Dict[str, Any] = {"origin": "kwarg", "kind": "float", "default": default_value}
#         if rng:
#             spec["range"] = rng
#         return spec

#     if re.search(r"indices|index|shape_control|segment_ids", lname, flags=re.I):
#         return {
#             "origin": "kwarg",
#             "kind": "tensor_optional",
#             "dtype_choices": ["int32", "int64"],
#             "shape_spec": ["TODO_SHAPE"],
#         }

#     if default_value is None:
#         return {"origin": "kwarg", "kind": "unknown_optional", "default": None}
#     if isinstance(default_value, bool):
#         return {"origin": "kwarg", "kind": "bool", "default": default_value}
#     if isinstance(default_value, int) and not isinstance(default_value, bool):
#         return {"origin": "kwarg", "kind": "int", "range": int_range_for_param_name(lname), "default": default_value}
#     if isinstance(default_value, float):
#         spec = {"origin": "kwarg", "kind": "float", "default": default_value}
#         rng = float_range_for_param_name(lname)
#         if rng:
#             spec["range"] = rng
#         return spec
#     if isinstance(default_value, str):
#         return {"origin": "kwarg", "kind": "string_optional", "default": default_value}

#     return {"origin": "kwarg", "kind": "unknown_optional", "default": default_value}


def _parse_highlevel_signature_from_doc(api_name: str, doc_snippet: str) -> List[Dict[str, Any]]:
    text = doc_snippet or ""
    if not text.strip():
        return []

    api_tail = re.escape(api_name.strip())
    patterns = [
        rf"{api_tail}\s*\(([^\n\)]*)\)",
        rf"{re.escape(api_name.split('.')[-1])}\s*\(([^\n\)]*)\)",
    ]

    args_blob: Optional[str] = None
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            args_blob = m.group(1).strip()
            if args_blob:
                break

    if not args_blob:
        return []

    parts = [p.strip() for p in re.split(r",\s*", args_blob) if p.strip()]
    out: List[Dict[str, Any]] = []
    for raw in parts:
        if raw in {"*", "/"}:
            continue
        if raw.startswith("**"):
            continue
        name = raw
        default = None
        if "=" in raw:
            name, default = raw.split("=", 1)
            name = name.strip()
            default = default.strip()
        name = name.lstrip("*").strip()
        if not name or name == "self":
            continue
        out.append({"name": name, "default": default})
    return out


# def _recover_params_from_doc_signature(api_name: str, doc_snippet: str) -> Dict[str, Any]:
#     param_entries = _parse_highlevel_signature_from_doc(api_name, doc_snippet)
#     params: Dict[str, Any] = {}
#     for entry in param_entries:
#         pname = entry["name"]
#         params[pname] = _guess_param_spec_from_name(pname, entry.get("default"))
#     return params


# def build_params_without_opdef(
#     schema: Dict[str, Any],
#     llm_client: Any = None,
#     doc_snippet: str = "",
# ) -> Dict[str, Any]:
#     api_name = schema.get("api_name", "")
#     py_sig = schema.get("python_signature") or {}

#     params: Dict[str, Any] = {}

#     if llm_client:
#         try:
#             params = classify_params_with_llm(
#                 api_name=api_name,
#                 py_sig=py_sig,
#                 llm_client=llm_client,
#                 doc_snippet=doc_snippet,
#             ) or {}
#         except Exception:
#             params = {}

#     if not params:
#         try:
#             params = classify_params_heuristic(api_name, py_sig) or {}
#         except Exception:
#             params = {}

#     if not params:
#         params = _recover_params_from_doc_signature(api_name, doc_snippet)

#     # Final fallback: if signature parameters exist but classifier returned nothing,
#     # build weak specs from names/defaults so Stage B can still route many APIs.
#     if not params:
#         py_params = py_sig.get("parameters") or []
#         for p in py_params:
#             if not isinstance(p, dict) or not p.get("name"):
#                 continue
#             pname = p["name"]
#             if pname == "self":
#                 continue
#             default_repr = p.get("default") if "default" in p else p.get("default_repr")
#             params[pname] = _guess_param_spec_from_name(pname, default_repr)

#     return params


# ══════════════════════════════════════════════════════════════════
# Primary param detection / safety net
# ══════════════════════════════════════════════════════════════════

def detect_primary_param(
    params: Dict[str, Any],
    input_args: List[Dict[str, Any]],
    api_name: str = "",
) -> Optional[str]:
    tensor_params = [
        name for name, spec in params.items()
        if isinstance(spec, dict) and spec.get("kind") in _TENSOR_KINDS
    ]
    if not tensor_params:
        return None

    primaries_by_role = [
        n for n in tensor_params
        if params[n].get("semantic_role") in ("data_tensor",) or params[n].get("role") == "primary"
    ]
    if primaries_by_role:
        input_order = {a["name"]: i for i, a in enumerate(input_args) if isinstance(a, dict) and a.get("name")}
        def _key(n: str) -> int:
            raw = params[n].get("_raw_op_param", n)
            return input_order.get(raw, input_order.get(n, 999))
        primaries_by_role.sort(key=_key)
        return primaries_by_role[0]

    primaries = [n for n in tensor_params if classify_param_role(n) == "primary"]
    if primaries:
        return primaries[0]

    return tensor_params[0]


def _repair_primary_param_if_missing(
    params: Dict[str, Any],
    primary_param: Optional[str],
) -> Optional[str]:
    if isinstance(primary_param, str) and primary_param in params:
        return primary_param

    tensor_params = [
        name for name, spec in params.items()
        if isinstance(spec, dict) and spec.get("kind") in _TENSOR_KINDS
    ]
    if not tensor_params:
        return None

    if len(tensor_params) == 1:
        return tensor_params[0]

    for cand in ("input", "input_tensor", "values", "x", "features", "images", "tensor", "data"):
        if cand in tensor_params:
            return cand

    primaries = [n for n in tensor_params if classify_param_role(n) == "primary"]
    if primaries:
        return primaries[0]

    return tensor_params[0]


# ══════════════════════════════════════════════════════════════════
# Role annotation
# ══════════════════════════════════════════════════════════════════

def annotate_param_roles(
    params: Dict[str, Any],
    primary_param: Optional[str],
) -> None:
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if name == primary_param:
            spec["role"] = "primary"
        elif spec.get("role"):
            pass
        elif spec.get("kind") in _TENSOR_KINDS:
            role = classify_param_role(name)
            spec["role"] = role if role != "unknown" else "aux"
        else:
            spec["role"] = "attr"


# ══════════════════════════════════════════════════════════════════
# Rank hints loading
# ══════════════════════════════════════════════════════════════════

def load_rank_hints_for_api(api_name, rank_index_dir):
    if rank_index_dir is None:
        from tf_schema_common import stable_rank_hints_placeholder
        return stable_rank_hints_placeholder()
    p = rank_index_dir / f"{safe_name(api_name)}.rank.json"
    if not p.exists():
        from tf_schema_common import stable_rank_hints_placeholder
        return stable_rank_hints_placeholder()
    return normalize_rank_hints(load_json(p))


def load_param_rank_details(api_name, rank_index_dir):
    if rank_index_dir is None:
        return {}
    p = rank_index_dir / f"{safe_name(api_name)}.rank.json"
    if not p.exists():
        return {}
    try:
        return load_json(p).get("param_rank_details") or {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════
# YAML skeleton builder
# ══════════════════════════════════════════════════════════════════

def build_yaml_skeleton(
    schema: Dict[str, Any],
    rank_hints: Dict[str, Any],
    param_rank_details: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    doc_snippet: str = "",
) -> Dict[str, Any]:
    api_name = schema["api_name"]
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}
    input_args = op_def.get("input_args") or []
    resolve_info = schema.get("resolve_info") or {}

    has_opdef = bool(op_def and op_def.get("input_args"))
    has_py_sig = bool(schema.get("python_signature") and (schema["python_signature"].get("parameters")))
    is_raw_ops = resolve_info.get("is_raw_ops", False)

    params: Dict[str, Any] = {}

    if is_raw_ops and has_opdef:
        params = _original_merge(schema)
    elif has_opdef and has_py_sig:
        params = merge_highlevel_with_opdef(schema)
    elif has_opdef and not has_py_sig:
        print(f"  [!] {api_name}: python_signature unavailable, using OpDef-direct fallback")
        params = _fallback_from_opdef(schema)
    else:
        params = build_params_without_opdef(schema, llm_client, doc_snippet)

    if not params and has_opdef:
        print(f"  [!!] {api_name}: all param paths failed, using _original_merge as last resort")
        params = _original_merge(schema)

    primary_param = detect_primary_param(params, input_args, api_name)
    primary_param = _repair_primary_param_if_missing(params, primary_param)
    annotate_param_roles(params, primary_param)

    op_name = tf_block.get("op_name") or resolve_info.get("raw_op_name")
    api_category = schema.get("api_category") or resolve_info.get("api_category", "unknown")
    api_module = schema.get("api_module") or resolve_info.get("api_module", "")

    y: Dict[str, Any] = {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "category": op_name or api_name.split(".")[-1],
        "api_category": api_category,
        "api_module": api_module,
        "primary_param": primary_param,
        "rank_hints": rank_hints,
        "tf": {
            "op_name": op_name,
            "raw_api_name": tf_block.get("raw_api_name"),
            "high_level_api_name": tf_block.get("high_level_api_name") or api_name,
            "schema_str": (schema.get("python_signature") or {}).get("signature_str"),
            "is_stateful": op_def.get("is_stateful"),
            "allows_uninitialized_input": op_def.get("allows_uninitialized_input"),
            "input_meta": input_args,
            "attr_meta": op_def.get("attrs") or [],
            "output_meta": op_def.get("output_args") or [],
            "param_mapping": tf_block.get("param_mapping") or {},
            "inverse_mapping": tf_block.get("inverse_mapping") or {},
        },
        "resolve_info": {
            "strategy": resolve_info.get("strategy", "unknown"),
            "confidence": resolve_info.get("confidence", 0.0),
            "is_raw_ops": is_raw_ops,
            "raw_op_name": resolve_info.get("raw_op_name"),
            "raw_op_names": resolve_info.get("raw_op_names") or [],
        },
        "shape_vars": {},
        "params": params,
        "constraints": [],
    }

    if param_rank_details:
        y["_param_rank_details"] = param_rank_details

    for _pname, spec in params.items():
        if isinstance(spec, dict):
            spec.pop("_raw_op_param", None)
            spec.pop("_shape_hint", None)

    return y


# ══════════════════════════════════════════════════════════════════
# Routability classification + quarantine output
# ══════════════════════════════════════════════════════════════════

def classify_skeleton_routability(y: Dict[str, Any]) -> Dict[str, Any]:
    params = y.get("params")
    primary_param = y.get("primary_param")
    resolve_info = y.get("resolve_info") or {}
    tf_block = y.get("tf") or {}

    reasons: List[str] = []

    if not isinstance(params, dict) or not params:
        reasons.append("empty_params")

    tensor_params: List[str] = []
    if isinstance(params, dict):
        for pname, spec in params.items():
            if isinstance(spec, dict) and _is_tensor_like_kind(spec.get("kind")):
                tensor_params.append(pname)

    if not tensor_params:
        reasons.append("no_tensor_params")

    if not isinstance(primary_param, str) or not primary_param:
        reasons.append("missing_primary_param")
    elif not isinstance(params, dict) or primary_param not in params:
        reasons.append("primary_param_not_in_params")

    has_opdef = bool((tf_block.get("input_meta") or []) or (tf_block.get("attr_meta") or []))
    has_python_signature = bool((schema_str := tf_block.get("schema_str")))

    return {
        "routable": not reasons,
        "reasons": reasons,
        "tensor_params": tensor_params,
        "params_count": len(params) if isinstance(params, dict) else 0,
        "tensor_params_count": len(tensor_params),
        "primary_param": primary_param,
        "resolve_strategy": resolve_info.get("strategy"),
        "confidence": resolve_info.get("confidence"),
        "has_opdef": has_opdef,
        "has_python_signature": has_python_signature,
        "schema_str_present": bool(schema_str),
    }


def write_unroutable_skeleton(
    y: Dict[str, Any],
    info: Dict[str, Any],
    empty_param_dir: Path,
) -> None:
    empty_param_dir.mkdir(parents=True, exist_ok=True)

    api_name = y.get("api_name", "unknown.api")
    yaml_path = empty_param_dir / f"{safe_name(api_name)}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(y, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    row = {
        "api_name": api_name,
        "reason": ";".join(info.get("reasons") or []),
        "resolve_strategy": info.get("resolve_strategy"),
        "confidence": info.get("confidence"),
        "params_count": info.get("params_count"),
        "tensor_params_count": info.get("tensor_params_count"),
        "primary_param": info.get("primary_param"),
        "tensor_params": info.get("tensor_params"),
        "yaml_file": yaml_path.name,
    }

    tsv_path = empty_param_dir / "EMPTY_PARAM_REPORT.tsv"
    jsonl_path = empty_param_dir / "EMPTY_PARAM_REPORT.jsonl"

    need_header = not tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8") as f:
        if need_header:
            f.write(
                "api_name\treason\tresolve_strategy\tconfidence\tparams_count\t"
                "tensor_params_count\tprimary_param\ttensor_params\tyaml_file\n"
            )
        f.write(
            f"{row['api_name']}\t{row['reason']}\t{row['resolve_strategy']}\t"
            f"{row['confidence']}\t{row['params_count']}\t{row['tensor_params_count']}\t"
            f"{row['primary_param']}\t{','.join(row['tensor_params'] or [])}\t{row['yaml_file']}\n"
        )

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified YAML skeleton builder for raw_ops + high-level TF APIs with empty-param quarantine."
    )
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--out_dir", default="./tf_yaml_skeleton")
    ap.add_argument("--empty_param_dir", default="./02_empty_param")
    ap.add_argument("--rank_index_dir", default=None)
    ap.add_argument("--doc_dir", default=None)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--llm_api_key", default=None)
    args = ap.parse_args()

    src = Path(args.schema_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    empty_param_dir = Path(args.empty_param_dir).resolve()
    empty_param_dir.mkdir(parents=True, exist_ok=True)

    rank_dir = Path(args.rank_index_dir).resolve() if args.rank_index_dir else None
    doc_dir = Path(args.doc_dir).resolve() if args.doc_dir else None

    llm_client = None
    if args.llm_base_url:
        from llm_doc_rank_extractor import LLMClient
        llm_client = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key or "",
        )
        print(f"[i] LLM enabled: {args.llm_base_url}")

    files = iter_files(src, ".json")
    if not files:
        raise SystemExit(f"No json files found under: {src}")

    count = 0
    stats = {
        "raw_ops": 0,
        "resolved_sig": 0,
        "resolved_opdef": 0,
        "unresolved": 0,
        "empty_param": 0,
    }

    for jp in files:
        schema = load_json(jp)
        if not isinstance(schema, dict) or not schema.get("api_name"):
            print(f"[!] skip invalid: {jp}")
            continue

        api_name = schema["api_name"]
        rank_hints = load_rank_hints_for_api(api_name, rank_dir)
        param_rank_details = load_param_rank_details(api_name, rank_dir)

        doc_snippet = ""
        if doc_dir:
            doc_path = doc_dir / f"{safe_name(api_name)}.txt"
            if doc_path.exists():
                doc_snippet = doc_path.read_text(encoding="utf-8", errors="ignore")[:4000]

        y = build_yaml_skeleton(
            schema,
            rank_hints,
            param_rank_details,
            llm_client=llm_client,
            doc_snippet=doc_snippet,
        )

        route_info = classify_skeleton_routability(y)

        ri = y.get("resolve_info") or {}
        if ri.get("is_raw_ops"):
            stats["raw_ops"] += 1
        elif bool((schema.get("python_signature") or {}).get("parameters")):
            stats["resolved_sig"] += 1
        elif ri.get("raw_op_name"):
            stats["resolved_opdef"] += 1
        else:
            stats["unresolved"] += 1

        if not route_info["routable"]:
            write_unroutable_skeleton(y, route_info, empty_param_dir)
            stats["empty_param"] += 1
            print(
                f"[empty] {api_name} → {empty_param_dir / (safe_name(api_name) + '.yaml')}  "
                f"reasons={route_info['reasons']}"
            )
            continue

        out_path = out_dir / f"{safe_name(api_name)}.yaml"
        out_path.write_text(
            yaml.safe_dump(y, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        pp = y.get("primary_param", "?")
        n_params = len(y.get("params") or {})
        print(f"[+] {api_name} → {out_path}  primary={pp}  params={n_params}")
        count += 1

    print(
        f"\n[done] wrote {count} yaml skeletons "
        f"(raw_ops={stats['raw_ops']}, "
        f"resolved_sig={stats['resolved_sig']}, "
        f"resolved_opdef={stats['resolved_opdef']}, "
        f"unresolved={stats['unresolved']}, "
        f"empty_param={stats['empty_param']})"
    )


if __name__ == "__main__":
    main()
