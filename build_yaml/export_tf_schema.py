#!/usr/bin/env python3
"""
export_tf_schema.py  –  Stage A: extract TF raw_ops schema to JSON.

For each API in the input list, produces a JSON file containing:
  - python_signature  (from inspect.signature)
  - tf.op_def         (from the C++ OpDef registry)

This script is unchanged from the original; it is included here for
completeness so the full pipeline can be run end-to-end.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tf_schema_common import dump_json, load_api_list, decode_bytes_maybe

GENERATOR_BLOCK = {
    "stage": "A-export-tf-schema",
    "version": "2026-03-23-tf-v2",
}


# ── resolve helpers ──────────────────────────────────────────────

def resolve_obj_from_qualname(qualname: str):
    qualname = qualname.strip()
    if qualname.endswith("()"):
        qualname = qualname[:-2]
    module_name, _, attr_name = qualname.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid qualified name: {qualname}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def python_signature_to_dict(sig: inspect.Signature) -> Dict[str, Any]:
    params = []
    for idx, (name, param) in enumerate(sig.parameters.items()):
        has_default = param.default is not inspect._empty
        default_repr = None if not has_default else repr(param.default)
        has_annot = param.annotation is not inspect._empty
        annot_repr = None if not has_annot else repr(param.annotation)
        params.append(
            {
                "name": name,
                "position": idx,
                "kind": param.kind.name,
                "has_default": has_default,
                "default": default_repr,
                "annotation": annot_repr,
            }
        )
    ret_annot = None if sig.return_annotation is inspect._empty else repr(sig.return_annotation)
    return {
        "signature_str": str(sig),
        "parameters": params,
        "return_annotation": ret_annot,
    }


# ── OpDef helpers ────────────────────────────────────────────────

def _dtype_enum_name(enum_val: int) -> Optional[str]:
    if not enum_val:
        return None
    try:
        from tensorflow.python.framework import dtypes  # type: ignore
        return dtypes.as_dtype(enum_val).name
    except Exception:
        return None


def _shape_to_dict(shape) -> Dict[str, Any]:
    dims = []
    for d in getattr(shape, "dim", []) or []:
        dims.append({"size": getattr(d, "size", None), "name": getattr(d, "name", None) or None})
    return {
        "unknown_rank": bool(getattr(shape, "unknown_rank", False)),
        "dims": dims,
    }


def _attr_value_to_python(attr_value) -> Any:
    if attr_value is None:
        return None
    which = None
    try:
        which = attr_value.WhichOneof("value")
    except Exception:
        which = None
    if which is None:
        return None
    if which == "s":
        return decode_bytes_maybe(attr_value.s)
    if which == "i":
        return int(attr_value.i)
    if which == "f":
        return float(attr_value.f)
    if which == "b":
        return bool(attr_value.b)
    if which == "type":
        return _dtype_enum_name(int(attr_value.type)) or int(attr_value.type)
    if which == "shape":
        return _shape_to_dict(attr_value.shape)
    if which == "list":
        lv = attr_value.list
        out: Dict[str, Any] = {"kind": "list"}
        if getattr(lv, "s", None):
            out["s"] = [decode_bytes_maybe(x) for x in lv.s]
        if getattr(lv, "i", None):
            out["i"] = [int(x) for x in lv.i]
        if getattr(lv, "f", None):
            out["f"] = [float(x) for x in lv.f]
        if getattr(lv, "b", None):
            out["b"] = [bool(x) for x in lv.b]
        if getattr(lv, "type", None):
            out["type"] = [(_dtype_enum_name(int(x)) or int(x)) for x in lv.type]
        if getattr(lv, "shape", None):
            out["shape"] = [_shape_to_dict(x) for x in lv.shape]
        if getattr(lv, "tensor", None):
            out["tensor_count"] = len(lv.tensor)
        if getattr(lv, "func", None):
            out["func_count"] = len(lv.func)
        return out
    if which == "tensor":
        return {"kind": "tensor_proto"}
    if which == "func":
        return {"kind": "func", "name": getattr(attr_value.func, "name", None)}
    if which == "placeholder":
        return {"kind": "placeholder", "value": decode_bytes_maybe(attr_value.placeholder)}
    return {"kind": which}


def _argdef_to_dict(arg) -> Dict[str, Any]:
    type_enum = int(getattr(arg, "type", 0) or 0)
    return {
        "name": arg.name,
        "description": getattr(arg, "description", None) or None,
        "type_enum": type_enum,
        "type_name": _dtype_enum_name(type_enum),
        "type_attr": getattr(arg, "type_attr", None) or None,
        "number_attr": getattr(arg, "number_attr", None) or None,
        "type_list_attr": getattr(arg, "type_list_attr", None) or None,
        "is_ref": bool(getattr(arg, "is_ref", False)),
    }


def _attrdef_to_dict(attr) -> Dict[str, Any]:
    has_default = False
    has_allowed_values = False
    default_value = None
    allowed_values = None
    try:
        has_default = attr.HasField("default_value")
        if has_default:
            default_value = _attr_value_to_python(attr.default_value)
    except Exception:
        has_default = False
    try:
        has_allowed_values = attr.HasField("allowed_values")
        if has_allowed_values:
            allowed_values = _attr_value_to_python(attr.allowed_values)
    except Exception:
        has_allowed_values = False
    return {
        "name": attr.name,
        "type": attr.type,
        "description": getattr(attr, "description", None) or None,
        "has_default": has_default,
        "default_value": default_value,
        "has_minimum": bool(getattr(attr, "has_minimum", False)),
        "minimum": int(getattr(attr, "minimum", 0)) if bool(getattr(attr, "has_minimum", False)) else None,
        "has_allowed_values": has_allowed_values,
        "allowed_values": allowed_values,
    }


def opdef_to_dict(opdef) -> Dict[str, Any]:
    return {
        "name": opdef.name,
        "summary": getattr(opdef, "summary", None) or None,
        "description": getattr(opdef, "description", None) or None,
        "is_commutative": bool(getattr(opdef, "is_commutative", False)),
        "is_aggregate": bool(getattr(opdef, "is_aggregate", False)),
        "is_stateful": bool(getattr(opdef, "is_stateful", False)),
        "allows_uninitialized_input": bool(getattr(opdef, "allows_uninitialized_input", False)),
        "input_args": [_argdef_to_dict(x) for x in opdef.input_arg],
        "output_args": [_argdef_to_dict(x) for x in opdef.output_arg],
        "attrs": [_attrdef_to_dict(x) for x in opdef.attr],
    }


def get_tf_opdef(op_name: str):
    from tensorflow.python.framework import op_def_registry  # type: ignore
    return op_def_registry.get(op_name)


# ── main export logic ────────────────────────────────────────────

def export_tf_api_schema(api_name: str, out_dir: Path) -> None:
    api_info: Dict[str, Any] = {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "python_signature": None,
        "tf": None,
        "error": None,
    }

    obj = None
    try:
        obj = resolve_obj_from_qualname(api_name)
        sig = inspect.signature(obj)
        api_info["python_signature"] = python_signature_to_dict(sig)
    except Exception as e:
        api_info["error"] = f"inspect.signature failed: {e}"

    op_name = api_name.split(".")[-1]
    try:
        opdef = get_tf_opdef(op_name)
        if opdef is not None:
            api_info["tf"] = {
                "op_name": op_name,
                "raw_api_name": api_name,
                "op_def": opdef_to_dict(opdef),
            }
        else:
            api_info["error"] = (api_info["error"] + " | " if api_info["error"] else "") + "no opdef found"
    except Exception as e:
        api_info["error"] = (api_info["error"] + " | " if api_info["error"] else "") + f"opdef load failed: {e}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{api_name.replace('.', '_')}_schema.json"
    dump_json(out_path, api_info)
    print(f"[+] saved schema for {api_name} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_list", required=True,
                    help="txt/json/pkl, each entry is a qualified TensorFlow API name")
    ap.add_argument("--out_dir", default="./tf_api_schema",
                    help="output dir for schema json files")
    args = ap.parse_args()

    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        raise SystemExit(f"TensorFlow import failed: {e}")

    api_list = load_api_list(args.api_list)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] loaded {len(api_list)} apis")
    for api in api_list:
        try:
            export_tf_api_schema(api, out_dir)
        except Exception as e:
            out_path = out_dir / f"{api.replace('.', '_')}_schema.json"
            dump_json(
                out_path,
                {
                    "generator": GENERATOR_BLOCK,
                    "api_name": api,
                    "python_signature": None,
                    "tf": None,
                    "error": f"fatal export error: {e}",
                },
            )
            print(f"[!] failed: {api}: {e}")


if __name__ == "__main__":
    main()
