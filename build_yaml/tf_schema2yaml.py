#!/usr/bin/env python3
"""
tf_schema_json_to_yaml_skeleton.py  –  Stage B: schema JSON → YAML skeleton.

Changes vs. original
---------------------
1. Adds `primary_param` field to the YAML output — automatically detected
   from the input_args list using `classify_param_role()`.
2. Adds per-param `role` annotation (primary / aux / attr) so downstream
   stages (C/D) know which params drive the shape structure.
3. Cleaner merge logic: when opdef and signature both exist, opdef
   input_args take priority for tensor params, signature fills in rest.
4. `rank_hints` now carries the `param_rank_details` from the doc
   extractor (if present), giving Stage C per-param rank context.
"""
from __future__ import annotations

import argparse
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

GENERATOR_BLOCK = {
    "stage": "B-json-to-yaml-skeleton-tf",
    "version": "2026-03-23-tf-v2",
}


# ── signature-only fallback classifier ───────────────────────────

def classify_from_signature_param(param: Dict[str, Any]) -> Dict[str, Any]:
    name = param["name"]
    has_default = bool(param.get("has_default", False))
    default_value = parse_default_repr(param.get("default"))
    lname = name.lower()

    if name == "name":
        return {"origin": "kwarg", "kind": "string_optional", "default": None}

    if isinstance(default_value, bool):
        return {"origin": "attr", "kind": "bool", "default": default_value}

    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return {"origin": "attr", "kind": "int", "range": int_range_for_param_name(name), "default": default_value}

    if isinstance(default_value, float):
        return {"origin": "attr", "kind": "float", "range": float_range_for_param_name(name), "default": default_value}

    if isinstance(default_value, str):
        if lname.endswith("format") or lname in ("padding", "mode", "data_format"):
            return {"origin": "attr", "kind": "enum", "values": [default_value]}
        return {"origin": "attr", "kind": "string_optional", "default": default_value}

    if isinstance(default_value, (list, tuple)):
        if all(isinstance(x, int) for x in default_value):
            L = len(default_value)
            return {
                "origin": "attr",
                "kind": "int_list",
                "len_range": [L, L] if L > 0 else DEFAULT_INT_LIST_LEN_RANGE[:],
                "range": DEFAULT_INT_LIST_RANGE[:],
                "default": list(default_value),
            }
        if all(isinstance(x, float) for x in default_value):
            L = len(default_value)
            return {
                "origin": "attr",
                "kind": "float_list",
                "len_range": [L, L] if L > 0 else DEFAULT_INT_LIST_LEN_RANGE[:],
                "range": [-1.0, 1.0],
                "default": list(default_value),
            }

    if name in ("input", "x", "features", "images", "value", "tensor", "a", "b", "filter", "filters"):
        return {
            "origin": "input",
            "kind": "tensor_optional" if has_default else "tensor",
            "dtype_choices": tensor_dtype_choices_for_param_name(name, has_default),
            "shape_spec": ["TODO_SHAPE"],
        }

    if lname in ("strides", "dilations", "ksize", "explicit_paddings", "perm", "shape"):
        return {
            "origin": "attr",
            "kind": "int_list",
            "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
            "range": DEFAULT_INT_LIST_RANGE[:],
        }

    if lname in ("padding", "data_format"):
        return {"origin": "attr", "kind": "enum", "values": [ENUM_TODO_MARKER]}

    if not has_default:
        return {
            "origin": "input",
            "kind": "tensor",
            "dtype_choices": tensor_dtype_choices_for_param_name(name, False),
            "shape_spec": ["TODO_SHAPE"],
        }
    return {"origin": "attr", "kind": "enum", "values": [ENUM_TODO_MARKER]}


# ── opdef attr → param spec ──────────────────────────────────────

def _attr_type_to_param_spec(attr: Dict[str, Any]) -> Dict[str, Any]:
    t = (attr.get("type") or "").strip()
    has_default = bool(attr.get("has_default"))
    default_value = attr.get("default_value")
    allowed = attr.get("allowed_values")
    spec: Dict[str, Any]

    if t == "bool":
        spec = {"origin": "attr", "kind": "bool"}
        if has_default and isinstance(default_value, bool):
            spec["default"] = default_value
        return spec

    if t == "int":
        spec = {"origin": "attr", "kind": "int", "range": [-1, 8]}
        if has_default and isinstance(default_value, int):
            spec["default"] = default_value
        return spec

    if t == "float":
        spec = {"origin": "attr", "kind": "float", "range": [-1.0, 1.0]}
        if has_default and isinstance(default_value, (int, float)):
            spec["default"] = float(default_value)
        return spec

    if t == "string":
        values = [ENUM_TODO_MARKER]
        if isinstance(allowed, dict) and allowed.get("kind") == "list" and isinstance(allowed.get("s"), list):
            values = [str(x) for x in allowed["s"]] or values
        spec = {"origin": "attr", "kind": "enum", "values": values}
        if has_default and isinstance(default_value, str):
            if default_value not in spec["values"]:
                spec["values"] = list(spec["values"]) + [default_value]
            spec["default"] = default_value
        return spec

    if t == "type":
        values = [ENUM_TODO_MARKER]
        if isinstance(allowed, dict) and allowed.get("kind") == "list" and isinstance(allowed.get("type"), list):
            values = [str(x) for x in allowed["type"]] or values
        spec = {"origin": "attr", "kind": "dtype_enum", "values": values}
        if has_default and isinstance(default_value, str):
            if default_value not in spec["values"]:
                spec["values"] = list(spec["values"]) + [default_value]
            spec["default"] = default_value
        return spec

    if t.startswith("list("):
        inner = t[len("list("):-1] if t.endswith(")") else ""
        if inner == "int":
            spec = {
                "origin": "attr",
                "kind": "int_list",
                "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
                "range": DEFAULT_INT_LIST_RANGE[:],
            }
            if has_default and isinstance(default_value, list) and all(isinstance(x, int) for x in default_value):
                L = len(default_value)
                spec["default"] = default_value
                spec["len_range"] = [L, L] if L > 0 else spec["len_range"]
            return spec
        if inner == "float":
            spec = {
                "origin": "attr",
                "kind": "float_list",
                "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
                "range": [-1.0, 1.0],
            }
            if has_default and isinstance(default_value, list) and all(isinstance(x, (int, float)) for x in default_value):
                L = len(default_value)
                spec["default"] = [float(x) for x in default_value]
                spec["len_range"] = [L, L] if L > 0 else spec["len_range"]
            return spec
        if inner == "string":
            spec = {
                "origin": "attr",
                "kind": "string_list",
                "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
            }
            if has_default and isinstance(default_value, list) and all(isinstance(x, str) for x in default_value):
                L = len(default_value)
                spec["default"] = default_value
                spec["len_range"] = [L, L] if L > 0 else spec["len_range"]
            return spec

    return {"origin": "attr", "kind": "enum", "values": [ENUM_TODO_MARKER]}


# ── merge signature + opdef → params dict ────────────────────────

def merge_signature_and_opdef(schema: Dict[str, Any]) -> Dict[str, Any]:
    py_sig = schema.get("python_signature") or {}
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}

    py_params = py_sig.get("parameters") or []
    py_names = [p["name"] for p in py_params if isinstance(p, dict) and p.get("name")]

    input_args = op_def.get("input_args") or []
    attr_defs = op_def.get("attrs") or []

    input_map = {x["name"]: x for x in input_args if isinstance(x, dict) and x.get("name")}
    attr_map = {x["name"]: x for x in attr_defs if isinstance(x, dict) and x.get("name")}

    params: Dict[str, Any] = {}
    consumed_inputs = set()
    consumed_attrs = set()

    if py_names:
        for name in py_names:
            if name == "name":
                params[name] = {"origin": "kwarg", "kind": "string_optional", "default": None}
                continue
            if name in input_map:
                meta = input_map[name]
                type_name = meta.get("type_name")
                if type_name:
                    params[name] = {
                        "origin": "input",
                        "kind": "tensor",
                        "dtype_choices": [type_name],
                        "shape_spec": ["TODO_SHAPE"],
                    }
                elif meta.get("type_attr"):
                    params[name] = {
                        "origin": "input",
                        "kind": "tensor",
                        "dtype_from_attr": meta["type_attr"],
                        "shape_spec": ["TODO_SHAPE"],
                    }
                elif meta.get("type_list_attr"):
                    params[name] = {
                        "origin": "input",
                        "kind": "tensor_list",
                        "dtype_from_attr": meta["type_list_attr"],
                        "shape_spec": ["TODO_SHAPE"],
                    }
                else:
                    params[name] = {
                        "origin": "input",
                        "kind": "tensor",
                        "dtype_choices": ["float32", "float64"],
                        "shape_spec": ["TODO_SHAPE"],
                    }
                consumed_inputs.add(name)
                continue
            if name in attr_map:
                params[name] = _attr_type_to_param_spec(attr_map[name])
                consumed_attrs.add(name)
                continue

            param_info = next((p for p in py_params if p.get("name") == name), None)
            if param_info is not None:
                params[name] = classify_from_signature_param(param_info)

    # append unconsumed opdef INPUT ARGS as tensor params
    # This is CRITICAL for tf.raw_ops.* where Python signature is empty
    # but OpDef input_args has the actual tensor inputs.
    for name, meta in input_map.items():
        if name in consumed_inputs or name in params:
            continue
        type_name = meta.get("type_name")
        if type_name:
            params[name] = {
                "origin": "input",
                "kind": "tensor",
                "dtype_choices": [type_name],
                "shape_spec": ["TODO_SHAPE"],
            }
        elif meta.get("type_attr"):
            params[name] = {
                "origin": "input",
                "kind": "tensor",
                "dtype_from_attr": meta["type_attr"],
                "shape_spec": ["TODO_SHAPE"],
            }
        elif meta.get("type_list_attr"):
            params[name] = {
                "origin": "input",
                "kind": "tensor_list",
                "dtype_from_attr": meta["type_list_attr"],
                "shape_spec": ["TODO_SHAPE"],
            }
        else:
            params[name] = {
                "origin": "input",
                "kind": "tensor",
                "dtype_choices": ["float32", "float64"],
                "shape_spec": ["TODO_SHAPE"],
            }

    # append unconsumed opdef attrs that are visible runtime knobs
    for name, attr in attr_map.items():
        if name in consumed_attrs or name == "T":
            continue
        if name not in params:
            params[name] = _attr_type_to_param_spec(attr)

    return params


# ── primary_param detection ──────────────────────────────────────

def detect_primary_param(
    params: Dict[str, Any],
    input_args: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Pick the single primary tensor parameter for this API.

    Strategy:
    1. Among tensor params, prefer those whose role == "primary".
    2. If multiple primaries, prefer the one that comes first in input_args.
    3. If no explicit primary, use the first tensor param in input_args order.
    """
    tensor_params = [
        name for name, spec in params.items()
        if isinstance(spec, dict) and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list")
    ]
    if not tensor_params:
        return None

    # Build input_arg ordering
    input_order = {a["name"]: i for i, a in enumerate(input_args) if isinstance(a, dict) and a.get("name")}

    # Classify roles
    primaries = [n for n in tensor_params if classify_param_role(n) == "primary"]
    if primaries:
        # Pick the one that comes earliest in input_args
        primaries.sort(key=lambda n: input_order.get(n, 999))
        return primaries[0]

    # No explicit primary — use first tensor in input_args order
    tensor_params.sort(key=lambda n: input_order.get(n, 999))
    return tensor_params[0]


# ── per-param role annotation ────────────────────────────────────

def annotate_param_roles(
    params: Dict[str, Any],
    primary_param: Optional[str],
) -> None:
    """Add a `role` field to each param spec in-place."""
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if name == primary_param:
            spec["role"] = "primary"
        elif spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
            role = classify_param_role(name)
            spec["role"] = role if role != "unknown" else "aux"
        else:
            spec["role"] = "attr"


# ── rank hints loading ───────────────────────────────────────────

def load_rank_hints_for_api(api_name: str, rank_index_dir: Optional[Path]) -> Dict[str, Any]:
    if rank_index_dir is None:
        from tf_schema_common import stable_rank_hints_placeholder
        return stable_rank_hints_placeholder()
    p = rank_index_dir / f"{safe_name(api_name)}.rank.json"
    if not p.exists():
        from tf_schema_common import stable_rank_hints_placeholder
        return stable_rank_hints_placeholder()
    raw = load_json(p)
    return normalize_rank_hints(raw)


def load_param_rank_details(api_name: str, rank_index_dir: Optional[Path]) -> Dict[str, Any]:
    """Load the param_rank_details block from the rank.json (new field)."""
    if rank_index_dir is None:
        return {}
    p = rank_index_dir / f"{safe_name(api_name)}.rank.json"
    if not p.exists():
        return {}
    try:
        raw = load_json(p)
        return raw.get("param_rank_details") or {}
    except Exception:
        return {}


# ── YAML skeleton builder ────────────────────────────────────────

def build_yaml_skeleton(schema: Dict[str, Any], rank_hints: Dict[str, Any],
                        param_rank_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    api_name = schema["api_name"]
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}
    input_args = op_def.get("input_args") or []

    params = merge_signature_and_opdef(schema)

    # Detect primary param
    primary_param = detect_primary_param(params, input_args)

    # Annotate roles
    annotate_param_roles(params, primary_param)

    y: Dict[str, Any] = {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "category": tf_block.get("op_name") or api_name.split(".")[-1],
        "primary_param": primary_param,
        "rank_hints": rank_hints,
        "tf": {
            "op_name": tf_block.get("op_name"),
            "raw_api_name": tf_block.get("raw_api_name"),
            "schema_str": (schema.get("python_signature") or {}).get("signature_str"),
            "is_stateful": op_def.get("is_stateful"),
            "allows_uninitialized_input": op_def.get("allows_uninitialized_input"),
            "input_meta": input_args,
            "attr_meta": op_def.get("attrs") or [],
            "output_meta": op_def.get("output_args") or [],
        },
        "shape_vars": {},
        "params": params,
        "constraints": [],
    }

    # If param_rank_details are available, attach them for Stage C reference
    if param_rank_details:
        y["_param_rank_details"] = param_rank_details

    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_json", required=True,
                    help="schema json file or directory produced by export_tf_schema.py")
    ap.add_argument("--out_dir", default="./tf_yaml_skeleton",
                    help="output dir for yaml skeletons")
    ap.add_argument("--rank_index_dir", default=None,
                    help="directory containing *.rank.json files")
    args = ap.parse_args()

    src = Path(args.schema_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = Path(args.rank_index_dir).resolve() if args.rank_index_dir else None

    files = iter_files(src, ".json")
    if not files:
        raise SystemExit(f"No json files found under: {src}")

    count = 0
    for jp in files:
        schema = load_json(jp)
        if not isinstance(schema, dict) or not schema.get("api_name"):
            print(f"[!] skip invalid schema json: {jp}")
            continue
        api_name = schema["api_name"]
        rank_hints = load_rank_hints_for_api(api_name, rank_dir)
        param_rank_details = load_param_rank_details(api_name, rank_dir)
        y = build_yaml_skeleton(schema, rank_hints, param_rank_details)
        out_path = out_dir / f"{safe_name(api_name)}.yaml"
        out_path.write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
        pp = y.get("primary_param", "?")
        print(f"[+] {api_name} -> {out_path}  primary_param={pp}")
        count += 1

    print(f"[done] wrote {count} yaml skeleton files")


if __name__ == "__main__":
    main()
