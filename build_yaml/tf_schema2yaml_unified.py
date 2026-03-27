#!/usr/bin/env python3
"""
tf_schema2yaml_unified.py  –  Stage B: unified schema JSON → YAML skeleton.

==========================================================================
FIX LOG (2026-03-27)
==========================================================================

BUG: When inspect.signature() fails for a high-level API (common with
TF generated functions like tf.nn.conv2d), python_signature is null.
The original merge_highlevel_with_opdef() depended entirely on py_names
from the Python signature.  When py_names was empty, it returned
params={}, primary_param=null — breaking everything downstream.

FIX: Three-layer fallback in build_yaml_skeleton():

  Layer 1: py_names available + OpDef → merge_highlevel_with_opdef()
           (uses Python param names, translates via param_mapping)

  Layer 2: py_names empty but OpDef available → _fallback_from_opdef()
           (uses OpDef input_args/attrs directly, translates names
            via inverse_mapping where possible)

  Layer 3: no OpDef at all → build_params_without_opdef()
           (LLM or heuristic classification from signature/doc)

  Safety net: if ALL layers produce empty params, we still populate
  from OpDef input_args as a last resort.

==========================================================================
USAGE  (unchanged)
==========================================================================

  python tf_schema2yaml_unified.py \\
      --schema_json ./tf_api_schema \\
      --out_dir ./tf_yaml_skeleton \\
      --rank_index_dir ./tf_rank_hints
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

# Import the original skeleton builder's helpers for raw_ops path
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
    "version": "2026-03-27-tf-v4",
}

# ── Internal type attrs to skip when building params from OpDef ──
_SKIP_TYPE_ATTRS = {
    "T", "Tparams", "Tindices", "Taxis", "SrcT", "DstT",
    "Tidx", "Tshape", "Tpaddings", "Tsegmentids",
    "out_type",  # sometimes a type attr, not a user-facing param
}


# ══════════════════════════════════════════════════════════════════
# Helper: build tensor spec from OpDef input_arg metadata
# ══════════════════════════════════════════════════════════════════

def _build_tensor_spec_from_opdef(
    meta: Dict[str, Any],
    raw_name: str,
) -> Dict[str, Any]:
    """Build a tensor param spec from an OpDef input_arg entry."""
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


# ══════════════════════════════════════════════════════════════════
# Path 1: Resolved API WITH Python signature available
# ══════════════════════════════════════════════════════════════════

def merge_highlevel_with_opdef(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build params dict for a high-level API that has been resolved to a
    raw_op AND whose Python signature is available.

    After processing py_names, appends any unconsumed OpDef entries.
    """
    py_sig = schema.get("python_signature") or {}
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}
    param_mapping = tf_block.get("param_mapping") or {}
    inverse_mapping = tf_block.get("inverse_mapping") or {}

    py_params = py_sig.get("parameters") or []
    py_names = [p["name"] for p in py_params
                if isinstance(p, dict) and p.get("name")]

    input_args = op_def.get("input_args") or []
    attr_defs = op_def.get("attrs") or []

    input_map = {x["name"]: x for x in input_args
                 if isinstance(x, dict) and x.get("name")}
    attr_map = {x["name"]: x for x in attr_defs
                if isinstance(x, dict) and x.get("name")}

    params: Dict[str, Any] = {}
    consumed_raw_inputs = set()
    consumed_raw_attrs = set()

    for py_name in py_names:
        if py_name == "name":
            params[py_name] = {
                "origin": "kwarg", "kind": "string_optional", "default": None
            }
            continue
        if py_name == "self":
            continue

        # Check param_mapping first
        raw_name = param_mapping.get(py_name)
        # Also try: py_name IS the raw_op name (identical names)
        if not raw_name:
            if py_name in input_map:
                raw_name = py_name
            elif py_name in attr_map:
                raw_name = py_name

        if raw_name and raw_name in input_map:
            params[py_name] = _build_tensor_spec_from_opdef(
                input_map[raw_name], raw_name
            )
            consumed_raw_inputs.add(raw_name)
            continue

        if raw_name and raw_name in attr_map:
            spec = _attr_type_to_param_spec(attr_map[raw_name])
            spec["_raw_op_param"] = raw_name
            params[py_name] = spec
            consumed_raw_attrs.add(raw_name)
            continue

        # No mapping → classify from Python signature
        param_info = next(
            (p for p in py_params if p.get("name") == py_name), None
        )
        if param_info is not None:
            params[py_name] = classify_from_signature_param(param_info)

    # ── Append unconsumed OpDef input_args ────────────────────────
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

    # ── Append unconsumed OpDef attrs ─────────────────────────────
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
# Path 1b: Fallback when signature is MISSING but OpDef exists
# ══════════════════════════════════════════════════════════════════

def _fallback_from_opdef(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build params directly from OpDef input_args and attrs.
    Used when inspect.signature() failed but the API was resolved
    to a raw_op with OpDef.

    Uses inverse_mapping to translate raw_op param names to high-level
    names (e.g., 'filter' → 'filters' for tf.nn.conv2d).
    """
    tf_block = schema.get("tf") or {}
    op_def = tf_block.get("op_def") or {}
    inverse_mapping = tf_block.get("inverse_mapping") or {}

    input_args = op_def.get("input_args") or []
    attr_defs = op_def.get("attrs") or []

    params: Dict[str, Any] = {}

    # ── Input args → tensor params ────────────────────────────────
    for arg in input_args:
        if not isinstance(arg, dict) or not arg.get("name"):
            continue
        raw_name = arg["name"]
        hl_name = inverse_mapping.get(raw_name, raw_name)
        params[hl_name] = _build_tensor_spec_from_opdef(arg, raw_name)

    # ── Attr defs → attr params ───────────────────────────────────
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
# Path 2: Unresolved API (no OpDef)
# ══════════════════════════════════════════════════════════════════

def build_params_without_opdef(
    schema: Dict[str, Any],
    llm_client: Any = None,
    doc_snippet: str = "",
) -> Dict[str, Any]:
    api_name = schema.get("api_name", "")
    py_sig = schema.get("python_signature") or {}
    if llm_client:
        return classify_params_with_llm(
            api_name=api_name, py_sig=py_sig,
            llm_client=llm_client, doc_snippet=doc_snippet,
        )
    else:
        return classify_params_heuristic(api_name, py_sig)


# ══════════════════════════════════════════════════════════════════
# Primary param detection
# ══════════════════════════════════════════════════════════════════

def detect_primary_param(
    params: Dict[str, Any],
    input_args: List[Dict[str, Any]],
    api_name: str = "",
) -> Optional[str]:
    tensor_params = [
        name for name, spec in params.items()
        if isinstance(spec, dict)
        and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list")
    ]
    if not tensor_params:
        return None

    # semantic_role first
    primaries_by_role = [
        n for n in tensor_params
        if params[n].get("semantic_role") in ("data_tensor",)
        or params[n].get("role") == "primary"
    ]
    if primaries_by_role:
        input_order = {a["name"]: i for i, a in enumerate(input_args)
                       if isinstance(a, dict) and a.get("name")}
        def _key(n):
            raw = params[n].get("_raw_op_param", n)
            return input_order.get(raw, input_order.get(n, 999))
        primaries_by_role.sort(key=_key)
        return primaries_by_role[0]

    # name heuristics
    primaries = [n for n in tensor_params
                 if classify_param_role(n) == "primary"]
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
        elif spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
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
# YAML skeleton builder — 3-layer fallback
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
    has_py_sig = bool(
        schema.get("python_signature")
        and (schema["python_signature"].get("parameters"))
    )
    is_raw_ops = resolve_info.get("is_raw_ops", False)

    # ── Layer selection ───────────────────────────────────────────
    params: Dict[str, Any] = {}

    if is_raw_ops and has_opdef:
        # Original raw_ops path
        params = _original_merge(schema)

    elif has_opdef and has_py_sig:
        # L1: Both Python signature and OpDef available
        params = merge_highlevel_with_opdef(schema)

    elif has_opdef and not has_py_sig:
        # L2: signature failed, but OpDef exists ← THE FIX
        print(f"  [!] {api_name}: python_signature unavailable, "
              f"using OpDef-direct fallback")
        params = _fallback_from_opdef(schema)

    else:
        # L3: No OpDef — composite or unknown
        params = build_params_without_opdef(
            schema, llm_client, doc_snippet
        )

    # ── Safety net ────────────────────────────────────────────────
    if not params and has_opdef:
        print(f"  [!!] {api_name}: all param paths failed, "
              f"using _original_merge as last resort")
        params = _original_merge(schema)

    # ── Primary param ─────────────────────────────────────────────
    primary_param = detect_primary_param(params, input_args, api_name)
    annotate_param_roles(params, primary_param)

    # ── Build YAML ────────────────────────────────────────────────
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

    # Clean up internal fields
    for _pname, spec in params.items():
        if isinstance(spec, dict):
            spec.pop("_raw_op_param", None)
            spec.pop("_shape_hint", None)

    return y


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified YAML skeleton builder for raw_ops + high-level TF APIs."
    )
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--out_dir", default="./tf_yaml_skeleton")
    ap.add_argument("--rank_index_dir", default=None)
    ap.add_argument("--doc_dir", default=None)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--llm_api_key", default=None)
    args = ap.parse_args()

    src = Path(args.schema_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
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
    stats = {"raw_ops": 0, "resolved_sig": 0, "resolved_opdef": 0, "unresolved": 0}

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
                doc_snippet = doc_path.read_text(encoding="utf-8", errors="ignore")[:3000]

        y = build_yaml_skeleton(
            schema, rank_hints, param_rank_details,
            llm_client=llm_client, doc_snippet=doc_snippet,
        )

        out_path = out_dir / f"{safe_name(api_name)}.yaml"
        out_path.write_text(
            yaml.safe_dump(y, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        pp = y.get("primary_param", "?")
        ri = y.get("resolve_info") or {}
        n_params = len(y.get("params") or {})

        if ri.get("is_raw_ops"):
            stats["raw_ops"] += 1
        elif bool((schema.get("python_signature") or {}).get("parameters")):
            stats["resolved_sig"] += 1
        elif ri.get("raw_op_name"):
            stats["resolved_opdef"] += 1
        else:
            stats["unresolved"] += 1

        print(f"[+] {api_name} → {out_path}  primary={pp}  params={n_params}")
        count += 1

    print(f"\n[done] wrote {count} yaml skeletons "
          f"(raw_ops={stats['raw_ops']}, "
          f"resolved_sig={stats['resolved_sig']}, "
          f"resolved_opdef={stats['resolved_opdef']}, "
          f"unresolved={stats['unresolved']})")


if __name__ == "__main__":
    main()
