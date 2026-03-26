#!/usr/bin/env python3
"""
tf_llm_patch_yaml.py  –  Stage C: LLM-driven FULL shape materialization
for TF raw_ops YAML.

==========================================================================
Changes vs. original
==========================================================================
1. PRE-PROCESSING: discretizes rank_any into concrete test_ranks before
   sending to LLM (using op_family rules + heuristics).
2. PRE-PROCESSING: identifies layout variants (NHWC/NCHW) and tells LLM
   exactly which (rank, layout) combinations to generate.
3. POST-PROCESSING: validates completeness — NO TODO_SHAPE may survive.
4. POST-PROCESSING: shrinks dtype space from full allowed list to test set.
5. Uses new prompt format (shape_spec_all_params instead of just primary).
6. Builds richer merged YAML with shape_spec_by_rank, test_ranks,
   test_dtype_choices, layout_variants.
"""
from __future__ import annotations

import os
import re
import argparse
import json
import traceback
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    from openai import OpenAI, BadRequestError
except ImportError:
    OpenAI = None  # type: ignore
    BadRequestError = Exception  # type: ignore

from tf_llm_prompts import TF_YAML_PATCH_SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════
# 1) helpers
# ════════════════════════════════════════════════════════════════

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def safe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2:]
    return head + "\n\n[...TRUNCATED...]\n\n" + tail


def load_yaml_obj(yaml_path: Path) -> Any:
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))


def dump_yaml_obj(obj: Any) -> str:
    return yaml.safe_dump(
        obj,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        default_flow_style=False,
    )


def safe_name(s: Any, max_len: int = 120) -> str:
    if s is None:
        return "null"
    s = str(s).strip()
    if not s:
        return "empty"
    s = s.replace("::", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "empty"
    return s[:max_len]


# ════════════════════════════════════════════════════════════════
# 2) PRE-PROCESSING: rank discretization
# ════════════════════════════════════════════════════════════════

# Default rank expansions for rank_any by op family
_RANK_ANY_DEFAULTS: Dict[str, List[int]] = {
    "bias_add":   [1, 2, 3, 4, 5],
    "reshape":    [1, 2, 3, 4],
    "transpose":  [2, 3, 4],
    "gather":     [1, 2, 3],
    "scatter":    [1, 2, 3],
    "reduce":     [1, 2, 3, 4],
    "concat":     [1, 2, 3, 4],
    "split":      [1, 2, 3, 4],
    "softmax":    [2, 3, 4],
    "one_hot":    [1, 2, 3],
}

# Default rank for fixed-rank families
_RANK_FIXED_DEFAULTS: Dict[str, List[int]] = {
    "conv2d":           [4],
    "conv3d":           [5],
    "conv1d":           [3],
    "depthwise_conv2d": [4],
    "pool2d":           [4],
    "pool3d":           [5],
    "matmul":           [2],
    "batch_norm":       [4],
}

# Generic fallback for unknown ops with rank_any
_GENERIC_RANK_ANY = [1, 2, 3, 4]


def discretize_ranks(yaml_obj: Dict[str, Any]) -> List[int]:
    """
    Convert rank_hints into a concrete list of integer ranks to test.

    Priority:
    1. rank_hints.rank_candidates (if assigned and concrete)
    2. op_family-based defaults
    3. Generic fallback
    """
    rank_hints = yaml_obj.get("rank_hints") or {}
    op_family = yaml_obj.get("op_family") or ""

    # Check for concrete rank candidates
    candidates = rank_hints.get("rank_candidates") or []
    concrete = [r for r in candidates if isinstance(r, int)]

    if concrete:
        return sorted(set(concrete))

    # rank_any = true → use family defaults or generic
    rank_any = rank_hints.get("rank_any", False)
    rank_min = rank_hints.get("rank_min")

    if op_family in _RANK_FIXED_DEFAULTS:
        return _RANK_FIXED_DEFAULTS[op_family]

    if rank_any or rank_hints.get("status") in ("missing", "unassigned"):
        base = _RANK_ANY_DEFAULTS.get(op_family, _GENERIC_RANK_ANY)
        if rank_min is not None:
            base = [r for r in base if r >= rank_min]
        if not base:
            base = [rank_min] if rank_min else _GENERIC_RANK_ANY
        return sorted(set(base))

    # Single known rank from rank_max
    rank_max = rank_hints.get("rank_max")
    if isinstance(rank_max, int):
        return [rank_max]

    return _GENERIC_RANK_ANY


def identify_layouts(yaml_obj: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Identify layout variants (e.g., NHWC/NCHW) and which ranks they apply to.

    Returns: { "NHWC": [4], "NCHW": [4] } or empty dict.
    """
    params = yaml_obj.get("params") or {}
    df_param = params.get("data_format")
    if not isinstance(df_param, dict):
        return {}
    if df_param.get("kind") != "enum":
        return {}

    values = df_param.get("values") or []
    if not values:
        return {}

    # Determine which ranks layout applies to
    op_family = yaml_obj.get("op_family") or ""
    layout_ranks: Dict[str, List[int]] = {}

    for v in values:
        if not isinstance(v, str):
            continue
        v_upper = v.upper()
        if v_upper in ("NHWC", "NCHW"):
            layout_ranks[v_upper] = [4]  # Standard 4-D layout
        elif v_upper in ("NDHWC", "NCDHW"):
            layout_ranks[v_upper] = [5]  # 5-D layout
        elif v_upper in ("NWC", "NCW"):
            layout_ranks[v_upper] = [3]  # 3-D layout

    return layout_ranks


# ════════════════════════════════════════════════════════════════
# 3) PRE-PROCESSING: dtype shrinking
# ════════════════════════════════════════════════════════════════

# Priority order for dtype selection
_DTYPE_PRIORITY = [
    "float32", "float64", "int32", "int64", "bool",
    "complex64", "complex128",
    "float16", "bfloat16",
    "uint8", "int16", "int8", "uint16", "uint32", "uint64",
]

# Quantized types to skip
_SKIP_DTYPES = {"qint8", "quint8", "qint32", "qint16", "quint16"}


def shrink_dtype_space(yaml_obj: Dict[str, Any], max_dtypes: int = 4) -> List[str]:
    """
    Select a representative test dtype set from the full allowed list.
    """
    # Find allowed types from attr_meta
    tf_block = yaml_obj.get("tf") or {}
    attr_meta = tf_block.get("attr_meta") or []

    allowed: Set[str] = set()
    for attr in attr_meta:
        if not isinstance(attr, dict):
            continue
        av = attr.get("allowed_values")
        if isinstance(av, dict) and av.get("kind") == "list":
            type_list = av.get("type") or []
            for t in type_list:
                if isinstance(t, str):
                    allowed.add(t)

    if not allowed:
        # Fallback: check params for dtype_choices
        params = yaml_obj.get("params") or {}
        for _pname, spec in params.items():
            if isinstance(spec, dict):
                dc = spec.get("dtype_choices") or []
                for t in dc:
                    if isinstance(t, str):
                        allowed.add(t)

    if not allowed:
        return ["float32", "float64"]

    # Filter out quantized types
    allowed = allowed - _SKIP_DTYPES

    # Select by priority
    selected: List[str] = []
    for dt in _DTYPE_PRIORITY:
        if dt in allowed and len(selected) < max_dtypes:
            selected.append(dt)

    if not selected:
        selected = [list(allowed)[0]]

    return selected


# ════════════════════════════════════════════════════════════════
# 4) PRE-PROCESSING: build rank plan for LLM
# ════════════════════════════════════════════════════════════════

def build_rank_plan(yaml_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a structured rank plan to include in the LLM prompt.
    This tells the LLM exactly what variants to produce.
    """
    test_ranks = discretize_ranks(yaml_obj)
    layouts = identify_layouts(yaml_obj)
    test_dtypes = shrink_dtype_space(yaml_obj)

    # Build variant plan: list of (rank, layout_or_None)
    variant_plan: List[Dict[str, Any]] = []

    # Collect which ranks have layout variants
    layout_rank_set: Set[int] = set()
    for _layout, ranks in layouts.items():
        layout_rank_set.update(ranks)

    for rank in test_ranks:
        if rank in layout_rank_set and layouts:
            # This rank has layout variants
            for layout_name, layout_ranks in layouts.items():
                if rank in layout_ranks:
                    variant_plan.append({
                        "rank": rank,
                        "layout": layout_name,
                    })
        else:
            # No layout for this rank
            variant_plan.append({
                "rank": rank,
                "layout": None,
            })

    # Identify all tensor params that need shape_spec
    tensor_params: List[Dict[str, str]] = []
    params = yaml_obj.get("params") or {}
    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("kind", "")
        if kind in ("tensor", "tensor_optional", "tensor_list"):
            tensor_params.append({
                "name": pname,
                "role": spec.get("role", "unknown"),
                "semantic_role": spec.get("semantic_role", "unknown"),
            })

    return {
        "test_ranks": test_ranks,
        "test_dtype_choices": test_dtypes,
        "layouts": layouts,
        "variant_plan": variant_plan,
        "tensor_params_needing_shape": tensor_params,
        "primary_param": yaml_obj.get("primary_param"),
        "op_family": yaml_obj.get("op_family"),
    }


# ════════════════════════════════════════════════════════════════
# 5) JSON patch parsing & normalization
# ════════════════════════════════════════════════════════════════

def extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    if text.startswith("{") and text.endswith("}"):
        return text
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return text[l: r + 1]
    raise ValueError("cannot find JSON object in model output")


def parse_patch(raw_text: str) -> Dict[str, Any]:
    js = extract_json_object(raw_text)
    return json.loads(js)


def normalize_multi_patch(patch_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rank_assignment"] = patch_dict.get("rank_assignment") or {}
    out["test_dtype_choices"] = patch_dict.get("test_dtype_choices") or []
    out["layout_variants"] = patch_dict.get("layout_variants") or {}
    out["variants"] = patch_dict.get("variants") or []
    out["shared_constraints"] = patch_dict.get("shared_constraints") or []
    out["changes"] = patch_dict.get("changes") or []
    out["warnings"] = patch_dict.get("warnings") or []

    if not isinstance(out["rank_assignment"], dict):
        out["rank_assignment"] = {}
    if not isinstance(out["test_dtype_choices"], list):
        out["test_dtype_choices"] = []
    if not isinstance(out["layout_variants"], dict):
        out["layout_variants"] = {}
    if not isinstance(out["variants"], list):
        out["variants"] = []
    if not isinstance(out["shared_constraints"], list):
        out["shared_constraints"] = []

    # Clean constraints
    def _clean_constraints(lst: Any) -> List[str]:
        cleaned: List[str] = []
        if not isinstance(lst, list):
            return cleaned
        for item in lst:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif isinstance(item, dict):
                for k in ("expr", "constraint"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        cleaned.append(v.strip())
                        break
        return cleaned

    out["shared_constraints"] = _clean_constraints(out["shared_constraints"])

    # Clean variants
    cleaned_variants: List[Dict[str, Any]] = []
    for v in out["variants"]:
        if not isinstance(v, dict):
            continue
        vv = dict(v)
        r = vv.get("rank", None)
        if not (r is None or isinstance(r, int)):
            try:
                vv["rank"] = int(r)
            except (ValueError, TypeError):
                vv["rank"] = None

        if not isinstance(vv.get("shape_vars"), dict):
            vv["shape_vars"] = {}
        if not isinstance(vv.get("shape_spec_all_params"), dict):
            # Backward compat: try shape_spec_fixes
            vv["shape_spec_all_params"] = vv.get("shape_spec_fixes") or {}
        if not isinstance(vv.get("shape_spec_all_params"), dict):
            vv["shape_spec_all_params"] = {}

        # Also accept primary_shape_spec and merge it
        pss = vv.get("primary_shape_spec")
        primary = (out.get("rank_assignment") or {}).get("primary_param")
        if isinstance(pss, list) and primary:
            if primary not in vv["shape_spec_all_params"]:
                vv["shape_spec_all_params"][primary] = pss

        vv["constraints"] = _clean_constraints(vv.get("constraints") or [])
        vv["layout"] = vv.get("layout", None)
        cleaned_variants.append(vv)

    out["variants"] = cleaned_variants
    return out


# ════════════════════════════════════════════════════════════════
# 6) validation
# ════════════════════════════════════════════════════════════════

_ALLOWED_BUILTINS = {
    "isinstance", "all", "any", "len", "tuple",
    "min", "max", "abs",
}


def validate_shape_vars(shape_vars: Dict[str, Any]) -> List[str]:
    errs = []
    for k, v in shape_vars.items():
        if not isinstance(k, str) or not k:
            errs.append(f"shape_vars key invalid: {k!r}")
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            errs.append(f"shape_vars[{k}] must be [lo,hi], got: {v!r}")
            continue
        lo, hi = v[0], v[1]
        if not (isinstance(lo, int) and isinstance(hi, int)):
            errs.append(f"shape_vars[{k}] lo/hi must be int, got: {v!r}")
            continue
        if lo < 1 or hi < 1 or lo > hi:
            errs.append(f"shape_vars[{k}] must satisfy 1<=lo<=hi, got: {v!r}")
    return errs


def validate_no_expressions(shape_spec: List[str]) -> Optional[str]:
    for item in shape_spec:
        if not isinstance(item, str):
            return f"shape_spec item must be str, got {item!r}"
        if any(ch in item for ch in ("+", "-", "*", "/", "%", "(", ")", " ", "[", "]", ".")):
            return f"shape_spec contains expression-like token: {item!r}"
    return None


def extract_names_ast(expr: str) -> Set[str]:
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return set()
    names: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
    return {x for x in names if x not in {"None", "True", "False"}}


def validate_multi_patch(
    patch: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> List[str]:
    """Validate the patch with completeness checks."""
    errs: List[str] = []

    params = base_yaml.get("params")
    if not isinstance(params, dict):
        errs.append("base YAML missing params dict")
        return errs

    # Identify all tensor params
    tensor_params = set()
    for pname, spec in params.items():
        if isinstance(spec, dict) and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
            tensor_params.add(pname)

    variants = patch.get("variants") or []
    if not isinstance(variants, list) or not variants:
        errs.append("variants must be a non-empty list")
        return errs

    rank_assignment = patch.get("rank_assignment") or {}
    primary_param = rank_assignment.get("primary_param") or base_yaml.get("primary_param")

    for i, v in enumerate(variants):
        if not isinstance(v, dict):
            errs.append(f"variants[{i}] must be object")
            continue

        rank = v.get("rank")
        if not (rank is None or isinstance(rank, int)):
            errs.append(f"variants[{i}].rank must be int or null")

        sv = v.get("shape_vars") or {}
        if not isinstance(sv, dict):
            errs.append(f"variants[{i}].shape_vars must be object")
            sv = {}
        errs += [f"variants[{i}].{e}" for e in validate_shape_vars(sv)]

        # COMPLETENESS CHECK: every tensor param must have a shape_spec
        all_params = v.get("shape_spec_all_params") or {}
        missing_params = tensor_params - set(all_params.keys())
        if missing_params:
            errs.append(
                f"variants[{i}] missing shape_spec for tensor params: {sorted(missing_params)}"
            )

        # Validate each shape_spec
        for pname, spec_list in all_params.items():
            if pname not in params:
                errs.append(f"variants[{i}].shape_spec_all_params has unknown param: {pname}")
                continue
            if not isinstance(spec_list, list) or not all(isinstance(x, str) for x in spec_list):
                errs.append(f"variants[{i}].shape_spec_all_params[{pname}] must be list[str]")
                continue

            err = validate_no_expressions(spec_list)
            if err:
                errs.append(f"variants[{i}].shape_spec_all_params[{pname}] invalid: {err}")
                continue

            # Check vars are defined
            missing_vars = [x for x in spec_list if x not in sv]
            if missing_vars:
                errs.append(
                    f"variants[{i}].shape_spec_all_params[{pname}] "
                    f"references vars not in shape_vars: {missing_vars}"
                )

            # Check primary param length matches rank
            if isinstance(rank, int) and primary_param and pname == primary_param:
                if len(spec_list) != rank:
                    errs.append(
                        f"variants[{i}] primary_param={primary_param} rank={rank} "
                        f"but shape_spec length={len(spec_list)}"
                    )

    return errs


# ════════════════════════════════════════════════════════════════
# 7) build merged YAML from patch
# ════════════════════════════════════════════════════════════════

def normalize_shape_vars_for_write(shape_vars: Dict[str, Any]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for k, v in shape_vars.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                out[str(k)] = [int(v[0]), int(v[1])]
            except (ValueError, TypeError):
                pass
    return out


def build_merged_yaml(
    base_yaml: Dict[str, Any],
    patch: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build the complete, harness-ready YAML from base + Stage C patch.

    Writes:
    - test_ranks: concrete list of ranks to test
    - test_dtype_choices: shrunk dtype list
    - layout_variants: layout info (if applicable)
    - shape_vars: union of all variant shape_vars
    - params[primary].shape_spec_by_rank: per-rank shape spec
    - params[primary].shape_spec_by_rank_and_layout: per-rank-and-layout (if applicable)
    - params[every_tensor].shape_spec: concrete (from first/min variant)
    - constraints: [] (Stage D fills these)
    """
    out = dict(base_yaml)
    params = out.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("base YAML missing params dict")

    rank_assignment = patch.get("rank_assignment") or {}
    primary_param = rank_assignment.get("primary_param") or out.get("primary_param")
    test_ranks = rank_assignment.get("test_ranks") or rank_plan.get("test_ranks") or []

    # Update top-level fields
    if primary_param:
        out["primary_param"] = primary_param
    out["test_ranks"] = sorted(set(int(r) for r in test_ranks if isinstance(r, int)))
    out["test_dtype_choices"] = (
        patch.get("test_dtype_choices")
        or rank_plan.get("test_dtype_choices")
        or ["float32", "float64"]
    )
    out["layout_variants"] = patch.get("layout_variants") or {}

    variants = patch.get("variants") or []

    # 1) Union shape_vars across all variants
    union_sv: Dict[str, List[int]] = {}
    for v in variants:
        if isinstance(v, dict) and isinstance(v.get("shape_vars"), dict):
            union_sv.update(normalize_shape_vars_for_write(v["shape_vars"]))

    old_sv = out.get("shape_vars")
    if not isinstance(old_sv, dict):
        old_sv = {}
    merged_sv = dict(old_sv)
    merged_sv.update(union_sv)
    out["shape_vars"] = merged_sv

    # 2) Build shape_spec_by_rank and shape_spec_by_rank_and_layout for primary
    shape_spec_by_rank: Dict[str, List[str]] = {}
    shape_spec_by_rank_and_layout: Dict[str, Dict[str, List[str]]] = {}

    # Also collect per-param shape_spec from variants (for aux params)
    # Use the first variant that provides a shape for each param
    aux_shape_map: Dict[str, List[str]] = {}

    for v in variants:
        if not isinstance(v, dict):
            continue
        r = v.get("rank")
        layout = v.get("layout")
        all_specs = v.get("shape_spec_all_params") or {}

        # Primary param
        if primary_param and primary_param in all_specs:
            spec = all_specs[primary_param]
            if isinstance(spec, list) and all(isinstance(x, str) for x in spec):
                if isinstance(r, int):
                    rank_key = str(r)
                    if layout:
                        if rank_key not in shape_spec_by_rank_and_layout:
                            shape_spec_by_rank_and_layout[rank_key] = {}
                        shape_spec_by_rank_and_layout[rank_key][str(layout)] = list(spec)
                        # Also store the default layout as the main shape_spec_by_rank
                        default_layout = (out.get("params", {}).get("data_format", {})
                                          .get("default", ""))
                        if str(layout) == default_layout and rank_key not in shape_spec_by_rank:
                            shape_spec_by_rank[rank_key] = list(spec)
                    else:
                        shape_spec_by_rank[rank_key] = list(spec)

        # Aux params
        for pname, spec in all_specs.items():
            if pname == primary_param:
                continue
            if pname not in aux_shape_map:
                if isinstance(spec, list) and all(isinstance(x, str) for x in spec):
                    aux_shape_map[pname] = list(spec)

    # 3) Write shape_spec_by_rank to primary param
    if primary_param and primary_param in params:
        p = params[primary_param]
        if isinstance(p, dict):
            if shape_spec_by_rank:
                p["shape_spec_by_rank"] = shape_spec_by_rank
            if shape_spec_by_rank_and_layout:
                p["shape_spec_by_rank_and_layout"] = shape_spec_by_rank_and_layout

            # Set fallback shape_spec to min rank (default layout)
            if shape_spec_by_rank:
                try:
                    min_rank_key = str(min(int(k) for k in shape_spec_by_rank))
                    p["shape_spec"] = list(shape_spec_by_rank[min_rank_key])
                except Exception:
                    pass
            elif shape_spec_by_rank_and_layout:
                # Use the first available
                for _rk, layout_map in sorted(shape_spec_by_rank_and_layout.items()):
                    for _layout, spec in layout_map.items():
                        p["shape_spec"] = list(spec)
                        break
                    break

    # 4) Write aux param shape_specs
    for pname, spec in aux_shape_map.items():
        if pname in params:
            p = params[pname]
            if isinstance(p, dict):
                # Only overwrite if current is TODO_SHAPE
                cur = p.get("shape_spec")
                if (isinstance(cur, list) and any(x == "TODO_SHAPE" for x in cur)) or not cur:
                    p["shape_spec"] = spec

    # 5) Ensure no TODO_SHAPE remains
    remaining_todo = []
    for pname, spec in params.items():
        if isinstance(spec, dict) and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
            ss = spec.get("shape_spec")
            if isinstance(ss, list) and any(x == "TODO_SHAPE" for x in ss):
                remaining_todo.append(pname)
            elif not ss:
                remaining_todo.append(pname)

    summary = {
        "primary_param": primary_param,
        "test_ranks": out.get("test_ranks"),
        "test_dtype_choices": out.get("test_dtype_choices"),
        "shape_spec_by_rank_keys": sorted(shape_spec_by_rank.keys(), key=lambda x: int(x)),
        "layout_variants_present": bool(shape_spec_by_rank_and_layout),
        "union_shape_vars": sorted(union_sv.keys()),
        "remaining_todo_shape": remaining_todo,
        "num_variants_received": len(variants),
    }
    return out, summary


# ════════════════════════════════════════════════════════════════
# 8) LLM call
# ════════════════════════════════════════════════════════════════

def call_llm_for_patch(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ════════════════════════════════════════════════════════════════
# 9) build allowed-names set
# ════════════════════════════════════════════════════════════════

def build_tf_allowed_names(yaml_obj: Dict[str, Any]) -> Set[str]:
    allowed = set(_ALLOWED_BUILTINS)
    params = yaml_obj.get("params")
    if isinstance(params, dict):
        allowed |= set(params.keys())
    shape_vars = yaml_obj.get("shape_vars")
    if isinstance(shape_vars, dict):
        allowed |= set(shape_vars.keys())
    return allowed


# ════════════════════════════════════════════════════════════════
# 10) main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage C: LLM full shape materialization for TF raw_ops YAML."
    )
    ap.add_argument("--doc_txt", required=True,
                    help="Path to documentation text file for the API")
    ap.add_argument("--yaml_in", required=True,
                    help="Path to input YAML skeleton (from Stage B)")
    ap.add_argument("--yaml_out_dir", required=True,
                    help="Output directory for patched YAML files")
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--fail_on_invalid", action="store_true")
    args = ap.parse_args()

    doc_path = Path(args.doc_txt).resolve()
    yaml_in_path = Path(args.yaml_in).resolve()
    out_dir = Path(args.yaml_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_text = safe_truncate(read_text(doc_path), args.max_doc_chars)
    yaml_obj = load_yaml_obj(yaml_in_path)
    yaml_text = read_text(yaml_in_path)

    if not isinstance(yaml_obj, dict):
        raise RuntimeError(f"Input YAML must be a mapping/dict: {yaml_in_path}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # ── PRE-PROCESSING ───────────────────────────────────────────
    rank_plan = build_rank_plan(yaml_obj)
    print(f"[i] Rank plan: test_ranks={rank_plan['test_ranks']}, "
          f"layouts={rank_plan['layouts']}, "
          f"dtype_choices={rank_plan['test_dtype_choices']}, "
          f"variants_to_generate={len(rank_plan['variant_plan'])}")

    # ── BUILD PROMPT ─────────────────────────────────────────────
    system_prompt = TF_YAML_PATCH_SYSTEM_PROMPT

    rank_plan_text = json.dumps(rank_plan, indent=2, ensure_ascii=False)

    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== YAML SKELETON (INPUT) ===\n"
        f"{yaml_text}\n\n"
        "=== RANK PLAN (pre-computed, follow this) ===\n"
        f"{rank_plan_text}\n\n"
        "Return ONLY JSON.\n"
        "CRITICAL: Generate one variant per entry in variant_plan.\n"
        "CRITICAL: Every tensor param must have shape_spec in shape_spec_all_params.\n"
        "CRITICAL: No TODO_SHAPE may remain.\n"
        "CRITICAL: constraints and shared_constraints must be empty [].\n"
    )

    # ── LLM LOOP ─────────────────────────────────────────────────
    last_errors: List[str] = []
    last_raw: str = ""
    final_patch: Optional[Dict[str, Any]] = None
    user_prompt = base_user_prompt

    for attempt in range(args.max_retries + 1):
        try:
            last_raw = call_llm_for_patch(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except BadRequestError:
            print("=== BadRequestError ===")
            traceback.print_exc()
            raise

        # Parse
        try:
            patch_dict = parse_patch(last_raw)
        except Exception as e:
            last_errors = [f"JSON parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = (
                    base_user_prompt
                    + "\n\nYour previous output was NOT valid JSON.\n"
                      "Please output ONLY a valid JSON object.\n"
                )
                continue
            # Final fallback
            final_patch = _build_fallback_patch(yaml_obj, rank_plan)
            break

        # Normalize
        patch = normalize_multi_patch(patch_dict)

        # Inject test_ranks from rank_plan if LLM didn't provide them
        ra = patch.get("rank_assignment") or {}
        if not ra.get("test_ranks"):
            ra["test_ranks"] = rank_plan["test_ranks"]
            patch["rank_assignment"] = ra
        if not patch.get("test_dtype_choices"):
            patch["test_dtype_choices"] = rank_plan["test_dtype_choices"]

        # Validate
        last_errors = validate_multi_patch(patch, yaml_obj, rank_plan)

        if not last_errors:
            final_patch = patch
            break

        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\n\nYour previous JSON FAILED validation.\n"
                  "Fix it and output ONLY JSON again.\n"
                  "Validation errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n\nPrevious JSON (for reference):\n"
                + (extract_json_object(last_raw) if last_raw else "")
                + "\n"
            )
        else:
            final_patch = patch  # best-effort

    if final_patch is None:
        final_patch = _build_fallback_patch(yaml_obj, rank_plan)

    # ── BUILD MERGED YAML ────────────────────────────────────────
    api_name = yaml_obj.get("api_name", "unknown_api")
    tf_block = yaml_obj.get("tf") or {}
    op_name = tf_block.get("op_name") or api_name.split(".")[-1]
    primary_param = (
        (final_patch.get("rank_assignment") or {}).get("primary_param")
        or yaml_obj.get("primary_param")
        or "unknown"
    )

    merged_yaml, summary = build_merged_yaml(
        base_yaml=yaml_obj,
        patch=final_patch,
        rank_plan=rank_plan,
    )

    out_name = f"{safe_name(api_name)}.yaml"
    out_path = out_dir / out_name
    out_path.write_text(dump_yaml_obj(merged_yaml), encoding="utf-8")

    meta = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(out_path),
        "rank_plan": rank_plan,
        "rank_assignment": final_patch.get("rank_assignment", {}),
        "num_variants": len(final_patch.get("variants") or []),
        "summary": summary,
        "warnings": final_patch.get("warnings", []),
        "validation_errors": last_errors,
        "raw_model_output_snippet": (last_raw[:8000] if last_raw else ""),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[+] wrote Stage-C yaml: {out_path}")
    if summary.get("remaining_todo_shape"):
        print(f"[!] WARNING: TODO_SHAPE still remains in: {summary['remaining_todo_shape']}")
    if last_errors:
        msg = "[!] Stage-C validation errors (not fully fixed):\n" + "\n".join(f"   - {e}" for e in last_errors)
        if args.fail_on_invalid:
            raise SystemExit(msg)
        else:
            print(msg)


# ════════════════════════════════════════════════════════════════
# 11) fallback patch builder (when LLM completely fails)
# ════════════════════════════════════════════════════════════════

def _build_fallback_patch(
    yaml_obj: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a minimal fallback patch using generic dimensions.
    Used when LLM fails completely.
    """
    primary_param = yaml_obj.get("primary_param")
    test_ranks = rank_plan.get("test_ranks") or [2, 3, 4]
    test_dtypes = rank_plan.get("test_dtype_choices") or ["float32", "float64"]

    params = yaml_obj.get("params") or {}
    tensor_params = [
        pname for pname, spec in params.items()
        if isinstance(spec, dict) and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list")
    ]

    dim_names = ["N", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

    variants = []
    for rank in test_ranks:
        sv = {}
        for i in range(rank):
            dname = dim_names[i] if i < len(dim_names) else f"D{i}"
            sv[dname] = [1, 16]

        primary_spec = [dim_names[i] if i < len(dim_names) else f"D{i}" for i in range(rank)]

        all_specs: Dict[str, List[str]] = {}
        for pname in tensor_params:
            if pname == primary_param:
                all_specs[pname] = primary_spec
            else:
                # Generic: 1-D with last dim of primary
                last_dim = primary_spec[-1] if primary_spec else "D1"
                all_specs[pname] = [last_dim]

        variants.append({
            "rank": rank,
            "layout": None,
            "shape_vars": sv,
            "shape_spec_all_params": all_specs,
            "constraints": [],
        })

    return {
        "rank_assignment": {
            "primary_param": primary_param,
            "test_ranks": test_ranks,
            "confidence": 0.1,
            "notes": ["fallback: LLM failed, using generic dimensions"],
        },
        "test_dtype_choices": test_dtypes,
        "layout_variants": {},
        "variants": variants,
        "shared_constraints": [],
        "changes": [],
        "warnings": ["LLM failed completely, using generic fallback shapes"],
    }


if __name__ == "__main__":
    main()
