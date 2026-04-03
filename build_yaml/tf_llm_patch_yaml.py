#!/usr/bin/env python3
"""
tf_llm_patch_yaml.py  –  Stage C: LLM-assisted YAML completion
for TF raw_ops / high-level TF API YAML.

DESIGN
==========================================================================
LLM is treated as a "completer", NOT the final schema authority.

Pipeline:
1. Read original YAML + official doc text.
2. Pre-compute concrete rank plan / dtype plan / layout plan.
3. Ask LLM to OUTPUT A FULL YAML FILE (not JSON patch).
4. Parse LLM YAML output.
5. RULE-BASED EXTRACTION:
   - extract ONLY whitelisted fields
   - ignore everything else
6. RULE-BASED NORMALIZATION:
   - normalize shape_spec / shape_spec_by_rank / shape_vars
   - repair partial / legacy / variant-style outputs when possible
7. VALIDATE
8. MERGE back into original YAML

Major salvage modes supported:
A) Standard list[str]
   shape_spec: [N, H, W, C]

B) Ranked object list
   shape_spec:
     - rank: 1
       dims: [P0]
     - rank: 2
       dims: [P0, P1]

C) Ellipsis rank-any
   shape_spec: [R0, R1, ...]
"""

from __future__ import annotations

import os
import re
import argparse
import json
import traceback
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


def _plain_var_name(s: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s))


def _sanitize_prefix(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "P"
    if s[0].isdigit():
        s = "P_" + s
    return s.upper()


def _is_tensor_param(spec: Any) -> bool:
    return isinstance(spec, dict) and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list")


def _coerce_int(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and int(x) == x:
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return None
    return None


def _default_var_range(var_name: str) -> List[int]:
    u = var_name.upper()
    if u in {"N", "BATCH", "B"} or u.startswith("N_"):
        return [1, 8]
    if u in {"C", "C_IN", "C_OUT", "CIN", "COUT"}:
        return [1, 64]
    if u in {"H", "W", "D", "L", "T"}:
        return [1, 32]
    if u.startswith(("H", "W", "D", "L", "T", "DIM")):
        return [1, 32]
    if u in {"KH", "KW", "KD", "K_H", "K_W", "K_D"} or u.startswith("K"):
        return [1, 11]
    if u in {"I", "J", "IDX", "AXIS_LEN", "R"} or u.startswith(("I", "IDX", "R")):
        return [1, 16]
    return [1, 16]


def _merge_shape_var(shape_vars: Dict[str, List[int]], name: str, lo: int, hi: int) -> None:
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    if name in shape_vars:
        old_lo, old_hi = shape_vars[name]
        shape_vars[name] = [min(old_lo, lo), max(old_hi, hi)]
    else:
        shape_vars[name] = [lo, hi]


def _ensure_shape_var(shape_vars: Dict[str, List[int]], name: str, rng: Optional[List[int]] = None) -> None:
    if rng is None:
        rng = _default_var_range(name)
    _merge_shape_var(shape_vars, name, int(rng[0]), int(rng[1]))


def _generated_var_name(
    pname: str,
    semantic_role: str,
    idx: int,
    is_primary: bool,
) -> str:
    if is_primary:
        return f"DIM{idx - 1}"
    if semantic_role == "index_input":
        return "I" if idx == 1 else f"I{idx - 1}"
    if semantic_role == "shape_control":
        return "R" if idx == 1 else f"R{idx - 1}"
    prefix = _sanitize_prefix(pname)
    return f"{prefix}_{idx - 1}"


# ════════════════════════════════════════════════════════════════
# 2) PRE-PROCESSING: rank discretization
# ════════════════════════════════════════════════════════════════

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

_GENERIC_RANK_ANY = [1, 2, 3, 4]


def discretize_ranks(yaml_obj: Dict[str, Any]) -> List[int]:
    rank_hints = yaml_obj.get("rank_hints") or {}
    op_family = yaml_obj.get("op_family") or ""

    candidates = rank_hints.get("rank_candidates") or []
    concrete = [r for r in candidates if isinstance(r, int)]
    if concrete:
        return sorted(set(concrete))

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

    rank_max = rank_hints.get("rank_max")
    if isinstance(rank_max, int):
        return [rank_max]

    return _GENERIC_RANK_ANY


def identify_layouts(yaml_obj: Dict[str, Any]) -> Dict[str, List[int]]:
    params = yaml_obj.get("params") or {}
    df_param = params.get("data_format")
    if not isinstance(df_param, dict):
        return {}
    if df_param.get("kind") != "enum":
        return {}

    values = df_param.get("values") or []
    if not values:
        return {}

    layout_ranks: Dict[str, List[int]] = {}
    for v in values:
        if not isinstance(v, str):
            continue
        v_upper = v.upper()
        if v_upper in ("NHWC", "NCHW"):
            layout_ranks[v_upper] = [4]
        elif v_upper in ("NDHWC", "NCDHW"):
            layout_ranks[v_upper] = [5]
        elif v_upper in ("NWC", "NCW"):
            layout_ranks[v_upper] = [3]

    return layout_ranks


# ════════════════════════════════════════════════════════════════
# 3) PRE-PROCESSING: dtype shrinking
# ════════════════════════════════════════════════════════════════

_DTYPE_PRIORITY = [
    "float32", "float64", "int32", "int64", "bool",
    "complex64", "complex128",
    "float16", "bfloat16",
    "uint8", "int16", "int8", "uint16", "uint32", "uint64",
]

_SKIP_DTYPES = {"qint8", "quint8", "qint32", "qint16", "quint16"}


def _collect_allowed_dtypes(yaml_obj: Dict[str, Any]) -> Set[str]:
    tf_block = yaml_obj.get("tf") or {}
    attr_meta = tf_block.get("attr_meta") or []

    allowed: Set[str] = set()
    for attr in attr_meta:
        if not isinstance(attr, dict):
            continue
        av = attr.get("allowed_values")
        if isinstance(av, dict) and av.get("kind") == "list":
            for t in av.get("type") or []:
                if isinstance(t, str):
                    allowed.add(t)

    if not allowed:
        params = yaml_obj.get("params") or {}
        for _pname, spec in params.items():
            if isinstance(spec, dict):
                for t in spec.get("dtype_choices") or []:
                    if isinstance(t, str):
                        allowed.add(t)

    return allowed - _SKIP_DTYPES


def shrink_dtype_space(yaml_obj: Dict[str, Any], max_dtypes: int = 4) -> List[str]:
    allowed = _collect_allowed_dtypes(yaml_obj)
    if not allowed:
        return ["float32", "float64"]

    selected: List[str] = []
    for dt in _DTYPE_PRIORITY:
        if dt in allowed and len(selected) < max_dtypes:
            selected.append(dt)

    if not selected:
        selected = [sorted(allowed)[0]]
    return selected


# ════════════════════════════════════════════════════════════════
# 4) PRE-PROCESSING: build rank plan for LLM
# ════════════════════════════════════════════════════════════════

def build_rank_plan(yaml_obj: Dict[str, Any]) -> Dict[str, Any]:
    test_ranks = discretize_ranks(yaml_obj)
    layouts = identify_layouts(yaml_obj)
    test_dtypes = shrink_dtype_space(yaml_obj)

    variant_plan: List[Dict[str, Any]] = []
    layout_rank_set: Set[int] = set()
    for _layout, ranks in layouts.items():
        layout_rank_set.update(ranks)

    for rank in test_ranks:
        if rank in layout_rank_set and layouts:
            for layout_name, layout_ranks in layouts.items():
                if rank in layout_ranks:
                    variant_plan.append({"rank": rank, "layout": layout_name})
        else:
            variant_plan.append({"rank": rank, "layout": None})

    tensor_params: List[Dict[str, str]] = []
    params = yaml_obj.get("params") or {}
    for pname, spec in params.items():
        if _is_tensor_param(spec):
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
# 5) LLM OUTPUT PARSING: full YAML first, tolerate extras
# ════════════════════════════════════════════════════════════════

def extract_yaml_or_json_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")

    m = re.search(r"```(?:yaml|yml|json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()

    if text.startswith("{") or text.startswith("["):
        return text
    if ":" in text and "\n" in text:
        return text

    lcurly = text.find("{")
    rcurly = text.rfind("}")
    if lcurly != -1 and rcurly != -1 and rcurly > lcurly:
        return text[lcurly:rcurly + 1]

    raise ValueError("cannot locate YAML/JSON content in model output")

import re

def strip_non_yaml_preamble(text: str) -> str:
    text = text or ""

    # 去掉 <think>...</think>
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)

    # 去掉前面的解释性文字，只保留从第一个顶层 YAML key 开始
    candidate_keys = [
        "generator:",
        "api_name:",
        "category:",
        "api_category:",
        "api_module:",
        "primary_param:",
        "rank_hints:",
        "tf:",
        "resolve_info:",
        "shape_vars:",
        "params:",
        "constraints:",
        "op_family:",
        "test_ranks:",
        "test_dtype_choices:",
        "layout_variants:",
    ]

    best_pos = None
    for key in candidate_keys:
        m = re.search(rf"(?m)^{re.escape(key)}", text)
        if m:
            if best_pos is None or m.start() < best_pos:
                best_pos = m.start()

    if best_pos is not None:
        text = text[best_pos:]

    return text.strip()


def parse_llm_completed_yaml(raw_text: str) -> Dict[str, Any]:
    block = extract_yaml_or_json_block(raw_text)
    block = strip_non_yaml_preamble(block)

    obj = yaml.safe_load(block)

    if isinstance(obj, str):
        obj2 = yaml.safe_load(obj)
        if isinstance(obj2, dict):
            return obj2

    if not isinstance(obj, dict):
        raise ValueError(f"LLM output is not a YAML mapping/dict: {type(obj).__name__}")
    return obj


# ════════════════════════════════════════════════════════════════
# 6) RULE-BASED EXTRACTION / NORMALIZATION
# ════════════════════════════════════════════════════════════════

def _new_internal_completion(base_yaml: Dict[str, Any], rank_plan: Dict[str, Any]) -> Dict[str, Any]:
    params = base_yaml.get("params") or {}
    out_params: Dict[str, Dict[str, Any]] = {}
    for pname, spec in params.items():
        if _is_tensor_param(spec):
            out_params[pname] = {}

    layout_variants: Dict[str, Any] = {}
    for layout_name, ranks in (rank_plan.get("layouts") or {}).items():
        layout_variants[layout_name] = {
            "applies_to_ranks": list(ranks),
            "notes": "",
        }

    return {
        "primary_param": base_yaml.get("primary_param"),
        "test_ranks": list(rank_plan.get("test_ranks") or []),
        "test_dtype_choices": list(rank_plan.get("test_dtype_choices") or []),
        "layout_variants": layout_variants,
        "shape_vars": {},
        "params": out_params,
        "warnings": [],
        "changes": [],
        "source_modes_used": [],
    }


def _merge_warning(comp: Dict[str, Any], msg: str) -> None:
    warnings = comp.setdefault("warnings", [])
    if msg not in warnings:
        warnings.append(msg)


def _merge_change(comp: Dict[str, Any], msg: str) -> None:
    changes = comp.setdefault("changes", [])
    if msg not in changes:
        changes.append(msg)


def _record_source_mode(comp: Dict[str, Any], mode: str) -> None:
    modes = comp.setdefault("source_modes_used", [])
    if mode not in modes:
        modes.append(mode)


def normalize_shape_vars_from_llm(raw_shape_vars: Any, out_shape_vars: Dict[str, List[int]]) -> int:
    count = 0
    if not isinstance(raw_shape_vars, dict):
        return count
    for k, v in raw_shape_vars.items():
        if not isinstance(k, str) or not _plain_var_name(k):
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo = _coerce_int(v[0])
            hi = _coerce_int(v[1])
            if lo is None or hi is None:
                continue
            if lo < 1:
                lo = 1
            if hi < lo:
                hi = lo
            _merge_shape_var(out_shape_vars, k, lo, hi)
            count += 1
    return count


def canonicalize_shape_token(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if s in {"...", "…"}:
        return "__ELLIPSIS__"

    # 1) plain token only
    if _plain_var_name(s):
        return s

    # 2) token followed only by parenthetical comment
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*\([^)]*\))?$", s)
    if m:
        tok = m.group(1)
        if _plain_var_name(tok):
            return tok

    return None


def _coerce_shape_list(raw_spec: Any) -> Optional[List[Any]]:
    if raw_spec is None:
        return None
    if isinstance(raw_spec, list):
        return raw_spec
    if isinstance(raw_spec, tuple):
        return list(raw_spec)
    if isinstance(raw_spec, str):
        s = raw_spec.strip()
        if not s:
            return None
        if s == "[]":
            return []
        try:
            parsed = yaml.safe_load(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, (str, int, float)):
                return [parsed]
        except Exception:
            pass
        if "," in s:
            parts = [x.strip() for x in s.strip("[]()").split(",") if x.strip()]
            return parts
        return [s]
    if isinstance(raw_spec, (int, float)):
        return [raw_spec]
    return None


def _normalize_shape_items_no_commit(
    vals: List[Any],
    pname: str,
    param_meta: Dict[str, Any],
) -> Optional[Tuple[List[str], Dict[str, List[int]]]]:
    is_primary = pname == param_meta.get("__primary_param__")
    sem_role = param_meta.get("semantic_role", "")

    if vals == []:
        return [], {}

    out: List[str] = []
    pending_defs: Dict[str, List[int]] = {}

    for i, item in enumerate(vals, start=1):
        if isinstance(item, str):
            tok = canonicalize_shape_token(item)
            if tok is None:
                iv = _coerce_int(item)
                if iv is None or iv < 1:
                    return None
                vname = _generated_var_name(pname, sem_role, i, is_primary)
                pending_defs[vname] = [iv, iv]
                out.append(vname)
                continue

            if tok == "__ELLIPSIS__":
                return None

            pending_defs.setdefault(tok, _default_var_range(tok))
            out.append(tok)
            continue

        iv = _coerce_int(item)
        if iv is not None and iv >= 1:
            vname = _generated_var_name(pname, sem_role, i, is_primary)
            pending_defs[vname] = [iv, iv]
            out.append(vname)
            continue

        return None

    return out, pending_defs


def normalize_shape_spec_value(
    raw_spec: Any,
    pname: str,
    param_meta: Dict[str, Any],
    out_shape_vars: Dict[str, List[int]],
) -> Optional[List[str]]:
    vals = _coerce_shape_list(raw_spec)
    if vals is None:
        return None

    normalized = _normalize_shape_items_no_commit(vals, pname, param_meta)
    if normalized is None:
        return None

    out, pending_defs = normalized
    for k, rng in pending_defs.items():
        _ensure_shape_var(out_shape_vars, k, rng)
    return out


def normalize_shape_spec_by_rank(
    raw_sbr: Any,
    pname: str,
    param_meta: Dict[str, Any],
    out_shape_vars: Dict[str, List[int]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(raw_sbr, dict):
        return out

    for rk, spec in raw_sbr.items():
        rki = _coerce_int(rk)
        if rki is None or rki < 0:
            continue
        ns = normalize_shape_spec_value(spec, pname, param_meta, out_shape_vars)
        if ns is not None:
            out[str(rki)] = ns
    return out


def normalize_shape_spec_by_rank_and_layout(
    raw_sbrl: Any,
    pname: str,
    param_meta: Dict[str, Any],
    out_shape_vars: Dict[str, List[int]],
) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    if not isinstance(raw_sbrl, dict):
        return out

    for rk, layout_map in raw_sbrl.items():
        rki = _coerce_int(rk)
        if rki is None or rki < 0 or not isinstance(layout_map, dict):
            continue
        rank_key = str(rki)
        out[rank_key] = {}
        for layout_name, spec in layout_map.items():
            if not isinstance(layout_name, str) or not layout_name.strip():
                continue
            ns = normalize_shape_spec_value(spec, pname, param_meta, out_shape_vars)
            if ns is not None:
                out[rank_key][layout_name] = ns
        if not out[rank_key]:
            del out[rank_key]
    return out


def normalize_ranked_shape_entries(
    raw_spec: Any,
    pname: str,
    param_meta: Dict[str, Any],
    out_shape_vars: Dict[str, List[int]],
) -> Dict[str, List[str]]:
    """
    Support:
      shape_spec:
        - rank: 1
          dims: [P0]
        - rank: 2
          dims: [P0, P1]
    """
    out: Dict[str, List[str]] = {}
    if not (isinstance(raw_spec, list) and raw_spec and all(isinstance(x, dict) for x in raw_spec)):
        return out

    for item in raw_spec:
        rk = _coerce_int(item.get("rank"))
        dims = item.get("dims")
        if dims is None:
            dims = item.get("shape")
        if dims is None:
            dims = item.get("shape_spec")
        if rk is None:
            continue
        ns = normalize_shape_spec_value(dims, pname, param_meta, out_shape_vars)
        if ns is not None:
            out[str(rk)] = ns

    return out


def expand_ellipsis_shape_spec(
    raw_spec: Any,
    pname: str,
    param_meta: Dict[str, Any],
    out_shape_vars: Dict[str, List[int]],
    test_ranks: List[int],
) -> Dict[str, List[str]]:
    """
    Support:
      shape_spec: [R0, R1, ...]
    Expands by test_ranks:
      rank 1 -> [R0]
      rank 2 -> [R0, R1]
      rank 3 -> [R0, R1, R2]
    """
    out: Dict[str, List[str]] = {}
    vals = _coerce_shape_list(raw_spec)
    if not isinstance(vals, list) or not vals:
        return out

    canonical: List[str] = []
    for item in vals:
        if isinstance(item, str):
            tok = canonicalize_shape_token(item)
            if tok is None:
                return out
            canonical.append(tok)
        else:
            return out

    if "__ELLIPSIS__" not in canonical:
        return out

    ell_idx = canonical.index("__ELLIPSIS__")
    prefix = canonical[:ell_idx]
    suffix = canonical[ell_idx + 1:]
    if suffix:
        return out
    if not prefix:
        return out

    prefix_norm = _normalize_shape_items_no_commit(prefix, pname, param_meta)
    if prefix_norm is None:
        return out

    prefix_tokens, prefix_defs = prefix_norm

    numbered = [re.match(r"^(.*?)(\d+)$", t) for t in prefix_tokens]
    if all(numbered):
        stems = {m.group(1) for m in numbered if m is not None}
        nums = [int(m.group(2)) for m in numbered if m is not None]

        if len(stems) == 1 and nums == list(range(nums[0], nums[0] + len(nums))):
            stem = next(iter(stems))
            start = nums[0]

            all_pending = dict(prefix_defs)
            for r in sorted(set(test_ranks)):
                if r < 0:
                    continue
                spec = []
                for j in range(r):
                    tok = f"{stem}{start + j}"
                    spec.append(tok)
                    all_pending.setdefault(tok, _default_var_range(tok))
                out[str(r)] = spec

            for k, rng in all_pending.items():
                _ensure_shape_var(out_shape_vars, k, rng)
            return out

    # Conservative fallback:
    # If all desired ranks are within the prefix length, use slices.
    max_rank = max(test_ranks) if test_ranks else 0
    if max_rank <= len(prefix_tokens):
        for k, rng in prefix_defs.items():
            _ensure_shape_var(out_shape_vars, k, rng)
        for r in sorted(set(test_ranks)):
            out[str(r)] = list(prefix_tokens[:r])
        return out

    return {}


def normalize_layout_variants(raw_layouts: Any, rank_plan: Dict[str, Any]) -> Dict[str, Any]:
    plan_layouts = rank_plan.get("layouts") or {}
    out: Dict[str, Any] = {}

    if isinstance(raw_layouts, dict):
        for lname, linfo in raw_layouts.items():
            if not isinstance(lname, str):
                continue
            if isinstance(linfo, dict):
                applies = linfo.get("applies_to_ranks")
                if not isinstance(applies, list):
                    applies = plan_layouts.get(lname, [])
                applies_int = sorted(set(r for r in (_coerce_int(x) for x in applies) if isinstance(r, int)))
                out[lname] = {
                    "applies_to_ranks": applies_int,
                    "notes": linfo.get("notes", "") if isinstance(linfo.get("notes"), str) else "",
                }
            elif isinstance(linfo, list):
                applies_int = sorted(set(r for r in (_coerce_int(x) for x in linfo) if isinstance(r, int)))
                out[lname] = {"applies_to_ranks": applies_int, "notes": ""}
    else:
        for lname, ranks in plan_layouts.items():
            out[lname] = {"applies_to_ranks": list(ranks), "notes": ""}

    if not out and plan_layouts:
        for lname, ranks in plan_layouts.items():
            out[lname] = {"applies_to_ranks": list(ranks), "notes": ""}
    return out


def choose_test_dtypes(raw_tdc: Any, base_yaml: Dict[str, Any], rank_plan: Dict[str, Any]) -> List[str]:
    allowed = _collect_allowed_dtypes(base_yaml)
    fallback = list(rank_plan.get("test_dtype_choices") or ["float32", "float64"])

    if not isinstance(raw_tdc, list):
        return fallback

    picked: List[str] = []
    for x in raw_tdc:
        if isinstance(x, str) and x in allowed and x not in picked:
            picked.append(x)

    return picked[:4] if picked else fallback


def choose_test_ranks(raw_ranks: Any, rank_plan: Dict[str, Any]) -> List[int]:
    fallback = list(rank_plan.get("test_ranks") or [1, 2, 3, 4])
    if not isinstance(raw_ranks, list):
        return fallback
    out = sorted(set(r for r in (_coerce_int(x) for x in raw_ranks) if isinstance(r, int) and r >= 0))
    return out or fallback


def _merge_param_completion(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key in ("shape_spec", "shape_spec_by_rank", "shape_spec_by_rank_and_layout"):
        if key not in src:
            continue
        if key == "shape_spec":
            if "shape_spec" not in dst and src["shape_spec"] is not None:
                dst["shape_spec"] = src["shape_spec"]
        elif key == "shape_spec_by_rank":
            s = dst.setdefault("shape_spec_by_rank", {})
            for rk, spec in (src.get("shape_spec_by_rank") or {}).items():
                if rk not in s:
                    s[rk] = spec
        elif key == "shape_spec_by_rank_and_layout":
            s = dst.setdefault("shape_spec_by_rank_and_layout", {})
            for rk, layout_map in (src.get("shape_spec_by_rank_and_layout") or {}).items():
                s.setdefault(rk, {})
                for layout_name, spec in layout_map.items():
                    if layout_name not in s[rk]:
                        s[rk][layout_name] = spec


def extract_from_yaml_params_section(
    llm_obj: Dict[str, Any],
    base_yaml: Dict[str, Any],
    completion: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> None:
    llm_params = llm_obj.get("params")
    if not isinstance(llm_params, dict):
        return

    base_params = base_yaml.get("params") or {}
    primary_param = base_yaml.get("primary_param")
    test_ranks = completion.get("test_ranks") or rank_plan.get("test_ranks") or []
    out_shape_vars = completion["shape_vars"]

    for pname, base_spec in base_params.items():
        if not _is_tensor_param(base_spec):
            continue

        llm_pspec = llm_params.get(pname)
        if not isinstance(llm_pspec, dict):
            continue

        param_meta = dict(base_spec)
        param_meta["__primary_param__"] = primary_param

        extracted: Dict[str, Any] = {}

        # 1) direct shape_spec
        ns = normalize_shape_spec_value(llm_pspec.get("shape_spec"), pname, param_meta, out_shape_vars)
        if ns is not None:
            extracted["shape_spec"] = ns

        # 2) direct shape_spec_by_rank
        sbr = normalize_shape_spec_by_rank(llm_pspec.get("shape_spec_by_rank"), pname, param_meta, out_shape_vars)
        if sbr:
            extracted["shape_spec_by_rank"] = sbr

        # 3) direct shape_spec_by_rank_and_layout
        sbrl = normalize_shape_spec_by_rank_and_layout(
            llm_pspec.get("shape_spec_by_rank_and_layout"),
            pname,
            param_meta,
            out_shape_vars,
        )
        if sbrl:
            extracted["shape_spec_by_rank_and_layout"] = sbrl

        # 4) ranked-entry shape_spec list
        if not extracted.get("shape_spec_by_rank"):
            ranked_sbr = normalize_ranked_shape_entries(
                llm_pspec.get("shape_spec"),
                pname,
                param_meta,
                out_shape_vars,
            )
            if ranked_sbr:
                extracted["shape_spec_by_rank"] = ranked_sbr
                # for scalar aux params: rank 0 -> []
                if pname != primary_param:
                    if "0" in ranked_sbr and "shape_spec" not in extracted:
                        extracted["shape_spec"] = ranked_sbr["0"]
                    else:
                        min_key = str(min(int(k) for k in ranked_sbr.keys()))
                        extracted["shape_spec"] = ranked_sbr[min_key]

        # 5) ellipsis rank-any shape_spec
        if not extracted.get("shape_spec_by_rank"):
            ellipsis_sbr = expand_ellipsis_shape_spec(
                llm_pspec.get("shape_spec"),
                pname,
                param_meta,
                out_shape_vars,
                test_ranks,
            )
            if ellipsis_sbr:
                extracted["shape_spec_by_rank"] = ellipsis_sbr
                if pname != primary_param:
                    min_key = str(min(int(k) for k in ellipsis_sbr.keys()))
                    extracted["shape_spec"] = ellipsis_sbr[min_key]

        # if we got shape_spec_by_rank for primary, choose min-rank default shape_spec
        if pname == primary_param and extracted.get("shape_spec_by_rank"):
            min_key = str(min(int(k) for k in extracted["shape_spec_by_rank"].keys()))
            extracted["shape_spec"] = extracted["shape_spec_by_rank"][min_key]

        if extracted:
            _merge_param_completion(completion["params"][pname], extracted)

    _record_source_mode(completion, "params_section")


def extract_from_variant_style(
    llm_obj: Dict[str, Any],
    base_yaml: Dict[str, Any],
    completion: Dict[str, Any],
) -> None:
    variants = llm_obj.get("variants")
    if not isinstance(variants, list):
        return

    base_params = base_yaml.get("params") or {}
    primary_param = base_yaml.get("primary_param")
    out_shape_vars = completion["shape_vars"]

    for v in variants:
        if not isinstance(v, dict):
            continue

        vrank = _coerce_int(v.get("rank"))
        vlayout = v.get("layout") if isinstance(v.get("layout"), str) else None

        normalize_shape_vars_from_llm(v.get("shape_vars"), out_shape_vars)

        all_specs: Dict[str, Any] = {}
        if isinstance(v.get("shape_spec_all_params"), dict):
            all_specs.update(v["shape_spec_all_params"])
        if isinstance(v.get("inputs"), dict):
            for pname, pinfo in v["inputs"].items():
                if isinstance(pinfo, dict) and "shape_spec" in pinfo and pname not in all_specs:
                    all_specs[pname] = pinfo.get("shape_spec")

        for pname, raw_spec in all_specs.items():
            if pname not in base_params or not _is_tensor_param(base_params[pname]):
                continue

            param_meta = dict(base_params[pname])
            param_meta["__primary_param__"] = primary_param
            ns = normalize_shape_spec_value(raw_spec, pname, param_meta, out_shape_vars)
            if ns is None:
                continue

            dst = completion["params"][pname]

            if "shape_spec" not in dst:
                dst["shape_spec"] = ns

            if pname == primary_param and vrank is not None:
                if vlayout:
                    sbrl = dst.setdefault("shape_spec_by_rank_and_layout", {})
                    sbrl.setdefault(str(vrank), {})
                    if vlayout not in sbrl[str(vrank)]:
                        sbrl[str(vrank)][vlayout] = ns
                else:
                    sbr = dst.setdefault("shape_spec_by_rank", {})
                    if str(vrank) not in sbr:
                        sbr[str(vrank)] = ns

    _record_source_mode(completion, "variant_style")

def apply_family_specific_shape_rules(
    completion: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> bool:
    """
    Apply high-confidence family-specific shape rules.
    Returns True if any rule was applied.
    """
    op_family = base_yaml.get("op_family")
    params = completion.get("params") or {}
    shape_vars = completion.get("shape_vars") or {}
    test_ranks = completion.get("test_ranks") or rank_plan.get("test_ranks") or []

    changed = False

    def sv(name: str, lo: int, hi: int) -> None:
        _ensure_shape_var(shape_vars, name, [lo, hi])

    def primary_name() -> Optional[str]:
        return base_yaml.get("primary_param")

    def set_primary_sbr_from_dims(pname: str, rank_to_dims: Dict[str, List[str]]) -> None:
        nonlocal changed
        if pname not in params or not rank_to_dims:
            return
        params[pname]["shape_spec_by_rank"] = rank_to_dims
        min_rank_key = str(min(int(k) for k in rank_to_dims.keys()))
        params[pname]["shape_spec"] = list(rank_to_dims[min_rank_key])
        changed = True

    # ---------- matmul ----------
    if op_family == "matmul" and "a" in params and "b" in params:
        sv("M", 1, 16)
        sv("K", 1, 64)
        sv("N", 1, 16)

        params["a"]["shape_spec"] = ["M", "K"]
        params["a"]["shape_spec_by_rank"] = {"2": ["M", "K"]}

        params["b"]["shape_spec"] = ["K", "N"]
        changed = True

    # ---------- conv1d ----------
    elif op_family == "conv1d" and "input" in params:
        sv("N", 1, 8)
        sv("W", 1, 32)
        sv("C_in", 1, 64)
        sv("C_out", 1, 64)
        sv("kW", 1, 11)

        params["input"]["shape_spec"] = ["N", "W", "C_in"]
        params["input"]["shape_spec_by_rank"] = {"3": ["N", "W", "C_in"]}
        params["input"]["shape_spec_by_rank_and_layout"] = {
            "3": {
                "NWC": ["N", "W", "C_in"],
                "NCW": ["N", "C_in", "W"],
            }
        }

        filt_name = "filters" if "filters" in params else "filter" if "filter" in params else None
        if filt_name:
            params[filt_name]["shape_spec"] = ["kW", "C_in", "C_out"]
        changed = True

    # ---------- conv2d ----------
    elif op_family == "conv2d" and "input" in params:
        sv("N", 1, 8)
        sv("H", 1, 32)
        sv("W", 1, 32)
        sv("C_in", 1, 64)
        sv("C_out", 1, 64)
        sv("kH", 1, 11)
        sv("kW", 1, 11)

        params["input"]["shape_spec"] = ["N", "H", "W", "C_in"]
        params["input"]["shape_spec_by_rank"] = {"4": ["N", "H", "W", "C_in"]}
        params["input"]["shape_spec_by_rank_and_layout"] = {
            "4": {
                "NHWC": ["N", "H", "W", "C_in"],
                "NCHW": ["N", "C_in", "H", "W"],
            }
        }

        filt_name = "filters" if "filters" in params else "filter" if "filter" in params else None
        if filt_name:
            params[filt_name]["shape_spec"] = ["kH", "kW", "C_in", "C_out"]
        changed = True

    # ---------- conv3d ----------
    elif op_family == "conv3d" and "input" in params:
        sv("N", 1, 8)
        sv("D", 1, 16)
        sv("H", 1, 32)
        sv("W", 1, 32)
        sv("C_in", 1, 64)
        sv("C_out", 1, 64)
        sv("kD", 1, 7)
        sv("kH", 1, 7)
        sv("kW", 1, 7)

        params["input"]["shape_spec"] = ["N", "D", "H", "W", "C_in"]
        params["input"]["shape_spec_by_rank"] = {"5": ["N", "D", "H", "W", "C_in"]}
        params["input"]["shape_spec_by_rank_and_layout"] = {
            "5": {
                "NDHWC": ["N", "D", "H", "W", "C_in"],
                "NCDHW": ["N", "C_in", "D", "H", "W"],
            }
        }

        filt_name = "filters" if "filters" in params else "filter" if "filter" in params else None
        if filt_name:
            params[filt_name]["shape_spec"] = ["kD", "kH", "kW", "C_in", "C_out"]
        changed = True

    # ---------- depthwise_conv2d ----------
    elif op_family == "depthwise_conv2d" and "input" in params:
        sv("N", 1, 8)
        sv("H", 1, 32)
        sv("W", 1, 32)
        sv("C_in", 1, 64)
        sv("M", 1, 8)   # channel_multiplier
        sv("kH", 1, 11)
        sv("kW", 1, 11)

        params["input"]["shape_spec"] = ["N", "H", "W", "C_in"]
        params["input"]["shape_spec_by_rank"] = {"4": ["N", "H", "W", "C_in"]}
        params["input"]["shape_spec_by_rank_and_layout"] = {
            "4": {
                "NHWC": ["N", "H", "W", "C_in"],
                "NCHW": ["N", "C_in", "H", "W"],
            }
        }

        filt_name = "filter" if "filter" in params else "filters" if "filters" in params else None
        if filt_name:
            params[filt_name]["shape_spec"] = ["kH", "kW", "C_in", "M"]
        changed = True

    # ---------- bias_add ----------
    elif op_family == "bias_add" and "value" in params and "bias" in params:
        sv("N", 1, 8)
        sv("C", 1, 64)
        sv("L", 1, 32)
        sv("H", 1, 32)
        sv("W", 1, 32)
        sv("D", 1, 16)

        value = params["value"]
        bias = params["bias"]

        sbr: Dict[str, List[str]] = {}
        for r in test_ranks:
            if r == 1:
                sbr["1"] = ["C"]
            elif r == 2:
                sbr["2"] = ["N", "C"]
            elif r == 3:
                sbr["3"] = ["N", "L", "C"]
            elif r == 4:
                sbr["4"] = ["N", "H", "W", "C"]
            elif r == 5:
                sbr["5"] = ["N", "D", "H", "W", "C"]

        if sbr:
            value["shape_spec_by_rank"] = sbr
            min_rank_key = str(min(int(k) for k in sbr.keys()))
            value["shape_spec"] = list(sbr[min_rank_key])
            changed = True

        value["shape_spec_by_rank_and_layout"] = {
            "4": {
                "NHWC": ["N", "H", "W", "C"],
                "NCHW": ["N", "C", "H", "W"],
            }
        }
        bias["shape_spec"] = ["C"]
        changed = True

    # ---------- pool2d ----------
    elif op_family == "pool2d" and "input" in params:
        sv("N", 1, 8)
        sv("H", 1, 32)
        sv("W", 1, 32)
        sv("C", 1, 64)

        params["input"]["shape_spec"] = ["N", "H", "W", "C"]
        params["input"]["shape_spec_by_rank"] = {"4": ["N", "H", "W", "C"]}
        params["input"]["shape_spec_by_rank_and_layout"] = {
            "4": {
                "NHWC": ["N", "H", "W", "C"],
                "NCHW": ["N", "C", "H", "W"],
            }
        }
        changed = True

    # ---------- pool3d ----------
    elif op_family == "pool3d" and "input" in params:
        sv("N", 1, 8)
        sv("D", 1, 16)
        sv("H", 1, 32)
        sv("W", 1, 32)
        sv("C", 1, 64)

        params["input"]["shape_spec"] = ["N", "D", "H", "W", "C"]
        params["input"]["shape_spec_by_rank"] = {"5": ["N", "D", "H", "W", "C"]}
        params["input"]["shape_spec_by_rank_and_layout"] = {
            "5": {
                "NDHWC": ["N", "D", "H", "W", "C"],
                "NCDHW": ["N", "C", "D", "H", "W"],
            }
        }
        changed = True

    # ---------- reduce ----------
    elif op_family == "reduce":
        pname = "input_tensor" if "input_tensor" in params else "input" if "input" in params else primary_name()
        if pname and pname in params:
            sbr = {}
            for r in test_ranks:
                dims = []
                for i in range(r):
                    name = f"DIM{i}"
                    sv(name, 1, 32)
                    dims.append(name)
                sbr[str(r)] = dims
            set_primary_sbr_from_dims(pname, sbr)

            if "axis" in params:
                params["axis"]["shape_spec"] = []
                changed = True

    # ---------- gather ----------
    elif op_family == "gather" and "params" in params:
        sbr = {}
        for r in test_ranks:
            dims = []
            for i in range(r):
                name = f"DIM{i}"
                sv(name, 1, 32)
                dims.append(name)
            sbr[str(r)] = dims
        set_primary_sbr_from_dims("params", sbr)

        if "indices" in params:
            sv("I", 1, 16)
            params["indices"]["shape_spec"] = ["I"]   # conservative default
            changed = True

        if "axis" in params:
            params["axis"]["shape_spec"] = []         # scalar int tensor
            changed = True

    # ---------- concat ----------
    elif op_family == "concat":
        values_name = "values" if "values" in params else None
        if values_name:
            min_rank = min(test_ranks) if test_ranks else 2
            dims = []
            for i in range(min_rank):
                name = f"DIM{i}"
                sv(name, 1, 32)
                dims.append(name)
            params[values_name]["shape_spec"] = dims
            changed = True

        if "axis" in params:
            params["axis"]["shape_spec"] = []
            changed = True

    # ---------- split ----------
    elif op_family == "split" and "value" in params:
        sbr = {}
        for r in test_ranks:
            dims = []
            for i in range(r):
                name = f"DIM{i}"
                sv(name, 1, 32)
                dims.append(name)
            sbr[str(r)] = dims
        set_primary_sbr_from_dims("value", sbr)

        if "axis" in params:
            params["axis"]["shape_spec"] = []
            changed = True

    # ---------- transpose ----------
    elif op_family == "transpose":
        pname = "a" if "a" in params else primary_name()
        if pname and pname in params:
            sbr = {}
            for r in test_ranks:
                dims = []
                for i in range(r):
                    name = f"DIM{i}"
                    sv(name, 1, 32)
                    dims.append(name)
                sbr[str(r)] = dims
            set_primary_sbr_from_dims(pname, sbr)

        if "perm" in params:
            sv("R", 1, max(test_ranks) if test_ranks else 4)
            params["perm"]["shape_spec"] = ["R"]
            changed = True

    # ---------- reshape ----------
    elif op_family == "reshape":
        pname = "tensor" if "tensor" in params else "input" if "input" in params else primary_name()
        if pname and pname in params:
            sbr = {}
            for r in test_ranks:
                dims = []
                for i in range(r):
                    name = f"DIM{i}"
                    sv(name, 1, 32)
                    dims.append(name)
                sbr[str(r)] = dims
            set_primary_sbr_from_dims(pname, sbr)

        if "shape" in params:
            sv("R", 1, max(test_ranks) if test_ranks else 4)
            params["shape"]["shape_spec"] = ["R"]    # 1-D int tensor
            changed = True

    # ---------- softmax / activation ----------
    elif op_family in {"softmax", "activation_hl"}:
        pname = primary_name()
        if pname and pname in params:
            sbr = {}
            for r in test_ranks:
                dims = []
                for i in range(r):
                    name = f"DIM{i}"
                    sv(name, 1, 32)
                    dims.append(name)
                sbr[str(r)] = dims
            set_primary_sbr_from_dims(pname, sbr)

        if "axis" in params:
            params["axis"]["shape_spec"] = []
            changed = True

    # ---------- one_hot ----------
    elif op_family == "one_hot" and "indices" in params:
        sbr = {}
        for r in test_ranks:
            dims = []
            for i in range(r):
                name = f"DIM{i}"
                sv(name, 1, 32)
                dims.append(name)
            sbr[str(r)] = dims
        set_primary_sbr_from_dims("indices", sbr)

        if "depth" in params:
            params["depth"]["shape_spec"] = []       # scalar int
            changed = True

    return changed

def derive_primary_sbr_from_existing_shape_spec(
    completion: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> bool:
    """
    If primary already has a valid shape_spec but no shape_spec_by_rank,
    derive shape_spec_by_rank from it instead of falling back to generic DIM0/DIM1.

    Returns True if shape_spec_by_rank was derived.
    """
    primary_param = base_yaml.get("primary_param")
    if not primary_param:
        return False

    params = completion.get("params") or {}
    p = params.get(primary_param)
    if not isinstance(p, dict):
        return False

    test_ranks = completion.get("test_ranks") or rank_plan.get("test_ranks") or []
    sbr = p.get("shape_spec_by_rank") or {}
    ss = p.get("shape_spec")

    if sbr:
        return False
    if not (isinstance(ss, list) and all(isinstance(x, str) for x in ss)):
        return False

    # Fixed-rank case: if there is only one test rank, trust the existing shape_spec
    if len(test_ranks) == 1:
        p["shape_spec_by_rank"] = {str(test_ranks[0]): list(ss)}
        return True

    # Otherwise, only derive if the existing shape_spec length matches a tested rank
    if len(ss) in test_ranks:
        p["shape_spec_by_rank"] = {str(len(ss)): list(ss)}
        return True

    return False

def finalize_completion_structure(
    completion: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> None:
    base_params = base_yaml.get("params") or {}
    primary_param = base_yaml.get("primary_param")
    out_shape_vars = completion["shape_vars"]
    test_ranks = completion.get("test_ranks") or rank_plan.get("test_ranks") or []

    # 1) Apply family-specific rules first
    apply_family_specific_shape_rules(
        completion=completion,
        base_yaml=base_yaml,
        rank_plan=rank_plan,
    )

    # 2) Primary param alignment / normalization
    if primary_param and primary_param in completion["params"]:
        p = completion["params"][primary_param]
        sbr = p.get("shape_spec_by_rank") or {}
        sbrl = p.get("shape_spec_by_rank_and_layout") or {}

        # If only rank+layout table exists, derive plain sbr from default layout when possible
        if not sbr and sbrl:
            df_param = base_params.get("data_format") or {}
            default_layout = df_param.get("default") if isinstance(df_param, dict) else None
            for rk, layout_map in sbrl.items():
                if not isinstance(layout_map, dict):
                    continue
                if default_layout and default_layout in layout_map:
                    sbr[rk] = layout_map[default_layout]
                else:
                    for _lname, spec in layout_map.items():
                        sbr[rk] = spec
                        break
            if sbr:
                p["shape_spec_by_rank"] = sbr

        # If there is already a concrete primary shape_spec, derive sbr from it before generic fallback
        derived = derive_primary_sbr_from_existing_shape_spec(
            completion=completion,
            base_yaml=base_yaml,
            rank_plan=rank_plan,
        )
        if derived:
            sbr = p.get("shape_spec_by_rank") or {}

        # Generic fallback only if still missing
        if not p.get("shape_spec_by_rank"):
            synthesized: Dict[str, List[str]] = {}
            for r in test_ranks:
                rank_spec = []
                for i in range(r):
                    vname = f"DIM{i}"
                    _ensure_shape_var(out_shape_vars, vname)
                    rank_spec.append(vname)
                synthesized[str(r)] = rank_spec
            if synthesized:
                p["shape_spec_by_rank"] = synthesized
                _merge_warning(completion, "primary shape_spec_by_rank synthesized generically")

        # IMPORTANT: always align primary shape_spec with the minimum-rank entry of shape_spec_by_rank
        sbr = p.get("shape_spec_by_rank") or {}
        if sbr:
            min_rank_key = str(min(int(k) for k in sbr.keys()))
            p["shape_spec"] = list(sbr[min_rank_key])

    # 3) Aux params fallback
    for pname, base_spec in base_params.items():
        if not _is_tensor_param(base_spec):
            continue
        dst = completion["params"][pname]
        if "shape_spec" in dst:
            continue

        sem_role = base_spec.get("semantic_role", "")
        if pname == primary_param:
            continue

        if sem_role == "index_input":
            _ensure_shape_var(out_shape_vars, "I")
            dst["shape_spec"] = ["I"]
            _merge_warning(completion, f"{pname}: shape_spec synthesized as generic index_input")

        elif sem_role == "shape_control":
            dst["shape_spec"] = []
            _merge_warning(completion, f"{pname}: shape_spec synthesized as scalar shape_control")

        elif sem_role == "scalar_attr":
            dst["shape_spec"] = []
            _merge_warning(completion, f"{pname}: shape_spec synthesized as scalar attr")

        elif base_spec.get("role") == "aux":
            _ensure_shape_var(out_shape_vars, "AUX0")
            dst["shape_spec"] = ["AUX0"]
            _merge_warning(completion, f"{pname}: shape_spec synthesized as generic aux tensor")

        else:
            _ensure_shape_var(out_shape_vars, "GEN0")
            dst["shape_spec"] = ["GEN0"]
            _merge_warning(completion, f"{pname}: shape_spec synthesized generically")

    # 4) Ensure every referenced symbolic token exists in shape_vars
    for pname, pinfo in completion["params"].items():
        for tok in pinfo.get("shape_spec") or []:
            if isinstance(tok, str):
                _ensure_shape_var(out_shape_vars, tok)

        for _rk, spec in (pinfo.get("shape_spec_by_rank") or {}).items():
            for tok in spec:
                if isinstance(tok, str):
                    _ensure_shape_var(out_shape_vars, tok)

        for _rk, layout_map in (pinfo.get("shape_spec_by_rank_and_layout") or {}).items():
            for _layout, spec in layout_map.items():
                for tok in spec:
                    if isinstance(tok, str):
                        _ensure_shape_var(out_shape_vars, tok)


def extract_whitelisted_completion(
    llm_obj: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> Dict[str, Any]:
    completion = _new_internal_completion(base_yaml, rank_plan)

    normalize_shape_vars_from_llm(llm_obj.get("shape_vars"), completion["shape_vars"])
    completion["test_ranks"] = choose_test_ranks(llm_obj.get("test_ranks"), rank_plan)
    completion["test_dtype_choices"] = choose_test_dtypes(llm_obj.get("test_dtype_choices"), base_yaml, rank_plan)
    completion["layout_variants"] = normalize_layout_variants(llm_obj.get("layout_variants"), rank_plan)

    if isinstance(llm_obj.get("warnings"), list):
        for w in llm_obj["warnings"]:
            if isinstance(w, str) and w.strip():
                _merge_warning(completion, w.strip())

    if isinstance(llm_obj.get("changes"), list):
        for c in llm_obj["changes"]:
            if isinstance(c, str) and c.strip():
                _merge_change(completion, c.strip())

    extract_from_yaml_params_section(llm_obj, base_yaml, completion, rank_plan)
    extract_from_variant_style(llm_obj, base_yaml, completion)
    finalize_completion_structure(completion, base_yaml, rank_plan)
    return completion


# ════════════════════════════════════════════════════════════════
# 7) validation on normalized completion
# ════════════════════════════════════════════════════════════════

def validate_completion(
    completion: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> List[str]:
    errs: List[str] = []

    base_params = base_yaml.get("params")
    if not isinstance(base_params, dict):
        return ["base YAML missing params dict"]

    primary_param = base_yaml.get("primary_param")
    if not primary_param or primary_param not in base_params:
        errs.append("base YAML missing valid primary_param")
        return errs

    test_ranks = completion.get("test_ranks") or rank_plan.get("test_ranks") or []
    if not isinstance(test_ranks, list) or not test_ranks or not all(isinstance(x, int) and x >= 0 for x in test_ranks):
        errs.append("test_ranks must be non-empty list[int>=0]")

    shape_vars = completion.get("shape_vars") or {}
    if not isinstance(shape_vars, dict):
        errs.append("shape_vars must be dict")
        shape_vars = {}

    for k, v in shape_vars.items():
        if not isinstance(k, str) or not _plain_var_name(k):
            errs.append(f"shape_vars key invalid: {k!r}")
            continue
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)):
            errs.append(f"shape_vars[{k}] must be [int,int], got {v!r}")
            continue
        lo, hi = v
        if lo < 1 or hi < lo:
            errs.append(f"shape_vars[{k}] invalid range {v!r}")

    cparams = completion.get("params") or {}
    if not isinstance(cparams, dict):
        errs.append("completion.params must be dict")
        return errs

    for pname, base_spec in base_params.items():
        if not _is_tensor_param(base_spec):
            continue
        if pname not in cparams:
            errs.append(f"missing completion for tensor param: {pname}")
            continue

        p = cparams[pname]
        if not isinstance(p, dict):
            errs.append(f"completion.params[{pname}] must be dict")
            continue

        ss = p.get("shape_spec")
        sbr = p.get("shape_spec_by_rank")
        sbrl = p.get("shape_spec_by_rank_and_layout")

        if ss is None and not sbr and not sbrl:
            errs.append(f"{pname}: no shape_spec / shape_spec_by_rank / shape_spec_by_rank_and_layout extracted")
            continue

        if ss is not None:
            if not isinstance(ss, list) or not all(isinstance(x, str) for x in ss):
                errs.append(f"{pname}.shape_spec must be list[str]")
            else:
                for tok in ss:
                    if tok not in shape_vars:
                        errs.append(f"{pname}.shape_spec references undefined var: {tok}")

        if pname == primary_param:
            if not isinstance(sbr, dict) or not sbr:
                errs.append(f"{primary_param}.shape_spec_by_rank missing or empty")
            else:
                for r in test_ranks:
                    rk = str(r)
                    if rk not in sbr:
                        errs.append(f"{primary_param}.shape_spec_by_rank missing rank {rk}")
                        continue
                    spec = sbr[rk]
                    if not isinstance(spec, list) or not all(isinstance(x, str) for x in spec):
                        errs.append(f"{primary_param}.shape_spec_by_rank[{rk}] must be list[str]")
                        continue
                    if len(spec) != r:
                        errs.append(
                            f"{primary_param}.shape_spec_by_rank[{rk}] length={len(spec)} != rank={r}"
                        )
                    for tok in spec:
                        if tok not in shape_vars:
                            errs.append(f"{primary_param}.shape_spec_by_rank[{rk}] uses undefined var: {tok}")

        if isinstance(sbrl, dict):
            for rk, layout_map in sbrl.items():
                if not isinstance(layout_map, dict):
                    errs.append(f"{pname}.shape_spec_by_rank_and_layout[{rk}] must be dict")
                    continue
                for layout_name, spec in layout_map.items():
                    if not isinstance(spec, list) or not all(isinstance(x, str) for x in spec):
                        errs.append(f"{pname}.shape_spec_by_rank_and_layout[{rk}][{layout_name}] must be list[str]")
                        continue
                    for tok in spec:
                        if tok not in shape_vars:
                            errs.append(f"{pname}.shape_spec_by_rank_and_layout[{rk}][{layout_name}] undefined var: {tok}")

    return errs


# ════════════════════════════════════════════════════════════════
# 8) merge normalized completion back into original YAML
# ════════════════════════════════════════════════════════════════

def build_merged_yaml_from_completion(
    base_yaml: Dict[str, Any],
    completion: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(base_yaml)
    params = out.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("base YAML missing params dict")

    out["test_ranks"] = list(completion.get("test_ranks") or rank_plan.get("test_ranks") or [])
    out["test_dtype_choices"] = list(completion.get("test_dtype_choices") or rank_plan.get("test_dtype_choices") or [])
    out["layout_variants"] = completion.get("layout_variants") or {}
    out["constraints"] = []

    old_sv = out.get("shape_vars")
    if not isinstance(old_sv, dict):
        old_sv = {}
    merged_sv = dict(old_sv)
    for k, v in (completion.get("shape_vars") or {}).items():
        if isinstance(k, str) and isinstance(v, list) and len(v) == 2:
            merged_sv[k] = [int(v[0]), int(v[1])]
    out["shape_vars"] = merged_sv

    cparams = completion.get("params") or {}
    primary_param = out.get("primary_param")

    for pname, pinfo in cparams.items():
        if pname not in params or not isinstance(params[pname], dict) or not isinstance(pinfo, dict):
            continue
        dst = params[pname]

        if "shape_spec" in pinfo:
            dst["shape_spec"] = list(pinfo["shape_spec"])

        if pname == primary_param and pinfo.get("shape_spec_by_rank"):
            dst["shape_spec_by_rank"] = {
                str(rk): list(spec)
                for rk, spec in pinfo["shape_spec_by_rank"].items()
            }

        if pname == primary_param and pinfo.get("shape_spec_by_rank_and_layout"):
            dst["shape_spec_by_rank_and_layout"] = {
                str(rk): {str(layout): list(spec) for layout, spec in layout_map.items()}
                for rk, layout_map in pinfo["shape_spec_by_rank_and_layout"].items()
            }

        if pname == primary_param and "shape_spec" not in pinfo and "shape_spec_by_rank" in pinfo:
            try:
                min_rank_key = str(min(int(k) for k in pinfo["shape_spec_by_rank"].keys()))
                dst["shape_spec"] = list(pinfo["shape_spec_by_rank"][min_rank_key])
            except Exception:
                pass

    remaining_missing: List[str] = []
    for pname, spec in params.items():
        if not _is_tensor_param(spec):
            continue
        if "shape_spec" not in spec or spec.get("shape_spec") is None:
            remaining_missing.append(pname)
            continue
        ss = spec.get("shape_spec")
        if isinstance(ss, list) and any(x == "TODO_SHAPE" for x in ss):
            remaining_missing.append(pname)

    summary = {
        "primary_param": primary_param,
        "test_ranks": out.get("test_ranks"),
        "test_dtype_choices": out.get("test_dtype_choices"),
        "layout_variants_present": bool(out.get("layout_variants")),
        "shape_vars_keys": sorted((out.get("shape_vars") or {}).keys()),
        "remaining_missing_shape": remaining_missing,
        "source_modes_used": completion.get("source_modes_used") or [],
    }
    return out, summary


# ════════════════════════════════════════════════════════════════
# 9) LLM call
# ════════════════════════════════════════════════════════════════

def call_llm_for_completion(
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
# 10) fallback completion when LLM fails badly
# ════════════════════════════════════════════════════════════════

def _build_fallback_completion(
    yaml_obj: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> Dict[str, Any]:
    completion = _new_internal_completion(yaml_obj, rank_plan)
    primary_param = yaml_obj.get("primary_param")
    params = yaml_obj.get("params") or {}
    test_ranks = completion["test_ranks"]
    shape_vars = completion["shape_vars"]

    if primary_param and primary_param in completion["params"]:
        sbr: Dict[str, List[str]] = {}
        for r in test_ranks:
            spec = []
            for i in range(r):
                vname = f"DIM{i}"
                _ensure_shape_var(shape_vars, vname)
                spec.append(vname)
            sbr[str(r)] = spec
        completion["params"][primary_param]["shape_spec_by_rank"] = sbr
        if test_ranks:
            completion["params"][primary_param]["shape_spec"] = list(sbr[str(min(test_ranks))])

    for pname, base_spec in params.items():
        if not _is_tensor_param(base_spec) or pname == primary_param:
            continue
        sem_role = base_spec.get("semantic_role", "")
        if sem_role == "index_input":
            _ensure_shape_var(shape_vars, "I")
            completion["params"][pname]["shape_spec"] = ["I"]
        elif sem_role == "shape_control":
            completion["params"][pname]["shape_spec"] = []
        elif sem_role == "scalar_attr":
            completion["params"][pname]["shape_spec"] = []
        else:
            _ensure_shape_var(shape_vars, "GEN0")
            completion["params"][pname]["shape_spec"] = ["GEN0"]

    _merge_warning(completion, "fallback completion used; LLM output unusable")
    _record_source_mode(completion, "fallback")
    return completion


# ════════════════════════════════════════════════════════════════
# 11) main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage C: LLM-assisted YAML completion for TF API YAML."
    )
    ap.add_argument("--doc_txt", required=True, help="Path to documentation text file for the API")
    ap.add_argument("--yaml_in", required=True, help="Path to input YAML skeleton (from Stage B)")
    ap.add_argument("--yaml_out_dir", required=True, help="Output directory for completed YAML files")
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=6000)
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

    rank_plan = build_rank_plan(yaml_obj)
    print(
        f"[i] Rank plan: test_ranks={rank_plan['test_ranks']}, "
        f"layouts={rank_plan['layouts']}, "
        f"dtype_choices={rank_plan['test_dtype_choices']}, "
        f"variants_to_generate={len(rank_plan['variant_plan'])}"
    )

    system_prompt = TF_YAML_PATCH_SYSTEM_PROMPT
    rank_plan_text = json.dumps(rank_plan, indent=2, ensure_ascii=False)

    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== INPUT YAML (copy this structure; fill only the requested fields) ===\n"
        f"{yaml_text}\n\n"
        "=== PRE-COMPUTED RANK PLAN (must follow) ===\n"
        f"{rank_plan_text}\n\n"
        "Return ONLY ONE COMPLETE YAML document.\n"
        "Do not return JSON.\n"
        "Do not omit existing sections.\n"
        "Fill shape-related fields using YAML param names from params.\n"
    )

    last_errors: List[str] = []
    last_raw: str = ""
    parsed_llm_yaml: Optional[Dict[str, Any]] = None
    final_completion: Optional[Dict[str, Any]] = None

    user_prompt = base_user_prompt

    for attempt in range(args.max_retries + 1):
        try:
            last_raw = call_llm_for_completion(
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

        try:
            parsed_llm_yaml = parse_llm_completed_yaml(last_raw)
        except Exception as e:
            last_errors = [f"LLM YAML parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = (
                    base_user_prompt
                    + "\n\nYour previous output could not be parsed as ONE YAML mapping.\n"
                      "Please return ONLY one valid YAML document.\n"
                )
                continue
            final_completion = _build_fallback_completion(yaml_obj, rank_plan)
            break

        completion = extract_whitelisted_completion(parsed_llm_yaml, yaml_obj, rank_plan)
        last_errors = validate_completion(completion, yaml_obj, rank_plan)

        if not last_errors:
            final_completion = completion
            break

        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\n\nYour previous YAML was parsed, but the extracted completion FAILED validation.\n"
                  "Please fix the YAML and return ONLY ONE COMPLETE YAML document.\n"
                  "Validation errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n"
            )
        else:
            final_completion = completion

    if final_completion is None:
        final_completion = _build_fallback_completion(yaml_obj, rank_plan)

    merged_yaml, summary = build_merged_yaml_from_completion(
        base_yaml=yaml_obj,
        completion=final_completion,
        rank_plan=rank_plan,
    )

    api_name = yaml_obj.get("api_name", "unknown_api")
    out_name = f"{safe_name(api_name)}.yaml"
    out_path = out_dir / out_name
    out_path.write_text(dump_yaml_obj(merged_yaml), encoding="utf-8")

    meta = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(out_path),
        "rank_plan": rank_plan,
        "completion_source_modes": final_completion.get("source_modes_used", []),
        "summary": summary,
        "warnings": final_completion.get("warnings", []),
        "changes": final_completion.get("changes", []),
        "validation_errors": last_errors,
        "raw_model_output_snippet": (last_raw[:12000] if last_raw else ""),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[+] wrote Stage-C yaml: {out_path}")
    if summary.get("remaining_missing_shape"):
        print(f"[!] WARNING: shape still missing for: {summary['remaining_missing_shape']}")
    if last_errors:
        msg = "[!] Stage-C validation errors (not fully fixed):\n" + "\n".join(f"   - {e}" for e in last_errors)
        if args.fail_on_invalid:
            raise SystemExit(msg)
        print(msg)


if __name__ == "__main__":
    main()