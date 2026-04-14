#!/usr/bin/env python3
"""
llm_patch_yaml_new.py  –  Stage C: LLM-assisted YAML completion
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

from tf_llm_prompts import TF_YAML_PATCH_SYSTEM_PROMPT1


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
    """
    Build the default rank plan for TF Stage C.

    Compared with the older TF version, this is slightly less aggressive:
    - prefer existing test_ranks if already present in the input YAML
    - prefer concrete rank_hints.rank_candidates if present
    - otherwise fall back to family defaults / rank-any defaults
    """
    existing_test_ranks = yaml_obj.get("test_ranks")
    if isinstance(existing_test_ranks, list):
        concrete = sorted(set(r for r in existing_test_ranks if isinstance(r, int) and r >= 0))
        if concrete:
            return concrete

    rank_hints = yaml_obj.get("rank_hints") or {}
    op_family = yaml_obj.get("op_family") or ""

    candidates = rank_hints.get("rank_candidates") or []
    concrete = [r for r in candidates if isinstance(r, int)]
    if concrete:
        return sorted(set(concrete))

    rank_any = rank_hints.get("rank_any", False)
    rank_min = rank_hints.get("rank_min")

    if op_family in _RANK_FIXED_DEFAULTS:
        base = list(_RANK_FIXED_DEFAULTS[op_family])
        if isinstance(rank_min, int):
            base = [r for r in base if r >= rank_min]
        return base or list(_RANK_FIXED_DEFAULTS[op_family])

    if rank_any or rank_hints.get("status") in ("missing", "unassigned", "assigned"):
        base = list(_RANK_ANY_DEFAULTS.get(op_family, _GENERIC_RANK_ANY))
        if isinstance(rank_min, int):
            base = [r for r in base if r >= rank_min]
        if not base:
            return [rank_min] if isinstance(rank_min, int) and rank_min >= 0 else list(_GENERIC_RANK_ANY)
        return sorted(set(base))

    rank_max = rank_hints.get("rank_max")
    if isinstance(rank_max, int):
        if isinstance(rank_min, int):
            return [r for r in range(rank_min, rank_max + 1) if r >= 0] or [rank_max]
        return [rank_max]

    return list(_GENERIC_RANK_ANY)

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
    """
    Strict symbolic normalization.

    Unlike the older TF version, do NOT silently convert concrete numeric shape
    literals like [2, 3] into generated vars. That behavior hides bad LLM output.
    Stage C expects symbolic variables only, similar to the stricter PyTorch path.
    """
    is_primary = pname == param_meta.get("__primary_param__")
    sem_role = param_meta.get("semantic_role", "")

    if vals == []:
        return [], {}

    out: List[str] = []
    pending_defs: Dict[str, List[int]] = {}

    for i, item in enumerate(vals, start=1):
        if not isinstance(item, str):
            return None

        tok = canonicalize_shape_token(item)
        if tok is None:
            return None
        if tok == "__ELLIPSIS__":
            return None

        if not _plain_var_name(tok):
            return None

        pending_defs.setdefault(tok, _default_var_range(tok))
        out.append(tok)

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
    Support symbolic variadic forms such as:
      ["...", M, K]
      [N, C, "..."]

    Expansion is restricted to the provided finite test_ranks.
    """
    out: Dict[str, List[str]] = {}
    vals = _coerce_shape_list(raw_spec)
    if not isinstance(vals, list) or not vals:
        return out

    canonical: List[str] = []
    for item in vals:
        if not isinstance(item, str):
            return out
        tok = canonicalize_shape_token(item)
        if tok is None:
            return out
        canonical.append(tok)

    if canonical.count("__ELLIPSIS__") != 1:
        return out

    ranks = sorted(set(r for r in test_ranks if isinstance(r, int) and r >= 0))
    if not ranks:
        return out

    ell_idx = canonical.index("__ELLIPSIS__")
    prefix_raw = canonical[:ell_idx]
    suffix_raw = canonical[ell_idx + 1:]

    prefix_norm = _normalize_shape_items_no_commit(prefix_raw, pname, param_meta)
    suffix_norm = _normalize_shape_items_no_commit(suffix_raw, pname, param_meta)
    if prefix_norm is None or suffix_norm is None:
        return out

    prefix_tokens, prefix_defs = prefix_norm
    suffix_tokens, suffix_defs = suffix_norm
    pending_defs = dict(prefix_defs)
    pending_defs.update(suffix_defs)

    def middle_tokens(extra: int) -> List[str]:
        if extra <= 0:
            return []

        if ell_idx == 0:
            toks = [f"B{i+1}" for i in range(extra)]
        elif ell_idx == len(canonical) - 1:
            if extra == 1:
                toks = ["L"]
            elif extra == 2:
                toks = ["H", "W"]
            elif extra == 3:
                toks = ["D", "H", "W"]
            else:
                toks = [f"X{i+1}" for i in range(extra)]
        else:
            toks = [f"X{i+1}" for i in range(extra)]

        for tok in toks:
            pending_defs.setdefault(tok, _default_var_range(tok))
        return toks

    for r in ranks:
        extra = r - len(prefix_tokens) - len(suffix_tokens)
        if extra < 0:
            continue
        spec = list(prefix_tokens) + middle_tokens(extra) + list(suffix_tokens)
        if len(spec) == r and all(_plain_var_name(x) for x in spec):
            out[str(r)] = spec

    for k, rng in pending_defs.items():
        _ensure_shape_var(out_shape_vars, k, rng)

    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))

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

    allowed = set(fallback)
    out = sorted(set(
        r for r in (_coerce_int(x) for x in raw_ranks)
        if isinstance(r, int) and r >= 0 and (not allowed or r in allowed)
    ))
    return out or fallback


def _sorted_rank_keys(d: Dict[str, Any]) -> List[str]:
    return sorted(list(d.keys()), key=lambda x: int(x))


def _candidate_list_like_variants(raw_spec: Any) -> bool:
    if not isinstance(raw_spec, list) or not raw_spec:
        return False
    if all(isinstance(x, str) for x in raw_spec):
        return False
    return any(
        isinstance(x, list)
        or (isinstance(x, str) and x.strip().startswith("["))
        for x in raw_spec
    )


def normalize_shape_spec_variants(
    raw_spec: Any,
    pname: str,
    param_meta: Dict[str, Any],
    out_shape_vars: Dict[str, List[int]],
    test_ranks: List[int],
) -> Dict[str, List[str]]:
    """
    Normalize mixed multi-rank candidates such as:
      [[K], [M, K], [B1, M, K]]
      [[K], ["...", M, K]]
      ["[K]", "[..., M, K]"]

    Returns rank -> spec.
    """
    out: Dict[str, List[str]] = {}
    if not _candidate_list_like_variants(raw_spec):
        return out

    for item in raw_spec:
        ns = normalize_shape_spec_value(item, pname, param_meta, out_shape_vars)
        if ns is not None:
            out[str(len(ns))] = ns
            continue

        ell = expand_ellipsis_shape_spec(item, pname, param_meta, out_shape_vars, test_ranks)
        if ell:
            out.update(ell)
            continue

        return {}

    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _merge_used_shape_vars_from_completion(
    completion: Dict[str, Any],
) -> Set[str]:
    used: Set[str] = set()
    for _pname, pinfo in (completion.get("params") or {}).items():
        if not isinstance(pinfo, dict):
            continue

        for tok in pinfo.get("shape_spec") or []:
            if isinstance(tok, str):
                used.add(tok)

        for spec in (pinfo.get("shape_spec_by_rank") or {}).values():
            if isinstance(spec, list):
                for tok in spec:
                    if isinstance(tok, str):
                        used.add(tok)

        for layout_map in (pinfo.get("shape_spec_by_rank_and_layout") or {}).values():
            if isinstance(layout_map, dict):
                for spec in layout_map.values():
                    if isinstance(spec, list):
                        for tok in spec:
                            if isinstance(tok, str):
                                used.add(tok)
    return used

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

    any_extracted = False

    for pname, base_spec in base_params.items():
        if not _is_tensor_param(base_spec):
            continue

        llm_pspec = llm_params.get(pname)
        if not isinstance(llm_pspec, dict):
            continue

        param_meta = dict(base_spec)
        param_meta["__primary_param__"] = primary_param

        extracted: Dict[str, Any] = {}

        # 1) direct single shape_spec
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

        # 4) ranked-entry object list inside shape_spec
        if not extracted.get("shape_spec_by_rank"):
            ranked_sbr = normalize_ranked_shape_entries(
                llm_pspec.get("shape_spec"),
                pname,
                param_meta,
                out_shape_vars,
            )
            if ranked_sbr:
                extracted["shape_spec_by_rank"] = ranked_sbr

        # 5) single variadic spec inside shape_spec
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

        # 6) mixed multi-rank candidates inside shape_spec
        if not extracted.get("shape_spec_by_rank"):
            mixed_sbr = normalize_shape_spec_variants(
                llm_pspec.get("shape_spec"),
                pname,
                param_meta,
                out_shape_vars,
                test_ranks,
            )
            if mixed_sbr:
                extracted["shape_spec_by_rank"] = mixed_sbr

        # choose default shape_spec from rank tables if needed
        if extracted.get("shape_spec_by_rank"):
            min_key = _sorted_rank_keys(extracted["shape_spec_by_rank"])[0]
            extracted["shape_spec"] = list(extracted["shape_spec_by_rank"][min_key])

        elif extracted.get("shape_spec_by_rank_and_layout") and "shape_spec" not in extracted:
            # choose one concrete default from layout tables for convenience
            sbrl2 = extracted["shape_spec_by_rank_and_layout"]
            min_key = _sorted_rank_keys(sbrl2)[0]
            layout_map = sbrl2[min_key]
            if isinstance(layout_map, dict) and layout_map:
                first_layout = sorted(layout_map.keys())[0]
                extracted["shape_spec"] = list(layout_map[first_layout])

        if extracted:
            _merge_param_completion(completion["params"][pname], extracted)
            any_extracted = True

    if any_extracted:
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
    Apply only small, high-confidence aux/control fixes.

    Unlike the older TF path, do NOT synthesize the primary tensor's full
    shape_spec_by_rank here. That should come from LLM extraction or fallback,
    not from family-specific templating during the normal path.
    """
    op_family = base_yaml.get("op_family")
    params = completion.get("params") or {}
    changed = False

    def set_if_missing(pname: str, spec: List[str]) -> None:
        nonlocal changed
        if pname in params and isinstance(params[pname], dict) and "shape_spec" not in params[pname]:
            params[pname]["shape_spec"] = list(spec)
            changed = True

    if op_family == "reduce":
        set_if_missing("axis", [])
        set_if_missing("reduction_indices", [])

    elif op_family == "concat":
        set_if_missing("axis", [])

    elif op_family == "transpose":
        if "perm" in params and isinstance(params["perm"], dict) and "shape_spec" not in params["perm"]:
            _ensure_shape_var(completion["shape_vars"], "R", [1, max(rank_plan.get("test_ranks") or [4])])
            params["perm"]["shape_spec"] = ["R"]
            changed = True

    elif op_family == "reshape":
        if "shape" in params and isinstance(params["shape"], dict) and "shape_spec" not in params["shape"]:
            _ensure_shape_var(completion["shape_vars"], "R", [1, max(rank_plan.get("test_ranks") or [4])])
            params["shape"]["shape_spec"] = ["R"]
            changed = True

    elif op_family == "gather":
        if "indices" in params and isinstance(params["indices"], dict) and "shape_spec" not in params["indices"]:
            _ensure_shape_var(completion["shape_vars"], "I", [1, 16])
            params["indices"]["shape_spec"] = ["I"]
            changed = True
        set_if_missing("axis", [])

    elif op_family == "one_hot":
        set_if_missing("depth", [])

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
    """
    Conservative post-processing after validation succeeds or for best-effort
    merge on the final failed attempt.

    Key differences from the older TF version:
    - do not synthesize generic primary shape_spec_by_rank in the normal path
    - do not auto-fill every aux tensor with generic placeholders here
    - only derive convenience fields from already-extracted structure
    - only add shape_vars for variables already referenced by extracted specs
    """
    base_params = base_yaml.get("params") or {}
    primary_param = base_yaml.get("primary_param")
    out_shape_vars = completion["shape_vars"]

    apply_family_specific_shape_rules(
        completion=completion,
        base_yaml=base_yaml,
        rank_plan=rank_plan,
    )

    for pname, pinfo in (completion.get("params") or {}).items():
        if not isinstance(pinfo, dict):
            continue

        # If only layout-specific rank tables exist, derive a plain rank table.
        sbr = pinfo.get("shape_spec_by_rank") or {}
        sbrl = pinfo.get("shape_spec_by_rank_and_layout") or {}
        if not sbr and isinstance(sbrl, dict) and sbrl:
            derived_sbr: Dict[str, List[str]] = {}
            df_param = base_params.get("data_format") or {}
            default_layout = df_param.get("default") if isinstance(df_param, dict) else None

            for rk in _sorted_rank_keys(sbrl):
                layout_map = sbrl.get(rk)
                if not isinstance(layout_map, dict) or not layout_map:
                    continue
                if default_layout and default_layout in layout_map:
                    derived_sbr[rk] = list(layout_map[default_layout])
                else:
                    first_layout = sorted(layout_map.keys())[0]
                    derived_sbr[rk] = list(layout_map[first_layout])

            if derived_sbr:
                pinfo["shape_spec_by_rank"] = derived_sbr
                sbr = derived_sbr

        # Prefer a deterministic default shape_spec when a rank table exists.
        if isinstance(sbr, dict) and sbr:
            min_key = _sorted_rank_keys(sbr)[0]
            pinfo["shape_spec"] = list(sbr[min_key])
        elif "shape_spec" not in pinfo and isinstance(sbrl, dict) and sbrl:
            min_key = _sorted_rank_keys(sbrl)[0]
            layout_map = sbrl[min_key]
            if isinstance(layout_map, dict) and layout_map:
                first_layout = sorted(layout_map.keys())[0]
                pinfo["shape_spec"] = list(layout_map[first_layout])

    # As a conservative fixed-rank convenience, derive primary rank table from
    # an existing validated shape_spec only when the rank plan is single-rank.
    derive_primary_sbr_from_existing_shape_spec(completion, base_yaml, rank_plan)

    # Only ensure vars already referenced by extracted specs.
    for tok in sorted(_merge_used_shape_vars_from_completion(completion)):
        _ensure_shape_var(out_shape_vars, tok)


def extract_whitelisted_completion_no_finalize(
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
    return completion


def extract_whitelisted_completion(
    llm_obj: Dict[str, Any],
    base_yaml: Dict[str, Any],
    rank_plan: Dict[str, Any],
) -> Dict[str, Any]:
    completion = extract_whitelisted_completion_no_finalize(llm_obj, base_yaml, rank_plan)
    finalize_completion_structure(completion, base_yaml, rank_plan)
    return completion


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

    allowed_test_ranks = set(rank_plan.get("test_ranks") or [])
    test_ranks = completion.get("test_ranks") or rank_plan.get("test_ranks") or []
    if not isinstance(test_ranks, list) or not test_ranks or not all(isinstance(x, int) and x >= 0 for x in test_ranks):
        errs.append("test_ranks must be non-empty list[int>=0]")
    elif allowed_test_ranks and any(r not in allowed_test_ranks for r in test_ranks):
        errs.append(f"test_ranks must be subset of rank_plan.test_ranks={sorted(allowed_test_ranks)}")

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

    def validate_shape_list(owner: str, spec: Any, expected_rank: Optional[int] = None) -> None:
        if not isinstance(spec, list) or not all(isinstance(x, str) for x in spec):
            errs.append(f"{owner} must be list[str]")
            return
        if expected_rank is not None and len(spec) != expected_rank:
            errs.append(f"{owner} length={len(spec)} != rank={expected_rank}")
        for tok in spec:
            if tok not in shape_vars:
                errs.append(f"{owner} references undefined var: {tok}")

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
            validate_shape_list(f"{pname}.shape_spec", ss)

        if isinstance(sbr, dict):
            for rk, spec in sbr.items():
                rki = _coerce_int(rk)
                if rki is None or rki < 0:
                    errs.append(f"{pname}.shape_spec_by_rank has invalid rank key: {rk!r}")
                    continue
                validate_shape_list(f"{pname}.shape_spec_by_rank[{rk}]", spec, rki)

        if isinstance(sbrl, dict):
            for rk, layout_map in sbrl.items():
                rki = _coerce_int(rk)
                if rki is None or rki < 0:
                    errs.append(f"{pname}.shape_spec_by_rank_and_layout has invalid rank key: {rk!r}")
                    continue
                if not isinstance(layout_map, dict):
                    errs.append(f"{pname}.shape_spec_by_rank_and_layout[{rk}] must be dict")
                    continue
                for layout_name, spec in layout_map.items():
                    validate_shape_list(f"{pname}.shape_spec_by_rank_and_layout[{rk}][{layout_name}]", spec, rki)

        if pname == primary_param:
            primary_ranks_covered: Set[int] = set()
            if isinstance(sbr, dict):
                primary_ranks_covered.update(
                    int(rk) for rk in sbr.keys() if isinstance(_coerce_int(rk), int)
                )
            if isinstance(sbrl, dict):
                primary_ranks_covered.update(
                    int(rk) for rk in sbrl.keys() if isinstance(_coerce_int(rk), int)
                )

            if len(test_ranks) >= 2:
                missing = [r for r in test_ranks if r not in primary_ranks_covered]
                if missing:
                    errs.append(f"{primary_param} missing primary rank coverage for: {missing}")
            elif len(test_ranks) == 1:
                only_rank = test_ranks[0]
                ok = False
                if only_rank in primary_ranks_covered:
                    ok = True
                elif isinstance(ss, list) and len(ss) == only_rank:
                    ok = True
                if not ok:
                    errs.append(
                        f"{primary_param} must provide shape_spec or rank-specific shape for rank {only_rank}"
                    )

    return errs


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
    for pname, pinfo in cparams.items():
        if pname not in params or not isinstance(params[pname], dict) or not isinstance(pinfo, dict):
            continue
        dst = params[pname]

        if "shape_spec" in pinfo:
            dst["shape_spec"] = list(pinfo["shape_spec"])

        if pinfo.get("shape_spec_by_rank"):
            dst["shape_spec_by_rank"] = {
                str(rk): list(spec)
                for rk, spec in pinfo["shape_spec_by_rank"].items()
            }

        if pinfo.get("shape_spec_by_rank_and_layout"):
            dst["shape_spec_by_rank_and_layout"] = {
                str(rk): {str(layout): list(spec) for layout, spec in layout_map.items()}
                for rk, layout_map in pinfo["shape_spec_by_rank_and_layout"].items()
            }

        if "shape_spec" not in pinfo and pinfo.get("shape_spec_by_rank"):
            try:
                min_rank_key = _sorted_rank_keys(pinfo["shape_spec_by_rank"])[0]
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
        "primary_param": out.get("primary_param"),
        "test_ranks": out.get("test_ranks"),
        "test_dtype_choices": out.get("test_dtype_choices"),
        "layout_variants_present": bool(out.get("layout_variants")),
        "shape_vars_keys": sorted((out.get("shape_vars") or {}).keys()),
        "remaining_missing_shape": remaining_missing,
        "source_modes_used": completion.get("source_modes_used") or [],
    }
    return out, summary

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

    system_prompt = TF_YAML_PATCH_SYSTEM_PROMPT1
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
        "Use YAML param names from params exactly.\n"
        "Fill only Stage-C shape-related fields:\n"
        "- test_ranks\n"
        "- test_dtype_choices\n"
        "- layout_variants\n"
        "- shape_vars\n"
        "- params.*.shape_spec\n"
        "- params.*.shape_spec_by_rank\n"
        "- params.*.shape_spec_by_rank_and_layout\n"
        "Do not add semantic constraints.\n"
        "Never use concrete numeric shapes like [2, 3]. Use symbolic vars only.\n"
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

        completion = extract_whitelisted_completion_no_finalize(parsed_llm_yaml, yaml_obj, rank_plan)
        last_errors = validate_completion(completion, yaml_obj, rank_plan)

        if not last_errors:
            finalize_completion_structure(completion, yaml_obj, rank_plan)
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
            finalize_completion_structure(completion, yaml_obj, rank_plan)
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

