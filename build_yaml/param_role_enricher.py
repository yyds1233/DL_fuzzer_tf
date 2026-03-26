#!/usr/bin/env python3
"""
param_role_enricher.py  –  Stage B (enrich): post-process YAML skeletons
with fine-grained semantic role classification and op-family-level rules.

==========================================================================
WHY THIS EXISTS
==========================================================================

The normalize_yaml_skeleton.py only has coarse param roles (primary / aux /
attr / unknown), which causes:

  - `strides`/`dilations` in Conv2D don't get fixed to length 4
  - `shape` in Reshape is classified as `origin: input` + `role: attr`
    (contradictory)
  - `perm` in Transpose is not recognized as an index input
  - No family-level rules for Pool, BatchNorm, etc.

This enricher adds:

  1. Fine-grained `semantic_role` to every param (data_tensor, weight_tensor,
     shape_control, index_input, fixed_arity_list, layout_attr, etc.)
  2. Op-family detection (Conv2D, Pool, MatMul, etc.) with automatic
     application of family-level constraints
  3. LLM-assisted enrichment (optional) for ambiguous params that can't be
     classified by name heuristics alone

==========================================================================
USAGE
==========================================================================

  # Basic (heuristic-only enrichment)
  python param_role_enricher.py \
      --yaml ./tf_yaml_skeleton \
      --out_dir ./tf_yaml_enriched

  # With rank hints from LLM extractor
  python param_role_enricher.py \
      --yaml ./tf_yaml_skeleton \
      --rank_dir ./tf_rank_hints \
      --out_dir ./tf_yaml_enriched

  # With LLM assistance for ambiguous params
  python param_role_enricher.py \
      --yaml ./tf_yaml_skeleton \
      --rank_dir ./tf_rank_hints \
      --out_dir ./tf_yaml_enriched \
      --llm_base_url http://localhost:11434/v1 \
      --llm_model qwen2.5:72b

  # In-place update
  python param_role_enricher.py \
      --yaml ./tf_yaml_skeleton

==========================================================================
WHAT IT CHANGES IN THE YAML
==========================================================================

For each param spec, adds/modifies:
  - `semantic_role`: fine-grained role string
  - `role`: updated coarse role for backward compatibility
  - For `fixed_arity_list` params: fixes `len_range` to [N, N]
  - For `shape_control` params: sets origin=input, kind=tensor (int32/int64)
  - For `index_input` params: sets origin=input, dtype_choices=[int32, int64]
  - For `weight_tensor` params with known rank: updates shape_spec

At the top level, adds/modifies:
  - `op_family`: detected family key (conv2d, pool2d, matmul, etc.)
  - `rank_hints`: may be updated from family rules if missing
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tf_schema_common import (
    ENUM_TODO_MARKER,
    RANK_MISS_MARKER,
    classify_param_semantic_role,
    find_op_family,
    safe_name,
    load_json,
    dump_json,
    normalize_rank_hints,
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
    OP_FAMILY_RULES,
)

GENERATOR_BLOCK = {
    "stage": "B-enrich-tf",
    "version": "2026-03-23-tf-v3",
}

# Semantic role → coarse role mapping
_SEMANTIC_TO_COARSE = {
    SEMANTIC_ROLE_DATA_TENSOR: "primary",
    SEMANTIC_ROLE_WEIGHT_TENSOR: "aux",
    SEMANTIC_ROLE_AUX_TENSOR: "aux",
    SEMANTIC_ROLE_INDEX_INPUT: "aux",
    SEMANTIC_ROLE_SHAPE_CONTROL: "attr",
    SEMANTIC_ROLE_FIXED_ARITY_LIST: "attr",
    SEMANTIC_ROLE_LAYOUT_ATTR: "attr",
    SEMANTIC_ROLE_SCALAR_ATTR: "attr",
    SEMANTIC_ROLE_DTYPE_ATTR: "attr",
    SEMANTIC_ROLE_META: "attr",
}


# ── op family detection ──────────────────────────────────────────

def detect_op_family(data: Dict[str, Any]) -> Optional[str]:
    """Detect which op family this YAML belongs to. Returns family key or None."""
    tf_block = data.get("tf") or {}
    op_name = tf_block.get("op_name") or ""
    if not op_name:
        api_name = data.get("api_name", "")
        op_name = api_name.split(".")[-1]

    for family_key, rule in OP_FAMILY_RULES.items():
        for pattern in rule.get("match", []):
            if op_name == pattern:
                return family_key
        # Partial match
        for pattern in rule.get("match", []):
            if pattern in op_name:
                return family_key
    return None


# ── semantic role assignment ─────────────────────────────────────

def assign_semantic_roles(
    data: Dict[str, Any],
    rank_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Assign fine-grained semantic_role to each param.
    Uses:
    1. rank_details from LLM extractor (if available, has semantic_role)
    2. Name-based heuristics from classify_param_semantic_role()
    3. Op context (origin, kind fields)

    Returns stats dict.
    """
    params = data.get("params") or {}
    tf_block = data.get("tf") or {}
    op_name = tf_block.get("op_name") or data.get("api_name", "").split(".")[-1]

    stats = {"roles_assigned": 0, "roles_from_llm": 0, "roles_from_heuristic": 0}

    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue

        # Check if LLM rank_details has a semantic_role for this param
        llm_role = None
        if rank_details and pname in rank_details:
            rd = rank_details[pname]
            if isinstance(rd, dict) and rd.get("semantic_role"):
                llm_role = rd["semantic_role"]

        if llm_role:
            spec["semantic_role"] = llm_role
            stats["roles_from_llm"] += 1
        else:
            # Heuristic assignment considering current spec context
            sem_role = _heuristic_semantic_role(pname, spec, op_name)
            spec["semantic_role"] = sem_role
            stats["roles_from_heuristic"] += 1

        # Update coarse role for backward compat
        sem = spec["semantic_role"]
        coarse = _SEMANTIC_TO_COARSE.get(sem, "attr")

        # Special: primary_param keeps role="primary"
        if pname == data.get("primary_param"):
            coarse = "primary"

        spec["role"] = coarse
        stats["roles_assigned"] += 1

    return stats


def _heuristic_semantic_role(pname: str, spec: Dict[str, Any], op_name: str) -> str:
    """
    Determine semantic role using name + spec context.
    More nuanced than classify_param_semantic_role() alone.
    """
    origin = spec.get("origin", "")
    kind = spec.get("kind", "")

    # Meta
    if pname == "name":
        return SEMANTIC_ROLE_META

    # dtype attrs
    if kind == "dtype_enum" or pname.lower() in ("t", "out_type", "tidx"):
        return SEMANTIC_ROLE_DTYPE_ATTR

    # If it's explicitly an input tensor, classify by name
    if origin == "input" and kind in ("tensor", "tensor_optional", "tensor_list"):
        base_role = classify_param_semantic_role(pname, op_name)
        # Refine: if classified as scalar_attr but origin=input, it's probably
        # a shape_control or index_input
        if base_role == SEMANTIC_ROLE_SCALAR_ATTR:
            low = pname.lower()
            if "shape" in low:
                return SEMANTIC_ROLE_SHAPE_CONTROL
            if "perm" in low or "ind" in low:
                return SEMANTIC_ROLE_INDEX_INPUT
            # Default: treat as data_tensor if no better match
            return SEMANTIC_ROLE_DATA_TENSOR
        return base_role

    # If it's an int_list attr, check if it's a fixed_arity_list
    if kind == "int_list":
        low = pname.lower()
        if low in ("strides", "dilations", "ksize", "explicit_paddings"):
            return SEMANTIC_ROLE_FIXED_ARITY_LIST
        # Check if the schema says it's an input (int tensor) vs attr
        if origin == "input":
            if "shape" in low:
                return SEMANTIC_ROLE_SHAPE_CONTROL
            return SEMANTIC_ROLE_INDEX_INPUT
        return SEMANTIC_ROLE_SCALAR_ATTR

    # Layout attrs
    if kind == "enum":
        low = pname.lower()
        if low in ("padding", "data_format") or low.endswith("_format"):
            return SEMANTIC_ROLE_LAYOUT_ATTR

    # Bool / int / float attrs
    if kind in ("bool", "int", "float"):
        return SEMANTIC_ROLE_SCALAR_ATTR

    # String optional
    if kind == "string_optional":
        if pname == "name":
            return SEMANTIC_ROLE_META
        return SEMANTIC_ROLE_SCALAR_ATTR

    # Fallback to name-based
    return classify_param_semantic_role(pname, op_name)


# ── family-level rule application ────────────────────────────────

def apply_family_rules(data: Dict[str, Any], family_key: str) -> Dict[str, int]:
    """
    Apply op-family-specific constraints to the YAML data.
    Returns stats dict.
    """
    rule = OP_FAMILY_RULES.get(family_key)
    if not rule:
        return {"family_fixes": 0}

    params = data.get("params") or {}
    stats = {"family_fixes": 0}

    # 1. Fix fixed_arity_list params to correct lengths
    fixed_arity = rule.get("fixed_arity_params") or {}
    for pname, required_len in fixed_arity.items():
        if pname in params:
            spec = params[pname]
            if not isinstance(spec, dict):
                continue
            if spec.get("kind") == "int_list":
                old_len = spec.get("len_range")
                new_len = [required_len, required_len]
                if old_len != new_len:
                    spec["len_range"] = new_len
                    spec["semantic_role"] = SEMANTIC_ROLE_FIXED_ARITY_LIST
                    stats["family_fixes"] += 1

                # Also fix range: strides/dilations/ksize should be >= 1
                rng = spec.get("range")
                if isinstance(rng, list) and len(rng) == 2:
                    if pname.lower() in ("strides", "dilations", "ksize") and rng[0] < 1:
                        rng[0] = 1
                        stats["family_fixes"] += 1
            elif spec.get("kind") == "tensor":
                # Some ops have strides as a tensor input — fix to int_list
                spec["origin"] = "attr"
                spec["kind"] = "int_list"
                spec["len_range"] = [required_len, required_len]
                spec["range"] = [1, 4]
                spec["semantic_role"] = SEMANTIC_ROLE_FIXED_ARITY_LIST
                spec["role"] = "attr"
                # Remove tensor-specific fields
                spec.pop("dtype_choices", None)
                spec.pop("dtype_from_attr", None)
                spec.pop("shape_spec", None)
                stats["family_fixes"] += 1

    # 2. Fix shape_control params
    shape_controls = rule.get("shape_control_params") or []
    for pname in shape_controls:
        if pname in params:
            spec = params[pname]
            if not isinstance(spec, dict):
                continue
            spec["semantic_role"] = SEMANTIC_ROLE_SHAPE_CONTROL
            spec["role"] = "attr"
            # Should be an int tensor or int_list
            if spec.get("kind") == "tensor":
                spec["dtype_choices"] = ["int32", "int64"]
                if "shape_spec" not in spec:
                    spec["shape_spec"] = ["TODO_SHAPE"]
            stats["family_fixes"] += 1

    # 3. Fix index_input params
    index_inputs = rule.get("index_input_params") or []
    for pname in index_inputs:
        if pname in params:
            spec = params[pname]
            if not isinstance(spec, dict):
                continue
            spec["semantic_role"] = SEMANTIC_ROLE_INDEX_INPUT
            spec["role"] = "aux"
            if spec.get("kind") in ("tensor", "tensor_optional"):
                spec["dtype_choices"] = ["int32", "int64"]
            stats["family_fixes"] += 1

    # 4. Update rank_hints from family rules if currently missing
    rank_hints = data.get("rank_hints") or {}
    primary_rank = rule.get("primary_rank")
    primary_rank_any = rule.get("primary_rank_any", False)
    primary_rank_min = rule.get("primary_rank_min")

    if rank_hints.get("status") in ("missing", "unassigned"):
        if primary_rank is not None:
            rank_hints["rank_candidates"] = [primary_rank]
            rank_hints["status"] = "assigned"
            rank_hints["marker"] = "__RANK_FROM_FAMILY__"
            stats["family_fixes"] += 1
        elif primary_rank_any:
            rank_hints["rank_any"] = True
            rank_hints["rank_candidates"] = []
            rank_hints["status"] = "assigned"
            rank_hints["marker"] = "__RANK_FROM_FAMILY__"
            stats["family_fixes"] += 1
        if primary_rank_min is not None:
            rank_hints["rank_min"] = primary_rank_min
        data["rank_hints"] = rank_hints

    # 5. Update weight tensor rank if known
    weight_rank = rule.get("weight_rank")
    if weight_rank is not None:
        for pname, spec in params.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("semantic_role") == SEMANTIC_ROLE_WEIGHT_TENSOR:
                # Could annotate expected rank for downstream use
                spec["_expected_rank"] = weight_rank

    return stats


# ── merge LLM rank details into params ───────────────────────────

def merge_llm_rank_details(
    data: Dict[str, Any],
    rank_dir: Optional[Path],
) -> Dict[str, Any]:
    """
    Load rank.json and merge param_rank_details into the YAML.
    Returns the param_rank_details dict (or empty dict).
    """
    if rank_dir is None:
        return data.get("_param_rank_details") or {}

    api_name = data.get("api_name", "")
    p = rank_dir / f"{safe_name(api_name)}.rank.json"
    if not p.exists():
        return data.get("_param_rank_details") or {}

    try:
        rank_data = load_json(p)
        details = rank_data.get("param_rank_details") or {}

        # Also update rank_hints from the rank.json if better
        rh = normalize_rank_hints(rank_data)
        current_rh = data.get("rank_hints") or {}
        if rh.get("status") == "assigned" and current_rh.get("status") != "assigned":
            data["rank_hints"] = rh

        return details
    except Exception:
        return data.get("_param_rank_details") or {}


# ── LLM-assisted param enrichment (optional) ────────────────────

def _build_llm_enrich_prompt(
    api_name: str,
    params: Dict[str, Any],
    op_name: str,
) -> str:
    """Build prompt to ask LLM to classify ambiguous params."""
    ambiguous = []
    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue
        sem = spec.get("semantic_role", "")
        # Only ask about params we're not confident about
        if sem in (SEMANTIC_ROLE_META, SEMANTIC_ROLE_DTYPE_ATTR):
            continue
        kind = spec.get("kind", "")
        origin = spec.get("origin", "")
        ambiguous.append({
            "name": pname,
            "current_semantic_role": sem,
            "origin": origin,
            "kind": kind,
        })

    if not ambiguous:
        return ""

    prompt = f"""For TensorFlow op `{op_name}` (API: {api_name}), classify each parameter's semantic role.

Parameters:
{json.dumps(ambiguous, indent=2)}

For each parameter, return a JSON object:
{{
  "<param_name>": {{
    "semantic_role": "<one of: data_tensor, weight_tensor, aux_tensor, index_input, shape_control, fixed_arity_list, layout_attr, scalar_attr, dtype_attr, meta>",
    "reason": "<brief>"
  }}
}}

Only include params where you disagree with the current_semantic_role.
Return ONLY a JSON object, no markdown fences."""

    return prompt


def llm_enrich_params(
    data: Dict[str, Any],
    llm_client: Any,
) -> Dict[str, int]:
    """Use LLM to refine semantic roles for ambiguous params."""
    if llm_client is None:
        return {"llm_enriched": 0}

    params = data.get("params") or {}
    tf_block = data.get("tf") or {}
    op_name = tf_block.get("op_name") or data.get("api_name", "").split(".")[-1]
    api_name = data.get("api_name", "")

    prompt = _build_llm_enrich_prompt(api_name, params, op_name)
    if not prompt:
        return {"llm_enriched": 0}

    system = ("You are a TensorFlow API expert. Classify parameter semantic roles precisely. "
              "Respond with ONLY a JSON object.")

    raw = llm_client.chat(system, prompt)
    if not raw:
        return {"llm_enriched": 0}

    # Parse response
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        corrections = json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        return {"llm_enriched": 0}

    if not isinstance(corrections, dict):
        return {"llm_enriched": 0}

    stats = {"llm_enriched": 0}
    valid_roles = {
        SEMANTIC_ROLE_DATA_TENSOR, SEMANTIC_ROLE_WEIGHT_TENSOR,
        SEMANTIC_ROLE_AUX_TENSOR, SEMANTIC_ROLE_INDEX_INPUT,
        SEMANTIC_ROLE_SHAPE_CONTROL, SEMANTIC_ROLE_FIXED_ARITY_LIST,
        SEMANTIC_ROLE_LAYOUT_ATTR, SEMANTIC_ROLE_SCALAR_ATTR,
        SEMANTIC_ROLE_DTYPE_ATTR, SEMANTIC_ROLE_META,
    }

    for pname, correction in corrections.items():
        if pname not in params or not isinstance(correction, dict):
            continue
        new_role = correction.get("semantic_role")
        if new_role and new_role in valid_roles:
            spec = params[pname]
            old_role = spec.get("semantic_role")
            if new_role != old_role:
                spec["semantic_role"] = new_role
                spec["role"] = _SEMANTIC_TO_COARSE.get(new_role, "attr")
                stats["llm_enriched"] += 1

    return stats


# ── consistency fixes ────────────────────────────────────────────

def fix_consistency(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Fix any inconsistencies between semantic_role and origin/kind.
    For example:
      - shape_control with origin=attr → change to origin=input
      - index_input with kind=int_list → keep as int_list but note it
      - fixed_arity_list with origin=input → change to origin=attr
    """
    params = data.get("params") or {}
    stats = {"consistency_fixes": 0}

    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue

        sem = spec.get("semantic_role", "")

        # shape_control: should be origin=input, kind=tensor with int dtype
        if sem == SEMANTIC_ROLE_SHAPE_CONTROL:
            if spec.get("kind") not in ("tensor", "tensor_optional", "int_list"):
                # It's marked as shape_control but has wrong kind
                # If it was an int_list attr, keep it — just note the semantic role
                if spec.get("kind") != "int_list":
                    spec["origin"] = "input"
                    spec["kind"] = "tensor"
                    spec["dtype_choices"] = ["int32", "int64"]
                    if "shape_spec" not in spec:
                        spec["shape_spec"] = ["TODO_SHAPE"]
                    stats["consistency_fixes"] += 1

        # index_input: should have int dtype
        if sem == SEMANTIC_ROLE_INDEX_INPUT:
            if spec.get("kind") in ("tensor", "tensor_optional"):
                dc = spec.get("dtype_choices") or []
                if dc and not any(d in dc for d in ("int32", "int64")):
                    spec["dtype_choices"] = ["int32", "int64"]
                    stats["consistency_fixes"] += 1
                elif not dc:
                    spec["dtype_choices"] = ["int32", "int64"]
                    stats["consistency_fixes"] += 1

        # fixed_arity_list: should be origin=attr, kind=int_list
        if sem == SEMANTIC_ROLE_FIXED_ARITY_LIST:
            if spec.get("origin") == "input" and spec.get("kind") == "tensor":
                # This is a tensor that should be an int_list attr
                spec["origin"] = "attr"
                spec["kind"] = "int_list"
                spec["role"] = "attr"
                spec.pop("dtype_choices", None)
                spec.pop("dtype_from_attr", None)
                spec.pop("shape_spec", None)
                if "len_range" not in spec:
                    spec["len_range"] = [1, 4]
                if "range" not in spec:
                    spec["range"] = [0, 4]
                stats["consistency_fixes"] += 1

        # data_tensor / weight_tensor / aux_tensor: should be origin=input
        if sem in (SEMANTIC_ROLE_DATA_TENSOR, SEMANTIC_ROLE_WEIGHT_TENSOR, SEMANTIC_ROLE_AUX_TENSOR):
            if spec.get("origin") != "input":
                # Don't force-change if it's already well-formed attr
                if spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
                    spec["origin"] = "input"
                    stats["consistency_fixes"] += 1

    return stats


# ── main enrichment pipeline ────────────────────────────────────

def enrich_one_yaml(
    data: Dict[str, Any],
    rank_dir: Optional[Path] = None,
    llm_client: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Enrich a single YAML skeleton with semantic roles and family rules.
    """
    all_stats: Dict[str, int] = {}

    # 0. Update generator
    data["generator"] = GENERATOR_BLOCK

    # 1. Load and merge LLM rank details
    rank_details = merge_llm_rank_details(data, rank_dir)

    # 2. Detect op family
    family_key = detect_op_family(data)
    if family_key:
        data["op_family"] = family_key

    # 3. Assign semantic roles
    role_stats = assign_semantic_roles(data, rank_details)
    all_stats.update(role_stats)

    # 4. Apply family rules
    if family_key:
        family_stats = apply_family_rules(data, family_key)
        all_stats.update(family_stats)

    # 5. Optional LLM enrichment for ambiguous params
    if llm_client:
        llm_stats = llm_enrich_params(data, llm_client)
        all_stats.update(llm_stats)

    # 6. Fix consistency issues
    consistency_stats = fix_consistency(data)
    all_stats.update(consistency_stats)

    # 7. Clean up internal fields
    data.pop("_param_rank_details", None)

    return data, all_stats


# ── CLI ──────────────────────────────────────────────────────────

def iter_yaml_files(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(src.glob("*.yaml"))
    return []


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich YAML skeletons with fine-grained semantic roles and family rules."
    )
    ap.add_argument("--yaml", required=True,
                    help="single yaml file or a directory of yamls")
    ap.add_argument("--out_dir", default=None,
                    help="write enriched yamls to a new directory; default overwrites in place")
    ap.add_argument("--rank_dir", default=None,
                    help="directory containing *.rank.json files (from llm_doc_rank_extractor)")
    ap.add_argument("--dry_run", action="store_true",
                    help="report changes without writing files")

    # LLM options (optional, for extra enrichment)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--llm_api_key", default=None)
    ap.add_argument("--llm_delay", type=float, default=0.5)

    args = ap.parse_args()

    src = Path(args.yaml).resolve()
    files = iter_yaml_files(src)
    if not files:
        raise SystemExit(f"No yaml files found under: {src}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    rank_dir = Path(args.rank_dir).resolve() if args.rank_dir else None

    # Build LLM client if requested
    llm_client = None
    if args.llm_base_url:
        from llm_doc_rank_extractor import LLMClient
        llm_client = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key or "",
        )
        print(f"[i] LLM enabled for param enrichment: {args.llm_base_url}")

    total = {
        "files": 0,
        "files_changed": 0,
        "roles_assigned": 0,
        "roles_from_llm": 0,
        "roles_from_heuristic": 0,
        "family_fixes": 0,
        "llm_enriched": 0,
        "consistency_fixes": 0,
    }

    for yp in files:
        raw = yp.read_text(encoding="utf-8", errors="ignore")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            print(f"[!] skip non-dict yaml: {yp}")
            continue

        before = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        new_data, stats = enrich_one_yaml(data, rank_dir, llm_client)
        after = yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True)
        changed = (before != after)

        total["files"] += 1
        if changed:
            total["files_changed"] += 1
        for k in total:
            if k in ("files", "files_changed"):
                continue
            if k in stats:
                total[k] += stats[k]

        family = new_data.get("op_family", "?")

        if args.dry_run:
            print(
                f"[DRY] {yp.name}: changed={changed}, family={family}, "
                f"roles={stats.get('roles_assigned', 0)}, "
                f"family_fixes={stats.get('family_fixes', 0)}, "
                f"consistency={stats.get('consistency_fixes', 0)}"
            )
            continue

        out_path = (out_dir / yp.name) if out_dir is not None else yp
        out_path.write_text(
            yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if changed:
            print(f"[+] enriched: {yp.name} -> {out_path}  family={family}")
        else:
            print(f"[=] unchanged: {yp.name}")

        if llm_client and args.llm_delay > 0:
            time.sleep(args.llm_delay)

    print("\n=== Enrichment Summary ===")
    for k, v in total.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
