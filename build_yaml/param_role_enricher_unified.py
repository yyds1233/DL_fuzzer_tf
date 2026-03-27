#!/usr/bin/env python3
"""
param_role_enricher_unified.py  –  Stage B (enrich): unified enrichment
that handles both raw_ops and high-level TF APIs.

==========================================================================
WHAT THIS ADDS OVER param_role_enricher.py
==========================================================================

1. Extended op family detection using HIGHLEVEL_FAMILY_RULES
2. Handles the case where tf.op_name is None (composite/unresolved APIs)
3. Uses api_name for family matching when op_name is unavailable
4. Reads resolve_info from the YAML for smarter enrichment

The enrichment logic itself is the same — assign semantic roles,
apply family rules, fix consistency. The difference is in HOW the
op family is detected.

==========================================================================
USAGE
==========================================================================

  # Drop-in replacement for param_role_enricher.py
  python param_role_enricher_unified.py \
      --yaml ./tf_yaml_skeleton \
      --out_dir ./tf_yaml_enriched \
      --rank_dir ./tf_rank_hints
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tf_schema_common import (
    ENUM_TODO_MARKER,
    RANK_MISS_MARKER,
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

from tf_schema_common_ext import (
    HIGHLEVEL_FAMILY_RULES,
    find_op_family_extended,
    get_family_rule,
    classify_param_semantic_role_extended,
)

# Import the original enricher's functions
from param_role_enricher import (
    assign_semantic_roles,
    apply_family_rules,
    fix_consistency,
    llm_enrich_params,
    merge_llm_rank_details,
    _SEMANTIC_TO_COARSE,
)

GENERATOR_BLOCK = {
    "stage": "B-enrich-unified-tf",
    "version": "2026-03-23-tf-v3",
}


# ══════════════════════════════════════════════════════════════════
# Extended op family detection
# ══════════════════════════════════════════════════════════════════

def detect_op_family_unified(data: Dict[str, Any]) -> Optional[str]:
    """
    Detect op family for both raw_ops and high-level APIs.

    Tries:
    1. Raw op_name from tf block (original behavior)
    2. api_name function name against extended rules
    3. resolve_info.raw_op_name against standard rules
    """
    tf_block = data.get("tf") or {}
    op_name = tf_block.get("op_name") or ""
    api_name = data.get("api_name", "")
    resolve_info = data.get("resolve_info") or {}
    raw_op_name = resolve_info.get("raw_op_name") or ""

    # Try raw_op_name against standard rules
    if raw_op_name:
        for family_key, rule in OP_FAMILY_RULES.items():
            for pattern in rule.get("match", []):
                if raw_op_name == pattern or pattern in raw_op_name:
                    return family_key

    # Try op_name
    if op_name:
        for family_key, rule in OP_FAMILY_RULES.items():
            for pattern in rule.get("match", []):
                if op_name == pattern or pattern in op_name:
                    return family_key

    # Try api_name against extended rules
    family = find_op_family_extended(api_name, op_name or raw_op_name)
    if family:
        return family

    return None


def apply_family_rules_unified(
    data: Dict[str, Any],
    family_key: str,
) -> Dict[str, int]:
    """
    Apply family rules from both standard and extended rule sets.
    """
    rule = get_family_rule(family_key)
    if not rule:
        return {"family_fixes": 0}

    # The actual application logic is the same as the original
    # We just potentially use rules from HIGHLEVEL_FAMILY_RULES
    return apply_family_rules(data, family_key)


# ══════════════════════════════════════════════════════════════════
# Unified enrichment pipeline
# ══════════════════════════════════════════════════════════════════

def enrich_one_yaml_unified(
    data: Dict[str, Any],
    rank_dir: Optional[Path] = None,
    llm_client: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Enrich a single YAML skeleton (works for both raw_ops and high-level).
    """
    all_stats: Dict[str, int] = {}

    # 0. Update generator
    data["generator"] = GENERATOR_BLOCK

    # 1. Load and merge LLM rank details
    rank_details = merge_llm_rank_details(data, rank_dir)

    # 2. Detect op family (unified)
    family_key = detect_op_family_unified(data)
    if family_key:
        data["op_family"] = family_key

    # 3. Assign semantic roles
    # For high-level APIs, also use the extended semantic role map
    role_stats = _assign_semantic_roles_extended(data, rank_details)
    all_stats.update(role_stats)

    # 4. Apply family rules
    if family_key:
        # Check if rule is in extended rules (need to merge into OP_FAMILY_RULES
        # temporarily for apply_family_rules to work)
        rule = get_family_rule(family_key)
        if rule and family_key not in OP_FAMILY_RULES:
            # Temporarily add to OP_FAMILY_RULES
            OP_FAMILY_RULES[family_key] = rule
        family_stats = apply_family_rules(data, family_key)
        all_stats.update(family_stats)

    # 5. Optional LLM enrichment
    if llm_client:
        llm_stats = llm_enrich_params(data, llm_client)
        all_stats.update(llm_stats)

    # 6. Fix consistency
    consistency_stats = fix_consistency(data)
    all_stats.update(consistency_stats)

    # 7. Clean up
    data.pop("_param_rank_details", None)

    return data, all_stats


def _assign_semantic_roles_extended(
    data: Dict[str, Any],
    rank_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Assign semantic roles using extended classification
    for high-level API parameters.
    """
    params = data.get("params") or {}
    api_name = data.get("api_name", "")
    tf_block = data.get("tf") or {}
    op_name = tf_block.get("op_name") or api_name.split(".")[-1]
    is_raw_ops = (data.get("resolve_info") or {}).get("is_raw_ops", False)

    stats = {"roles_assigned": 0, "roles_from_llm": 0, "roles_from_heuristic": 0}

    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue

        # Check if LLM rank_details has a role
        llm_role = None
        if rank_details and pname in rank_details:
            rd = rank_details[pname]
            if isinstance(rd, dict) and rd.get("semantic_role"):
                llm_role = rd["semantic_role"]

        if llm_role:
            spec["semantic_role"] = llm_role
            stats["roles_from_llm"] += 1
        elif spec.get("semantic_role"):
            # Already assigned (e.g., by LLM classifier) — keep it
            stats["roles_from_heuristic"] += 1
        else:
            # Use extended heuristics
            if is_raw_ops:
                from param_role_enricher import _heuristic_semantic_role
                sem_role = _heuristic_semantic_role(pname, spec, op_name)
            else:
                sem_role = classify_param_semantic_role_extended(
                    pname, op_name, api_name
                )
            spec["semantic_role"] = sem_role
            stats["roles_from_heuristic"] += 1

        # Update coarse role
        sem = spec.get("semantic_role", "")
        coarse = _SEMANTIC_TO_COARSE.get(sem, "attr")
        if pname == data.get("primary_param"):
            coarse = "primary"
        spec["role"] = coarse
        stats["roles_assigned"] += 1

    return stats


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def iter_yaml_files(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(src.glob("*.yaml"))
    return []


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified enrichment for raw_ops + high-level TF API YAMLs."
    )
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--rank_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
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
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = Path(args.rank_dir).resolve() if args.rank_dir else None

    llm_client = None
    if args.llm_base_url:
        from llm_doc_rank_extractor import LLMClient
        llm_client = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key or "",
        )

    total = {"files": 0, "files_changed": 0, "roles_assigned": 0,
             "family_fixes": 0, "consistency_fixes": 0}

    for yp in files:
        raw = yp.read_text(encoding="utf-8", errors="ignore")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            continue

        before = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        new_data, stats = enrich_one_yaml_unified(data, rank_dir, llm_client)
        after = yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True)
        changed = before != after

        total["files"] += 1
        if changed:
            total["files_changed"] += 1
        for k in ("roles_assigned", "family_fixes", "consistency_fixes"):
            total[k] += stats.get(k, 0)

        family = new_data.get("op_family", "?")
        is_raw = (new_data.get("resolve_info") or {}).get("is_raw_ops", False)
        api_type = "raw" if is_raw else "hl"

        if args.dry_run:
            print(f"[DRY] {yp.name}: type={api_type} family={family} changed={changed}")
            continue

        out_path = (out_dir / yp.name) if out_dir else yp
        out_path.write_text(
            yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if changed:
            print(f"[+] {yp.name} type={api_type} family={family}")
        else:
            print(f"[=] {yp.name}")

        if llm_client and args.llm_delay > 0:
            time.sleep(args.llm_delay)

    print(f"\n=== Summary ===")
    for k, v in total.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
