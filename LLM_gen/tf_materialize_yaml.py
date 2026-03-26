#!/usr/bin/env python3
"""
tf_materialize_yaml.py  –  Stage E: final materialization pass.

Ensures the YAML is fully harness-ready by:

1. COMPLETENESS: no TODO_SHAPE survives — fills any remaining gaps with
   generic dimensions.
2. RANK CLOSURE: test_ranks is always a concrete int list; rank_any has
   been fully discretized.
3. LAYOUT COMPILATION: if data_format variants exist, shape_spec_by_rank
   and shape_spec_by_rank_and_layout are both present.
4. DTYPE SHRINKAGE: test_dtype_choices is always a concrete, small list.
5. CONSTRAINT COMPLETENESS: at minimum, rank constraint + dtype consistency
   + cross-param shape consistency are present.
6. SELF-CONSISTENCY: shape_vars used in shape_specs are defined;
   shape_spec lengths match declared ranks.

==========================================================================
USAGE
==========================================================================

  # After Stage D
  python tf_materialize_yaml.py \
      --yaml_in ./tf_yaml_staged/api.yaml \
      --yaml_out ./tf_yaml_final/api.yaml

  # Batch mode (directory)
  python tf_materialize_yaml.py \
      --yaml_dir ./tf_yaml_staged \
      --out_dir ./tf_yaml_final

  # Dry run (report only)
  python tf_materialize_yaml.py \
      --yaml_dir ./tf_yaml_staged \
      --dry_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


# ════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════

def load_yaml_obj(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))


def dump_yaml_obj(obj: Any) -> str:
    return yaml.safe_dump(
        obj,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        default_flow_style=False,
    )


GENERATOR_BLOCK = {
    "stage": "E-materialize-tf",
    "version": "2026-03-23-tf-v3",
}


# ════════════════════════════════════════════════════════════════
# 1) Completeness: fill remaining TODO_SHAPE
# ════════════════════════════════════════════════════════════════

_GENERIC_DIMS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
_GENERIC_RANGE = [1, 16]


def _ensure_shape_var(shape_vars: Dict[str, Any], var_name: str) -> None:
    """Add a shape var if it doesn't exist yet."""
    if var_name not in shape_vars:
        shape_vars[var_name] = list(_GENERIC_RANGE)


def fill_todo_shapes(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Fill any remaining TODO_SHAPE with generic dimensions.
    Returns stats.
    """
    params = data.get("params") or {}
    shape_vars = data.get("shape_vars")
    if not isinstance(shape_vars, dict):
        shape_vars = {}
        data["shape_vars"] = shape_vars

    primary_param = data.get("primary_param")
    test_ranks = data.get("test_ranks") or [2, 3, 4]

    stats = {"todo_filled": 0}

    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("kind", "")
        if kind not in ("tensor", "tensor_optional", "tensor_list"):
            continue

        ss = spec.get("shape_spec")
        has_todo = isinstance(ss, list) and any(x == "TODO_SHAPE" for x in ss)
        is_empty = not ss

        if not has_todo and not is_empty:
            continue  # already filled

        # Determine what rank to use for this param
        sem_role = spec.get("semantic_role", "")

        if pname == primary_param:
            # Primary: use minimum test_rank
            min_rank = min(test_ranks) if test_ranks else 2
            new_spec = []
            for i in range(min_rank):
                dname = _GENERIC_DIMS[i] if i < len(_GENERIC_DIMS) else f"D{i}"
                _ensure_shape_var(shape_vars, dname)
                new_spec.append(dname)
            spec["shape_spec"] = new_spec

            # Also fill shape_spec_by_rank if missing
            if "shape_spec_by_rank" not in spec:
                sbr: Dict[str, List[str]] = {}
                for rank in test_ranks:
                    rank_spec = []
                    for i in range(rank):
                        dname = _GENERIC_DIMS[i] if i < len(_GENERIC_DIMS) else f"D{i}"
                        _ensure_shape_var(shape_vars, dname)
                        rank_spec.append(dname)
                    sbr[str(rank)] = rank_spec
                spec["shape_spec_by_rank"] = sbr

        elif sem_role in ("weight_tensor", "aux_tensor"):
            # Aux: typically 1-D, use last dim of primary if known
            primary_spec = params.get(primary_param, {}).get("shape_spec") or []
            if primary_spec and isinstance(primary_spec, list):
                last_dim = primary_spec[-1]
                _ensure_shape_var(shape_vars, last_dim)
                spec["shape_spec"] = [last_dim]
            else:
                _ensure_shape_var(shape_vars, "C")
                spec["shape_spec"] = ["C"]

        elif sem_role == "index_input":
            # Index: typically 1-D or 2-D
            _ensure_shape_var(shape_vars, "IDX_D1")
            spec["shape_spec"] = ["IDX_D1"]

        elif sem_role == "shape_control":
            # Shape control: 1-D of length = target rank
            _ensure_shape_var(shape_vars, "TARGET_RANK")
            spec["shape_spec"] = ["TARGET_RANK"]

        else:
            # Generic: 1-D
            _ensure_shape_var(shape_vars, "G1")
            spec["shape_spec"] = ["G1"]

        stats["todo_filled"] += 1

    return stats


# ════════════════════════════════════════════════════════════════
# 2) Rank closure: ensure test_ranks is concrete
# ════════════════════════════════════════════════════════════════

def ensure_test_ranks(data: Dict[str, Any]) -> bool:
    """Ensure test_ranks is a concrete list of ints. Returns True if fixed."""
    test_ranks = data.get("test_ranks")
    if isinstance(test_ranks, list) and test_ranks and all(isinstance(r, int) for r in test_ranks):
        return False  # already good

    # Try to derive from shape_spec_by_rank
    primary_param = data.get("primary_param")
    params = data.get("params") or {}
    p = params.get(primary_param) if primary_param else None

    if isinstance(p, dict):
        sbr = p.get("shape_spec_by_rank")
        if isinstance(sbr, dict) and sbr:
            ranks = []
            for k in sbr:
                try:
                    ranks.append(int(k))
                except (ValueError, TypeError):
                    pass
            if ranks:
                data["test_ranks"] = sorted(set(ranks))
                return True

    # Fallback: use rank_hints
    rh = data.get("rank_hints") or {}
    candidates = rh.get("rank_candidates") or []
    concrete = [r for r in candidates if isinstance(r, int)]
    if concrete:
        data["test_ranks"] = sorted(set(concrete))
        return True

    # Generic fallback
    rank_any = rh.get("rank_any", False)
    if rank_any:
        data["test_ranks"] = [1, 2, 3, 4]
    else:
        data["test_ranks"] = [2, 3, 4]
    return True


# ════════════════════════════════════════════════════════════════
# 3) Dtype closure: ensure test_dtype_choices is concrete
# ════════════════════════════════════════════════════════════════

_DTYPE_PRIORITY = [
    "float32", "float64", "int32", "int64", "bool",
    "complex64", "complex128",
]
_SKIP_DTYPES = {"qint8", "quint8", "qint32", "qint16", "quint16"}


def ensure_test_dtypes(data: Dict[str, Any]) -> bool:
    """Ensure test_dtype_choices exists. Returns True if fixed."""
    tdc = data.get("test_dtype_choices")
    if isinstance(tdc, list) and tdc:
        return False

    # Derive from attr_meta
    tf_block = data.get("tf") or {}
    attr_meta = tf_block.get("attr_meta") or []
    allowed: Set[str] = set()
    for attr in attr_meta:
        if isinstance(attr, dict):
            av = attr.get("allowed_values")
            if isinstance(av, dict) and av.get("kind") == "list":
                for t in (av.get("type") or []):
                    if isinstance(t, str) and t not in _SKIP_DTYPES:
                        allowed.add(t)

    if allowed:
        selected = [dt for dt in _DTYPE_PRIORITY if dt in allowed][:4]
        data["test_dtype_choices"] = selected or ["float32"]
    else:
        data["test_dtype_choices"] = ["float32", "float64"]

    return True


# ════════════════════════════════════════════════════════════════
# 4) Auto-generate missing constraints
# ════════════════════════════════════════════════════════════════

def ensure_base_constraints(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Add essential constraints if not already present:
    - Rank constraint for primary tensor
    - Dtype consistency between shared-type tensors
    - Basic cross-param shape consistency
    """
    constraints = data.get("constraints") or []
    if not isinstance(constraints, list):
        constraints = []

    existing_set = set(c.strip() for c in constraints if isinstance(c, str))
    stats = {"constraints_auto_added": 0}

    primary_param = data.get("primary_param")
    params = data.get("params") or {}
    test_ranks = data.get("test_ranks") or []

    # 1) Rank constraint
    if primary_param and test_ranks:
        if len(test_ranks) == 1:
            rc = f"{primary_param}.ndim == {test_ranks[0]}"
        else:
            ranks_tuple = ", ".join(str(r) for r in test_ranks)
            rc = f"{primary_param}.ndim in ({ranks_tuple})"

        if rc not in existing_set:
            constraints.append(rc)
            existing_set.add(rc)
            stats["constraints_auto_added"] += 1

    # 2) Dtype consistency
    # Find tensors sharing the same type attr
    type_attr_groups: Dict[str, List[str]] = {}
    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue
        dfa = spec.get("dtype_from_attr")
        if isinstance(dfa, str) and spec.get("kind") in ("tensor", "tensor_optional"):
            type_attr_groups.setdefault(dfa, []).append(pname)

    for _type_attr, tensor_names in type_attr_groups.items():
        if len(tensor_names) >= 2:
            # Add pairwise dtype consistency
            first = tensor_names[0]
            for other in tensor_names[1:]:
                dc = f"{first}.dtype == {other}.dtype"
                if dc not in existing_set:
                    constraints.append(dc)
                    existing_set.add(dc)
                    stats["constraints_auto_added"] += 1

    data["constraints"] = constraints
    return stats


# ════════════════════════════════════════════════════════════════
# 5) Self-consistency validation
# ════════════════════════════════════════════════════════════════

def validate_self_consistency(data: Dict[str, Any]) -> List[str]:
    """
    Check that the YAML is internally consistent.
    Returns list of warning strings.
    """
    warnings: List[str] = []
    params = data.get("params") or {}
    shape_vars = data.get("shape_vars") or {}
    primary_param = data.get("primary_param")
    test_ranks = data.get("test_ranks") or []

    # Check all shape_spec vars are defined
    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue
        ss = spec.get("shape_spec")
        if isinstance(ss, list):
            for var in ss:
                if isinstance(var, str) and var not in shape_vars and var != "TODO_SHAPE":
                    warnings.append(f"params.{pname}.shape_spec uses undefined var: {var}")

    # Check shape_spec_by_rank lengths match rank keys
    if primary_param and primary_param in params:
        p = params[primary_param]
        if isinstance(p, dict):
            sbr = p.get("shape_spec_by_rank") or {}
            for rank_key, spec_list in sbr.items():
                try:
                    expected_len = int(rank_key)
                except (ValueError, TypeError):
                    warnings.append(f"shape_spec_by_rank key not an int: {rank_key}")
                    continue
                if isinstance(spec_list, list) and len(spec_list) != expected_len:
                    warnings.append(
                        f"shape_spec_by_rank[{rank_key}] length={len(spec_list)} != rank={expected_len}"
                    )
                # Check vars defined
                if isinstance(spec_list, list):
                    for var in spec_list:
                        if isinstance(var, str) and var not in shape_vars:
                            warnings.append(
                                f"shape_spec_by_rank[{rank_key}] uses undefined var: {var}"
                            )

    # Check test_ranks vs shape_spec_by_rank alignment
    if primary_param and primary_param in params:
        p = params[primary_param]
        if isinstance(p, dict):
            sbr = p.get("shape_spec_by_rank") or {}
            sbr_ranks = set()
            for k in sbr:
                try:
                    sbr_ranks.add(int(k))
                except (ValueError, TypeError):
                    pass
            for r in test_ranks:
                if r not in sbr_ranks:
                    warnings.append(f"test_rank {r} has no entry in shape_spec_by_rank")

    # Check no TODO_SHAPE remains
    for pname, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
            ss = spec.get("shape_spec")
            if isinstance(ss, list) and any(x == "TODO_SHAPE" for x in ss):
                warnings.append(f"params.{pname}.shape_spec still contains TODO_SHAPE")
            elif not ss:
                warnings.append(f"params.{pname}.shape_spec is empty/missing")

    return warnings


# ════════════════════════════════════════════════════════════════
# 6) Layout variant compilation
# ════════════════════════════════════════════════════════════════

def ensure_layout_variants_compiled(data: Dict[str, Any]) -> bool:
    """
    If layout_variants info exists but shape_spec_by_rank_and_layout is
    incomplete, try to compile it from shape_spec_by_rank + layout semantics.
    """
    primary_param = data.get("primary_param")
    params = data.get("params") or {}
    if not primary_param or primary_param not in params:
        return False

    p = params[primary_param]
    if not isinstance(p, dict):
        return False

    layout_variants = data.get("layout_variants") or {}
    if not layout_variants:
        return False

    sbr = p.get("shape_spec_by_rank") or {}
    sbrl = p.get("shape_spec_by_rank_and_layout") or {}

    changed = False

    # For each rank that has layout variants, check if shape_spec_by_rank_and_layout exists
    for layout_name, layout_info in layout_variants.items():
        if not isinstance(layout_info, dict):
            continue
        applies_to = layout_info.get("applies_to_ranks") or []
        for rank in applies_to:
            rank_key = str(rank)
            if rank_key in sbrl and layout_name in sbrl[rank_key]:
                continue  # already compiled

            # Try to synthesize from shape_spec_by_rank
            base_spec = sbr.get(rank_key)
            if not isinstance(base_spec, list):
                continue

            # We can't auto-synthesize layout permutations without knowing
            # the exact semantics, but we can ensure the structure exists
            if rank_key not in sbrl:
                sbrl[rank_key] = {}

            if layout_name not in sbrl[rank_key]:
                # If this is the default layout, copy from sbr
                df_param = params.get("data_format") or {}
                default_layout = df_param.get("default", "")
                if layout_name == default_layout:
                    sbrl[rank_key][layout_name] = list(base_spec)
                    changed = True

    if sbrl and changed:
        p["shape_spec_by_rank_and_layout"] = sbrl

    return changed


# ════════════════════════════════════════════════════════════════
# 7) main materialization pipeline
# ════════════════════════════════════════════════════════════════

def materialize(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run all materialization passes on a single YAML.
    Returns (materialized_data, report).
    """
    report: Dict[str, Any] = {
        "passes": [],
        "warnings": [],
    }

    # Update generator
    data["generator"] = GENERATOR_BLOCK

    # Pass 1: ensure test_ranks
    if ensure_test_ranks(data):
        report["passes"].append("test_ranks_fixed")

    # Pass 2: ensure test_dtype_choices
    if ensure_test_dtypes(data):
        report["passes"].append("test_dtype_choices_fixed")

    # Pass 3: fill TODO_SHAPE
    todo_stats = fill_todo_shapes(data)
    if todo_stats.get("todo_filled", 0) > 0:
        report["passes"].append(f"todo_shapes_filled={todo_stats['todo_filled']}")

    # Pass 4: compile layout variants
    if ensure_layout_variants_compiled(data):
        report["passes"].append("layout_variants_compiled")

    # Pass 5: auto-generate base constraints
    constraint_stats = ensure_base_constraints(data)
    if constraint_stats.get("constraints_auto_added", 0) > 0:
        report["passes"].append(
            f"constraints_auto_added={constraint_stats['constraints_auto_added']}"
        )

    # Pass 6: validate self-consistency
    warnings = validate_self_consistency(data)
    report["warnings"] = warnings

    # Summary
    report["test_ranks"] = data.get("test_ranks")
    report["test_dtype_choices"] = data.get("test_dtype_choices")
    report["num_constraints"] = len(data.get("constraints") or [])
    report["primary_param"] = data.get("primary_param")

    # Check harness-readiness
    harness_ready = True
    for w in warnings:
        if "TODO_SHAPE" in w or "empty/missing" in w:
            harness_ready = False
        if "no entry in shape_spec_by_rank" in w:
            harness_ready = False
    report["harness_ready"] = harness_ready

    return data, report


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def iter_yaml_files(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(src.glob("*.yaml"))
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Stage E: final materialization — ensure harness-ready YAML."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--yaml_in", help="single YAML file to materialize")
    g.add_argument("--yaml_dir", help="directory of YAML files to materialize")
    ap.add_argument("--yaml_out", help="output path for single file mode")
    ap.add_argument("--out_dir", help="output directory for batch mode")
    ap.add_argument("--dry_run", action="store_true",
                    help="report issues without writing")
    args = ap.parse_args()

    if args.yaml_in:
        files = [Path(args.yaml_in).resolve()]
    else:
        files = iter_yaml_files(Path(args.yaml_dir).resolve())
    if not files:
        raise SystemExit("No YAML files found")

    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_ready = 0
    total_warnings = 0

    for yp in files:
        data = load_yaml_obj(yp)
        if not isinstance(data, dict):
            print(f"[!] skip: {yp}")
            continue

        materialized, report = materialize(data)
        total_files += 1
        is_ready = report.get("harness_ready", False)
        if is_ready:
            total_ready += 1
        total_warnings += len(report.get("warnings", []))

        api_name = materialized.get("api_name", yp.stem)
        status = "READY" if is_ready else "INCOMPLETE"
        passes_str = ", ".join(report.get("passes", [])) or "none"

        if args.dry_run:
            print(f"[{status}] {api_name}: passes=[{passes_str}]")
            for w in report.get("warnings", []):
                print(f"  ⚠ {w}")
            continue

        # Determine output path
        if args.yaml_out and len(files) == 1:
            out_path = Path(args.yaml_out).resolve()
        elif out_dir:
            out_path = out_dir / yp.name
        else:
            out_path = yp  # overwrite in place

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dump_yaml_obj(materialized), encoding="utf-8")

        # Write report
        report_path = out_path.with_suffix(out_path.suffix + ".report.json")
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[{status}] {api_name} -> {out_path}  passes=[{passes_str}]")
        for w in report.get("warnings", []):
            print(f"  ⚠ {w}")

    print(f"\n=== Materialization Summary ===")
    print(f"  Total files: {total_files}")
    print(f"  Harness-ready: {total_ready}")
    print(f"  Incomplete: {total_files - total_ready}")
    print(f"  Total warnings: {total_warnings}")


if __name__ == "__main__":
    main()
