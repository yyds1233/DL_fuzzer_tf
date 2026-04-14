#!/usr/bin/env python3
"""
normalize_yaml_skeleton.py  –  Stage B (normalize): post-process YAML
skeletons to fix up attr defaults, ranges, enum values, and ensure
structural consistency.

Changes vs. original
---------------------
1. Preserves the new `primary_param` and `role` fields added by the
   skeleton builder.
2. Validates that `primary_param` points to an actual tensor param;
   if not, re-detects it.
3. Ensures every tensor param has a `role` field.
4. Minor: generator version bump.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from tf_schema_common import (
    ENUM_TODO_MARKER,
    classify_param_role,
    normalize_rank_hints,
)

GENERATOR_BLOCK = {
    "stage": "B-normalize-tf",
    "version": "2026-03-23-tf-v2",
}


def iter_yaml_files(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(src.glob("*.yaml"))
    return []


def _attr_meta_map(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tf_block = data.get("tf") or {}
    attr_meta = tf_block.get("attr_meta") or []
    return {x["name"]: x for x in attr_meta if isinstance(x, dict) and x.get("name")}


def _allowed_string_values(meta: Dict[str, Any]) -> List[str]:
    allowed = meta.get("allowed_values")
    if isinstance(allowed, dict) and allowed.get("kind") == "list" and isinstance(allowed.get("s"), list):
        return [str(x) for x in allowed.get("s")]
    return []


def _default_list_value(default_value: Any):
    if isinstance(default_value, dict) and default_value.get("kind") == "list":
        for key in ("i", "f", "b", "s", "type"):
            if isinstance(default_value.get(key), list):
                return default_value[key]
    if isinstance(default_value, list):
        return default_value
    return None


def _bump_numeric_range(spec: Dict[str, Any], min_value: int) -> bool:
    changed = False
    rng = spec.get("range")
    if isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, (int, float)) for x in rng):
        if rng[0] < min_value:
            rng[0] = min_value
            changed = True
    return changed


def _ensure_bool_values(spec: Dict[str, Any]) -> bool:
    if spec.get("kind") != "bool":
        return False
    vals = spec.get("values")
    if vals == [True, False] or vals == [False, True]:
        return False
    if "default" not in spec:
        spec["values"] = [True, False]
        return True
    return False


def _normalize_attr_param(name: str, spec: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, int]:
    stats = {
        "default_fixed": 0,
        "range_fixed": 0,
        "kind_fixed": 0,
        "len_fixed": 0,
    }
    attr_type = (meta.get("type") or "").strip()
    default_value = meta.get("default_value") if meta.get("has_default") else None
    minimum = meta.get("minimum") if meta.get("has_minimum") else None
    allowed_strings = _allowed_string_values(meta)

    if attr_type == "bool" and spec.get("kind") != "bool":
        role = spec.get("role", "attr")
        spec.clear()
        spec.update({"origin": "attr", "kind": "bool", "role": role})
        stats["kind_fixed"] += 1
    if attr_type == "int" and spec.get("kind") != "int":
        role = spec.get("role", "attr")
        spec.clear()
        spec.update({"origin": "attr", "kind": "int", "range": [-1, 8], "role": role})
        stats["kind_fixed"] += 1
    if attr_type == "float" and spec.get("kind") != "float":
        role = spec.get("role", "attr")
        spec.clear()
        spec.update({"origin": "attr", "kind": "float", "range": [-1.0, 1.0], "role": role})
        stats["kind_fixed"] += 1
    if attr_type == "string" and spec.get("kind") not in ("enum", "string_optional"):
        role = spec.get("role", "attr")
        spec.clear()
        spec.update({"origin": "attr", "kind": "enum", "values": [ENUM_TODO_MARKER], "role": role})
        stats["kind_fixed"] += 1
    if attr_type == "type" and spec.get("kind") != "dtype_enum":
        role = spec.get("role", "attr")
        spec.clear()
        spec.update({"origin": "attr", "kind": "dtype_enum", "values": [ENUM_TODO_MARKER], "role": role})
        stats["kind_fixed"] += 1
    if attr_type == "list(int)" and spec.get("kind") != "int_list":
        role = spec.get("role", "attr")
        spec.clear()
        spec.update({"origin": "attr", "kind": "int_list", "len_range": [1, 3], "range": [0, 4], "role": role})
        stats["kind_fixed"] += 1

    if minimum is not None and spec.get("kind") == "int":
        if _bump_numeric_range(spec, int(minimum)):
            stats["range_fixed"] += 1

    # low-risk common knobs
    low_name = name.lower()
    if spec.get("kind") == "int" and low_name in ("groups",):
        if _bump_numeric_range(spec, 1):
            stats["range_fixed"] += 1
    if spec.get("kind") == "int_list" and low_name in ("strides", "dilations", "ksize"):
        if _bump_numeric_range(spec, 1):
            stats["range_fixed"] += 1
    if spec.get("kind") == "int" and low_name in ("stride", "dilation", "ksize", "kernel_size"):
        if _bump_numeric_range(spec, 1):
            stats["range_fixed"] += 1
    if spec.get("kind") == "int" and low_name in ("padding", "output_padding"):
        if _bump_numeric_range(spec, 0):
            stats["range_fixed"] += 1
    if spec.get("kind") == "float" and low_name in ("epsilon", "eps"):
        rng = spec.get("range")
        if isinstance(rng, list) and len(rng) == 2 and isinstance(rng[0], (int, float)) and rng[0] <= 0:
            rng[0] = 1e-12
            stats["range_fixed"] += 1

    if allowed_strings and spec.get("kind") in ("enum", "dtype_enum"):
        cur_vals = spec.get("values") or []
        if not isinstance(cur_vals, list):
            cur_vals = []
        merged = list(cur_vals)
        for x in allowed_strings:
            if x not in merged:
                merged.append(x)
        if merged != cur_vals:
            spec["values"] = merged
            stats["kind_fixed"] += 1
    elif spec.get("kind") in ("enum", "dtype_enum"):
        cur_vals = spec.get("values")
        if not isinstance(cur_vals, list) or not cur_vals:
            spec["values"] = [ENUM_TODO_MARKER]
            stats["kind_fixed"] += 1

    if meta.get("has_default"):
        if isinstance(default_value, bool) and spec.get("kind") == "bool":
            if spec.get("default") != default_value:
                spec["default"] = default_value
                stats["default_fixed"] += 1
        elif isinstance(default_value, int) and spec.get("kind") == "int":
            if spec.get("default") != default_value:
                spec["default"] = default_value
                stats["default_fixed"] += 1
        elif isinstance(default_value, float) and spec.get("kind") == "float":
            if spec.get("default") != float(default_value):
                spec["default"] = float(default_value)
                stats["default_fixed"] += 1
        elif isinstance(default_value, str) and spec.get("kind") in ("enum", "dtype_enum"):
            vals = spec.get("values") or []
            if default_value not in vals:
                vals = list(vals) + [default_value]
                spec["values"] = vals
                stats["kind_fixed"] += 1
            if spec.get("default") != default_value:
                spec["default"] = default_value
                stats["default_fixed"] += 1
        elif spec.get("kind") in ("int_list", "float_list", "string_list"):
            default_list = _default_list_value(default_value)
            if isinstance(default_list, list):
                if spec.get("default") != default_list:
                    spec["default"] = default_list
                    stats["default_fixed"] += 1
                L = max(1, len(default_list))
                if spec.get("len_range") != [L, L]:
                    spec["len_range"] = [L, L]
                    stats["len_fixed"] += 1

    if _ensure_bool_values(spec):
        stats["kind_fixed"] += 1

    return stats


def _ensure_param_roles(data: Dict[str, Any]) -> int:
    """Ensure every param has a `role` field. Returns count of fixes."""
    params = data.get("params") or {}
    primary_param = data.get("primary_param")
    fixes = 0
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if "role" not in spec:
            if name == primary_param:
                spec["role"] = "primary"
            elif spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
                role = classify_param_role(name)
                spec["role"] = role if role != "unknown" else "aux"
            else:
                spec["role"] = "attr"
            fixes += 1
    return fixes


def _validate_primary_param(data: Dict[str, Any]) -> bool:
    """Check that primary_param points to a tensor param. Fix if not."""
    params = data.get("params") or {}
    pp = data.get("primary_param")
    if pp and pp in params:
        spec = params[pp]
        if isinstance(spec, dict) and spec.get("kind") in ("tensor", "tensor_optional", "tensor_list"):
            return False  # no fix needed

    # Re-detect
    tensor_names = [
        n for n, s in params.items()
        if isinstance(s, dict) and s.get("kind") in ("tensor", "tensor_optional", "tensor_list")
    ]
    if tensor_names:
        # Prefer primary-role names
        primaries = [n for n in tensor_names if classify_param_role(n) == "primary"]
        data["primary_param"] = primaries[0] if primaries else tensor_names[0]
    else:
        data["primary_param"] = None
    return True


def normalize_one_yaml(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    stats = {
        "files_changed": 0,
        "params_touched": 0,
        "default_fixed": 0,
        "range_fixed": 0,
        "kind_fixed": 0,
        "len_fixed": 0,
        "rank_hints_fixed": 0,
        "generator_fixed": 0,
        "role_fixed": 0,
        "primary_fixed": 0,
    }
    before = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    data["rank_hints"] = normalize_rank_hints(data.get("rank_hints"))
    stats["rank_hints_fixed"] += 1

    if data.get("generator") != GENERATOR_BLOCK:
        data["generator"] = GENERATOR_BLOCK
        stats["generator_fixed"] += 1

    # Ensure roles
    role_fixes = _ensure_param_roles(data)
    stats["role_fixed"] = role_fixes

    # Validate primary_param
    if _validate_primary_param(data):
        stats["primary_fixed"] = 1

    params = data.get("params") or {}
    attr_meta_map = _attr_meta_map(data)

    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        touched = False
        if spec.get("origin") == "attr" and name in attr_meta_map:
            sub = _normalize_attr_param(name, spec, attr_meta_map[name])
            for k, v in sub.items():
                if k in stats:
                    stats[k] += v
                if v:
                    touched = True
        if spec.get("kind") in ("tensor", "tensor_list"):
            if "shape_spec" not in spec:
                spec["shape_spec"] = ["TODO_SHAPE"]
                touched = True
        if touched:
            stats["params_touched"] += 1

    after = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if after != before:
        stats["files_changed"] = 1
    return data, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True,
                    help="single yaml file or a directory of yamls")
    ap.add_argument("--out_dir", default=None,
                    help="write normalized yamls to a new directory; default overwrites in place")
    ap.add_argument("--dry_run", action="store_true",
                    help="report changes without writing files")
    args = ap.parse_args()

    src = Path(args.yaml).resolve()
    files = iter_yaml_files(src)
    if not files:
        raise SystemExit(f"No yaml files found under: {src}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = {
        "files": 0,
        "files_changed": 0,
        "params_touched": 0,
        "default_fixed": 0,
        "range_fixed": 0,
        "kind_fixed": 0,
        "len_fixed": 0,
        "rank_hints_fixed": 0,
        "generator_fixed": 0,
        "role_fixed": 0,
        "primary_fixed": 0,
    }

    for yp in files:
        raw = yp.read_text(encoding="utf-8", errors="ignore")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            print(f"[!] skip non-dict yaml: {yp}")
            continue
        new_data, stats = normalize_one_yaml(data)
        total["files"] += 1
        for k in total.keys():
            if k == "files":
                continue
            if k in stats:
                total[k] += stats[k]
        if args.dry_run:
            print(
                f"[DRY] {yp.name}: changed={bool(stats['files_changed'])}, "
                f"touched={stats['params_touched']}, default_fixed={stats['default_fixed']}, "
                f"range_fixed={stats['range_fixed']}, kind_fixed={stats['kind_fixed']}, "
                f"len_fixed={stats['len_fixed']}, role_fixed={stats['role_fixed']}, "
                f"primary_fixed={stats['primary_fixed']}"
            )
            continue
        out_path = (out_dir / yp.name) if out_dir is not None else yp
        out_path.write_text(yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if stats["files_changed"]:
            print(f"[+] normalized: {yp} -> {out_path}")
        else:
            print(f"[=] unchanged: {yp} -> {out_path}")

    print("\n=== Summary ===")
    for k, v in total.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
