#!/usr/bin/env python3
"""
tf_patch_constraints.py  –  Stage D: LLM-driven constraints patch for
TF raw_ops YAML.

==========================================================================
Changes vs. original
==========================================================================
1. Supports per_rank_constraints — constraints that only apply at specific
   ranks (e.g., layout-dependent constraints at rank 4).
2. Better validation using the now-complete YAML (shape_vars are populated).
3. Constraint merging writes to both top-level and per-rank locations.
4. Layout-conditional constraints are encouraged and properly handled.
"""
from __future__ import annotations

import os
import re
import argparse
import json
import traceback
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

try:
    from openai import OpenAI, BadRequestError
except ImportError:
    OpenAI = None  # type: ignore
    BadRequestError = Exception  # type: ignore

from tf_llm_prompts import TF_YAML_CONSTRAINT_SYSTEM_PROMPT


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


# ════════════════════════════════════════════════════════════════
# 2) JSON parsing
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
    return json.loads(extract_json_object(raw_text))


def normalize_constraint_patch(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["constraints_append"] = p.get("constraints_append") or []
    out["constraints_remove"] = p.get("constraints_remove") or []
    out["per_rank_constraints"] = p.get("per_rank_constraints") or {}
    out["changes"] = p.get("changes") or []
    out["warnings"] = p.get("warnings") or []
    out["confidence"] = p.get("confidence", 0.5)

    if not isinstance(out["constraints_append"], list):
        out["constraints_append"] = []
    if not isinstance(out["constraints_remove"], list):
        out["constraints_remove"] = []
    if not isinstance(out["per_rank_constraints"], dict):
        out["per_rank_constraints"] = {}
    try:
        out["confidence"] = float(out["confidence"])
    except Exception:
        out["confidence"] = 0.5

    def _clean_list(lst: List[Any]) -> List[str]:
        cleaned: List[str] = []
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

    out["constraints_append"] = _clean_list(out["constraints_append"])
    out["constraints_remove"] = _clean_list(out["constraints_remove"])

    # Clean per_rank_constraints
    cleaned_prc: Dict[str, List[str]] = {}
    for rank_key, clist in out["per_rank_constraints"].items():
        if isinstance(clist, list):
            cleaned_prc[str(rank_key)] = _clean_list(clist)
    out["per_rank_constraints"] = cleaned_prc

    return out


# ════════════════════════════════════════════════════════════════
# 3) validation
# ════════════════════════════════════════════════════════════════

_ALLOWED_BUILTINS = {"isinstance", "all", "any", "len", "tuple", "min", "max", "abs"}


def validate_constraints_eval_safety(constraints: List[Any]) -> List[str]:
    errs: List[str] = []
    for i, c in enumerate(constraints):
        if not isinstance(c, str):
            errs.append(f"constraints[{i}] must be a string")
            continue
        if ";" in c or "\n" in c:
            errs.append(f"constraints[{i}] contains ';' or newline: {c!r}")
        if "import " in c or "lambda" in c or "def " in c or "class " in c:
            errs.append(f"constraints[{i}] contains forbidden keyword: {c!r}")
    return errs


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


def validate_constraint_names_defined(
    constraints: List[str], allowed_names: Set[str]
) -> List[str]:
    errs: List[str] = []
    for i, c in enumerate(constraints):
        names = extract_names_ast(c)
        names = {n for n in names if n not in _ALLOWED_BUILTINS}
        unknown = sorted([n for n in names if n not in allowed_names])
        if unknown:
            errs.append(f"constraints[{i}] references undefined names: {unknown} | expr={c!r}")
    return errs


def build_tf_allowed_names(yaml_obj: Dict[str, Any]) -> Set[str]:
    allowed = set(_ALLOWED_BUILTINS)
    params = yaml_obj.get("params")
    if isinstance(params, dict):
        allowed |= set(params.keys())
    shape_vars = yaml_obj.get("shape_vars")
    if isinstance(shape_vars, dict):
        allowed |= set(shape_vars.keys())
    return allowed


def validate_patch(patch: Dict[str, Any], allowed_names: Set[str]) -> List[str]:
    errs: List[str] = []

    # Validate top-level constraints
    errs += validate_constraints_eval_safety(patch["constraints_append"])
    errs += validate_constraints_eval_safety(patch["constraints_remove"])
    errs += validate_constraint_names_defined(patch["constraints_append"], allowed_names)

    # Validate per-rank constraints
    for rank_key, clist in (patch.get("per_rank_constraints") or {}).items():
        prefix = f"per_rank_constraints[{rank_key}]"
        sub_errs = validate_constraints_eval_safety(clist)
        errs += [f"{prefix}.{e}" for e in sub_errs]
        sub_errs = validate_constraint_names_defined(clist, allowed_names)
        errs += [f"{prefix}.{e}" for e in sub_errs]

    return errs


# ════════════════════════════════════════════════════════════════
# 4) apply patch
# ════════════════════════════════════════════════════════════════

def merge_constraints(
    existing: List[Any],
    to_remove: List[str],
    to_append: List[str],
) -> List[str]:
    old: List[str] = []
    for x in existing or []:
        if isinstance(x, str) and x.strip():
            old.append(x.strip())

    remove_set = {s.strip() for s in to_remove if s.strip()}
    kept = [c for c in old if c not in remove_set]

    seen = set(kept)
    for c in to_append:
        c = c.strip()
        if c and c not in seen:
            kept.append(c)
            seen.add(c)
    return kept


def apply_constraint_patch(
    yaml_obj: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply constraint patch to YAML, including per-rank constraints."""
    out = dict(yaml_obj)

    # Top-level constraints
    old_constraints = out.get("constraints") or []
    out["constraints"] = merge_constraints(
        existing=old_constraints,
        to_remove=patch.get("constraints_remove") or [],
        to_append=patch.get("constraints_append") or [],
    )

    # Per-rank constraints → store under primary_param.constraints_by_rank
    per_rank = patch.get("per_rank_constraints") or {}
    if per_rank:
        primary_param = out.get("primary_param")
        params = out.get("params") or {}
        if primary_param and primary_param in params:
            p = params[primary_param]
            if isinstance(p, dict):
                existing_cbr = p.get("constraints_by_rank") or {}
                if not isinstance(existing_cbr, dict):
                    existing_cbr = {}

                for rank_key, clist in per_rank.items():
                    rank_key = str(rank_key)
                    old = existing_cbr.get(rank_key) or []
                    merged = merge_constraints(old, [], clist)
                    existing_cbr[rank_key] = merged

                p["constraints_by_rank"] = existing_cbr

    return out


# ════════════════════════════════════════════════════════════════
# 5) LLM call
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
# 6) main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage D: LLM constraint patch for TF raw_ops YAML."
    )
    ap.add_argument("--doc_txt", required=True)
    ap.add_argument("--yaml_in", required=True)
    ap.add_argument("--yaml_out", required=True)
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--fail_on_invalid", action="store_true")
    args = ap.parse_args()

    doc_path = Path(args.doc_txt).resolve()
    yaml_in_path = Path(args.yaml_in).resolve()
    yaml_out_path = Path(args.yaml_out).resolve()
    meta_out_path = yaml_out_path.with_suffix(yaml_out_path.suffix + ".meta.json")
    yaml_out_path.parent.mkdir(parents=True, exist_ok=True)

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
    allowed_names = build_tf_allowed_names(yaml_obj)

    system_prompt = TF_YAML_CONSTRAINT_SYSTEM_PROMPT
    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== INPUT YAML ===\n"
        f"{yaml_text}\n\n"
        "Return ONLY a JSON object patch.\n"
        "Stage D: constraints ONLY. Minimal, high-confidence.\n"
        f"test_ranks from YAML: {yaml_obj.get('test_ranks', [])}\n"
        f"primary_param: {yaml_obj.get('primary_param')}\n"
    )

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
            traceback.print_exc()
            raise

        try:
            patch_dict = parse_patch(last_raw)
        except Exception as e:
            last_errors = [f"JSON parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = (
                    base_user_prompt
                    + "\nYour previous output was NOT valid JSON. Output ONLY JSON.\n"
                )
                continue
            final_patch = {
                "constraints_append": [],
                "constraints_remove": [],
                "per_rank_constraints": {},
                "changes": [],
                "warnings": [f"Failed to parse JSON: {e}"],
                "confidence": 0.0,
            }
            break

        patch = normalize_constraint_patch(patch_dict)
        last_errors = validate_patch(patch, allowed_names)

        if not last_errors:
            final_patch = patch
            break

        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\nYour previous JSON FAILED validation. Fix and output ONLY JSON.\n"
                + "Errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n\nPrevious JSON:\n"
                + (extract_json_object(last_raw) if last_raw else "")
                + "\n"
            )
        else:
            final_patch = patch

    if final_patch is None:
        final_patch = {
            "constraints_append": [],
            "constraints_remove": [],
            "per_rank_constraints": {},
            "changes": [],
            "warnings": ["LLM completely failed"],
            "confidence": 0.0,
        }

    # Apply
    out_yaml = apply_constraint_patch(yaml_obj, final_patch)

    yaml_out_path.write_text(dump_yaml_obj(out_yaml), encoding="utf-8")

    meta = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(yaml_out_path),
        "confidence": float(final_patch.get("confidence", 0.5)),
        "constraints_added": len(final_patch.get("constraints_append", [])),
        "per_rank_constraints_added": {
            k: len(v) for k, v in (final_patch.get("per_rank_constraints") or {}).items()
        },
        "changes": final_patch.get("changes", []),
        "warnings": final_patch.get("warnings", []),
        "validation_errors": last_errors,
        "raw_model_output_snippet": (last_raw[:8000] if last_raw else ""),
    }
    meta_out_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[+] wrote Stage-D yaml: {yaml_out_path}")
    if last_errors:
        msg = "[!] Stage-D validation errors:\n" + "\n".join(f"   - {e}" for e in last_errors)
        if args.fail_on_invalid:
            raise SystemExit(msg)
        print(msg)


if __name__ == "__main__":
    main()
