#!/usr/bin/env python3
"""
extract_tf_doc_hints.py  –  Stage A (doc): parameter-aware rank extraction
from TF documentation text files.

==========================================================================
KEY DESIGN CHANGE vs. original
==========================================================================

The original extractor did flat regex scanning across the entire document,
treating every "N-D" / "rank N" mention equally.  This caused rank
pollution — e.g. Conv2D would yield rank_candidates=[1,2,4] because:
  - `input` is 4-D  → 4 ✓  (primary tensor)
  - `filter` is 4-D  → 4   (aux tensor, same rank, harmless)
  - `strides` is 1-D → 1 ✗  (attr list, NOT a tensor rank)
  - `Flattens … to 2-D` → 2 ✗  (intermediate derivation)

The new extractor works in three phases:

Phase 1 – **Paragraph segmentation**
    Split the doc text into paragraphs.  For each paragraph, try to
    identify which *parameter* it is describing (by looking for
    "`param_name`:" or "param_name:" at the start of the paragraph,
    or a header-like pattern).

Phase 2 – **Parameter-scoped rank collection**
    Within each paragraph that is associated with a known parameter,
    collect rank mentions.  Tag each mention with the parameter's role
    (primary / aux / attr / unknown) using `classify_param_role()`.

Phase 3 – **API-level rank_hints synthesis**
    - Keep only ranks from "primary" parameters (and "unknown" params
      that look tensor-ish from the schema's input_args).
    - If *no* primary-rank info is found, fall back to the first
      input_arg's paragraph as a secondary heuristic.
    - Auxiliary tensor ranks are stored separately in `aux_rank_info`
      for possible future use, but are NOT mixed into `rank_candidates`.

This gives Conv2D → rank_candidates=[4] (from `input`),
     BiasAdd → rank_candidates=[] + rank_any=True (from `value`),
     MatMul  → rank_candidates=[2] (from `a`).

==========================================================================
Schema-awareness (optional but recommended)
==========================================================================

If --schema_dir is provided, the extractor loads the Stage-A schema
JSON for each API to get the list of input_arg names.  This makes
paragraph↔parameter matching much more reliable than pure heuristic.

Without --schema_dir, the extractor falls back to a built-in name list
(PRIMARY_INPUT_NAMES ∪ AUX_INPUT_NAMES ∪ ATTR_LIKE_NAMES).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tf_schema_common import (
    dump_json,
    load_api_list,
    load_json,
    safe_name,
    classify_param_role,
    PRIMARY_INPUT_NAMES,
    AUX_INPUT_NAMES,
    ATTR_LIKE_NAMES,
)

GENERATOR_BLOCK = {
    "stage": "A-doc-rank-extract-tf",
    "version": "2026-03-23-tf-v2",
}

# ── regex patterns ───────────────────────────────────────────────

# Matches "4-D", "4D", "1-D" etc.  Captures the digit.
RE_ND = re.compile(r"\b([1-9])\s*-?\s*D\b", re.IGNORECASE)

# Matches "rank 4", "rank=4", "rank: 4"
RE_RANK_EQ = re.compile(r"\brank\s*[:=]?\s*([1-9])\b", re.IGNORECASE)

# Matches "rank 2 or higher", "at least 2-D"
RE_RANK_OR_HIGHER = [
    re.compile(r"\brank\s*([1-9])\s*or\s*higher\b", re.IGNORECASE),
    re.compile(r"\b([1-9])\s*-?\s*D[^\n.]{0,60}\bor\s+higher\b", re.IGNORECASE),
    re.compile(r"\bat\s+least\s+([1-9])\s*-?\s*D\b", re.IGNORECASE),
]

# Matches "any number of dimensions", "arbitrary rank", "N-D"
RE_ANY_RANK = [
    re.compile(r"\bany\s+number\s+of\s+dimensions\b", re.IGNORECASE),
    re.compile(r"\barbitrary\s+rank\b", re.IGNORECASE),
    re.compile(r"\bN-?D\b"),  # case-sensitive: "N-D" not "n-d"
    re.compile(r"\bany\s+rank\b", re.IGNORECASE),
    re.compile(r"\bany\s+shape\b", re.IGNORECASE),
]

# Patterns for things we want to *exclude* — intermediate derivations,
# output descriptions, formula lines.
RE_EXCLUDE_CONTEXT = [
    # "Flattens … to a 2-D matrix" — intermediate derivation
    re.compile(r"\b(flatten|reshape|broadcast|expand|squeeze)\w*\b.*\b\d-?\s*D\b", re.IGNORECASE),
    # "produces a N-D output" — output, not input
    re.compile(r"\b(output|result|return)\w*\b.*\b\d-?\s*D\b", re.IGNORECASE),
    # Lines that are clearly formulae
    re.compile(r"^\s*output\[", re.IGNORECASE),
]

# Pattern to detect the start of a parameter description paragraph.
# Matches things like:
#   "input: A 4-D tensor"
#   "`filter`: A `Tensor`."
#   "value:  A Tensor …"
RE_PARAM_HEADER = re.compile(
    r"^[`'\"]?([A-Za-z_][A-Za-z0-9_]*)[`'\"]?\s*:\s*",
    re.MULTILINE,
)

# Pattern to detect lines that should be skipped entirely
# (example code, formula, internal comment).
RE_SKIP_LINE = [
    re.compile(r"^\s*>>>"),       # doctest / example
    re.compile(r"^\s*#"),         # comment
    re.compile(r"^\s*\|"),        # table row
    re.compile(r"^\s*```"),       # code fence
    re.compile(r"^\s*\.\.\s"),    # rst continuation
]


# ── paragraph-level helpers ──────────────────────────────────────

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (blank-line separated)."""
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]


def _is_skip_line(line: str) -> bool:
    return any(pat.search(line) for pat in RE_SKIP_LINE)


def _is_excluded_context(line: str) -> bool:
    return any(pat.search(line) for pat in RE_EXCLUDE_CONTEXT)


def _extract_ranks_from_text(text: str) -> Tuple[List[int], Optional[int], bool]:
    """
    Extract rank info from a chunk of text.
    Returns (fixed_ranks, rank_min_from_or_higher, is_any_rank).
    """
    fixed: List[int] = []
    rank_min: Optional[int] = None
    rank_any = False

    for line in text.splitlines():
        if _is_skip_line(line):
            continue
        if _is_excluded_context(line):
            continue

        for m in RE_ND.finditer(line):
            try:
                fixed.append(int(m.group(1)))
            except ValueError:
                pass
        for m in RE_RANK_EQ.finditer(line):
            try:
                fixed.append(int(m.group(1)))
            except ValueError:
                pass

        for pat in RE_RANK_OR_HIGHER:
            for m in pat.finditer(line):
                try:
                    val = int(m.group(1))
                    if rank_min is None or val < rank_min:
                        rank_min = val
                except ValueError:
                    pass

        if not rank_any:
            for pat in RE_ANY_RANK:
                if pat.search(line):
                    rank_any = True
                    break

    return sorted(set(fixed)), rank_min, rank_any


def _identify_paragraph_param(paragraph: str, known_params: Set[str]) -> Optional[str]:
    """
    Try to figure out which parameter a paragraph is describing.
    Returns the param name or None.
    """
    m = RE_PARAM_HEADER.match(paragraph)
    if m:
        candidate = m.group(1)
        # Check exact match first
        if candidate in known_params:
            return candidate
        # Case-insensitive fallback
        for kp in known_params:
            if kp.lower() == candidate.lower():
                return kp
    return None


# ── schema-aware param list ──────────────────────────────────────

def _load_input_arg_names(api_name: str, schema_dir: Optional[Path]) -> List[str]:
    """Load input_arg names from Stage-A schema JSON if available."""
    if schema_dir is None:
        return []
    p = schema_dir / f"{api_name.replace('.', '_')}_schema.json"
    if not p.exists():
        return []
    try:
        schema = load_json(p)
        tf_block = schema.get("tf") or {}
        op_def = tf_block.get("op_def") or {}
        input_args = op_def.get("input_args") or []
        return [a["name"] for a in input_args if isinstance(a, dict) and a.get("name")]
    except Exception:
        return []


def _load_attr_names(api_name: str, schema_dir: Optional[Path]) -> List[str]:
    """Load attr names from Stage-A schema JSON if available."""
    if schema_dir is None:
        return []
    p = schema_dir / f"{api_name.replace('.', '_')}_schema.json"
    if not p.exists():
        return []
    try:
        schema = load_json(p)
        tf_block = schema.get("tf") or {}
        op_def = tf_block.get("op_def") or {}
        attrs = op_def.get("attrs") or []
        return [a["name"] for a in attrs if isinstance(a, dict) and a.get("name")]
    except Exception:
        return []


# ── core extraction logic ────────────────────────────────────────

def extract_doc_rank_hints(
    api_name: str,
    doc_text: str,
    schema_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Parameter-aware rank extraction from documentation text.

    Returns a dict suitable for writing to *.rank.json.
    """
    if not doc_text.strip():
        return {
            "generator": GENERATOR_BLOCK,
            "api_name": api_name,
            "rank_candidates": [],
            "rank_any": False,
            "rank_min": None,
            "rank_max": None,
            "marker": "__RANK_FROM_DOC__",
            "param_rank_details": {},
            "error": "empty doc text",
        }

    # ── build the set of known parameter names ───────────────────
    input_arg_names = _load_input_arg_names(api_name, schema_dir)
    attr_names = _load_attr_names(api_name, schema_dir)

    # All params we might encounter in the doc
    all_known: Set[str] = set()
    all_known.update(input_arg_names)
    all_known.update(attr_names)
    all_known.update(PRIMARY_INPUT_NAMES)
    all_known.update(AUX_INPUT_NAMES)
    all_known.update(ATTR_LIKE_NAMES)

    # Determine which input_args are primary vs aux
    # If we have schema info, use input_arg position: first input = primary
    primary_params: Set[str] = set()
    aux_params: Set[str] = set()
    attr_params: Set[str] = set(attr_names)

    if input_arg_names:
        for i, name in enumerate(input_arg_names):
            role = classify_param_role(name)
            if role == "primary":
                primary_params.add(name)
            elif role == "aux":
                aux_params.add(name)
            elif role == "attr":
                attr_params.add(name)
            else:
                # Unknown role: if it's the first input_arg and nothing
                # else is primary, treat it as primary.
                if i == 0 and not primary_params:
                    primary_params.add(name)
                elif name.lower() not in {n.lower() for n in ATTR_LIKE_NAMES}:
                    # Looks tensor-ish, but we don't know its role.
                    # Default to primary if we still have no primary.
                    if not primary_params:
                        primary_params.add(name)
                    else:
                        aux_params.add(name)
        # If after all that we still have no primary, use the first input_arg.
        if not primary_params and input_arg_names:
            primary_params.add(input_arg_names[0])
    else:
        # No schema — rely on name heuristics only
        primary_params = {n for n in PRIMARY_INPUT_NAMES}
        aux_params = {n for n in AUX_INPUT_NAMES}
        attr_params = {n for n in ATTR_LIKE_NAMES}

    # ── Phase 1: segment paragraphs and tag with param ───────────
    paragraphs = _split_paragraphs(doc_text)
    # Each entry: (param_name_or_None, paragraph_text)
    tagged: List[Tuple[Optional[str], str]] = []
    for para in paragraphs:
        pname = _identify_paragraph_param(para, all_known)
        tagged.append((pname, para))

    # ── Phase 2: extract ranks per paragraph, group by role ──────
    # Collect rank info keyed by param name → {fixed, rank_min, rank_any}
    param_ranks: Dict[str, Dict[str, Any]] = {}
    untagged_ranks: List[int] = []
    untagged_rank_min: Optional[int] = None
    untagged_rank_any = False

    for pname, para in tagged:
        fixed, rmin, rany = _extract_ranks_from_text(para)
        if not fixed and rmin is None and not rany:
            continue  # no rank info in this paragraph

        if pname is not None:
            if pname not in param_ranks:
                param_ranks[pname] = {"fixed": [], "rank_min": None, "rank_any": False}
            entry = param_ranks[pname]
            entry["fixed"] = sorted(set(entry["fixed"] + fixed))
            if rmin is not None:
                entry["rank_min"] = min(rmin, entry["rank_min"]) if entry["rank_min"] is not None else rmin
            if rany:
                entry["rank_any"] = True
        else:
            # Untagged paragraph — only use as last-resort fallback
            untagged_ranks.extend(fixed)
            if rmin is not None:
                untagged_rank_min = min(rmin, untagged_rank_min) if untagged_rank_min is not None else rmin
            if rany:
                untagged_rank_any = True

    # ── Phase 3: synthesize API-level rank_hints ─────────────────
    primary_fixed: List[int] = []
    primary_rank_min: Optional[int] = None
    primary_rank_any = False

    aux_rank_info: Dict[str, Any] = {}

    for pname, info in param_ranks.items():
        pname_low = pname.lower()
        # Determine role
        is_primary = pname in primary_params or pname_low in {n.lower() for n in primary_params}
        is_aux = pname in aux_params or pname_low in {n.lower() for n in aux_params}
        is_attr = pname in attr_params or pname_low in {n.lower() for n in attr_params}

        if is_attr:
            # Skip attr params entirely — their "1-D" is list length, not tensor rank
            continue

        if is_primary:
            primary_fixed.extend(info["fixed"])
            if info["rank_min"] is not None:
                primary_rank_min = (
                    min(info["rank_min"], primary_rank_min)
                    if primary_rank_min is not None
                    else info["rank_min"]
                )
            if info["rank_any"]:
                primary_rank_any = True
        elif is_aux:
            aux_rank_info[pname] = info
        else:
            # Unknown param with rank info — if it's in input_arg_names,
            # treat it as primary (conservative).
            if pname in set(input_arg_names) and pname not in attr_params:
                primary_fixed.extend(info["fixed"])
                if info["rank_min"] is not None:
                    primary_rank_min = (
                        min(info["rank_min"], primary_rank_min)
                        if primary_rank_min is not None
                        else info["rank_min"]
                    )
                if info["rank_any"]:
                    primary_rank_any = True
            else:
                aux_rank_info[pname] = info

    # ── Fallback: if no primary info found, use untagged paragraphs ──
    if not primary_fixed and primary_rank_min is None and not primary_rank_any:
        # Use untagged paragraph ranks as weak fallback, but filter
        # out the very small ranks (1) which are likely attr/list lengths.
        fallback = [r for r in untagged_ranks if r >= 2]
        if fallback:
            primary_fixed = fallback
        primary_rank_min = untagged_rank_min
        primary_rank_any = untagged_rank_any

    rank_candidates = sorted(set(primary_fixed))

    # Compute rank_max from candidates
    rank_max: Optional[int] = None
    if rank_candidates:
        rank_max = max(rank_candidates)
    if primary_rank_min is not None and rank_max is None:
        rank_max = None  # unbounded above

    # Build per-param detail block for transparency / debugging
    param_rank_details: Dict[str, Any] = {}
    for pname, info in param_ranks.items():
        pname_low = pname.lower()
        is_primary = pname in primary_params or pname_low in {n.lower() for n in primary_params}
        is_aux = pname in aux_params or pname_low in {n.lower() for n in aux_params}
        is_attr = pname in attr_params or pname_low in {n.lower() for n in attr_params}
        role = "primary" if is_primary else ("aux" if is_aux else ("attr" if is_attr else "unknown"))
        param_rank_details[pname] = {
            "role": role,
            "fixed_ranks": info["fixed"],
            "rank_min": info["rank_min"],
            "rank_any": info["rank_any"],
            "included_in_api_rank": role == "primary" or (
                role == "unknown" and pname in set(input_arg_names)
            ),
        }

    return {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "rank_candidates": rank_candidates,
        "rank_any": bool(primary_rank_any),
        "rank_min": primary_rank_min,
        "rank_max": rank_max,
        "marker": "__RANK_FROM_DOC__",
        "status": "assigned" if (rank_candidates or primary_rank_any) else "unassigned",
        "primary_params_detected": sorted(primary_params),
        "aux_rank_info": {k: v for k, v in aux_rank_info.items()},
        "param_rank_details": param_rank_details,
    }


# ── file I/O helpers ─────────────────────────────────────────────

def read_texts(paths: List[Path]) -> str:
    chunks = []
    for p in paths:
        if not p.exists():
            continue
        chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(chunks)


def load_mapping_json(path: Path) -> Dict[str, List[Path]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("mapping json must be an object: {api_name: [txt1, txt2, ...]}")
    out: Dict[str, List[Path]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[str(k)] = [Path(v)]
        elif isinstance(v, list):
            out[str(k)] = [Path(x) for x in v]
    return out


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parameter-aware rank extraction from TF doc text files."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mapping_json",
                    help="json object: {api_name: [txt1, txt2, ...]}")
    g.add_argument("--api_list",
                    help="txt/json/pkl API list; requires --doc_dir")
    ap.add_argument("--doc_dir",
                    help="directory containing per-api txt docs named by safe_name(api)+.txt")
    ap.add_argument("--schema_dir", default=None,
                    help="directory containing *_schema.json from export_tf_schema.py "
                         "(enables schema-aware param classification)")
    ap.add_argument("--out_dir", default="./tf_rank_hints",
                    help="where to write *.rank.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_dir = Path(args.schema_dir).resolve() if args.schema_dir else None

    mapping: Dict[str, List[Path]] = {}
    if args.mapping_json:
        mapping = load_mapping_json(Path(args.mapping_json))
    else:
        if not args.doc_dir:
            raise SystemExit("--doc_dir is required with --api_list")
        api_list = load_api_list(args.api_list)
        doc_dir = Path(args.doc_dir)
        for api in api_list:
            p = doc_dir / f"{safe_name(api)}.txt"
            mapping[api] = [p]

    count = 0
    for api_name, paths in mapping.items():
        text = read_texts(paths)
        out = extract_doc_rank_hints(api_name, text, schema_dir)
        out_path = out_dir / f"{safe_name(api_name)}.rank.json"
        dump_json(out_path, out)
        rc = out.get("rank_candidates", [])
        status = out.get("status", "?")
        print(f"[+] {api_name} -> {out_path}  ranks={rc} status={status}")
        count += 1

    print(f"[done] wrote {count} rank hint files")


if __name__ == "__main__":
    main()
