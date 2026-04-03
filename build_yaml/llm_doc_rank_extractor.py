#!/usr/bin/env python3
"""
llm_doc_rank_extractor.py  –  Stage A (doc): LLM-assisted parameter-aware
rank extraction from TF documentation text files.

==========================================================================
WHY THIS EXISTS
==========================================================================

The original `extract_tf_doc_hints.py` uses regex patterns to extract rank
information from documentation.  This works for simple cases but fails when:

  - Rank mentions in one paragraph refer to *different* parameters
  - Intermediate derivations ("Flattens to a 2-D matrix") are confused
    with input tensor rank
  - Attribute lists ("strides: 1-D tensor of length 4") pollute rank info
  - Docs use indirect language ("any number of dimensions")

The LLM-assisted extractor solves this by:

  1. Sending the doc text + schema info to an LLM with a structured prompt
  2. Asking the LLM to identify, for each parameter, its semantic role and
     rank information
  3. Parsing the structured JSON response into the same output format as
     the regex extractor
  4. Falling back to the regex extractor if the LLM call fails

==========================================================================
USAGE
==========================================================================

  # With OpenAI-compatible API (works with OpenAI, vLLM, Ollama, etc.)
  python llm_doc_rank_extractor.py \
      --api_list apis.txt \
      --doc_dir ./tf_docs \
      --schema_dir ./tf_api_schema \
      --out_dir ./tf_rank_hints \
      --llm_base_url http://localhost:11434/v1 \
      --llm_model qwen2.5:72b \
      --llm_api_key none

  # With OpenAI API
  python llm_doc_rank_extractor.py \
      --api_list apis.txt \
      --doc_dir ./tf_docs \
      --schema_dir ./tf_api_schema \
      --out_dir ./tf_rank_hints \
      --llm_base_url https://api.openai.com/v1 \
      --llm_model gpt-4o \
      --llm_api_key sk-xxx

  # Fallback-only mode (no LLM, same as original regex extractor)
  python llm_doc_rank_extractor.py \
      --api_list apis.txt \
      --doc_dir ./tf_docs \
      --schema_dir ./tf_api_schema \
      --out_dir ./tf_rank_hints \
      --no_llm

==========================================================================
OUTPUT FORMAT
==========================================================================

Same as extract_tf_doc_hints.py: *.rank.json files with:
  - rank_candidates: list of int (primary input tensor ranks)
  - rank_any: bool
  - rank_min: int or null
  - rank_max: int or null
  - param_rank_details: per-param breakdown
  - llm_raw_response: (new) the raw LLM response for debugging
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tf_schema_common import (
    dump_json,
    load_api_list,
    load_json,
    safe_name,
    classify_param_role,
    classify_param_semantic_role,
    PRIMARY_INPUT_NAMES,
    AUX_INPUT_NAMES,
    ATTR_LIKE_NAMES,
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
)

GENERATOR_BLOCK = {
    "stage": "A-doc-rank-extract-llm-tf",
    "version": "2026-03-23-tf-v3",
}

# ── LLM client ───────────────────────────────────────────────────

class LLMClient:
    """
    Simple OpenAI-compatible chat completion client.
    Works with: OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio,
    DeepSeek, Together AI, Groq, etc.
    """

    def __init__(
        self,
        base_url: str = "https://api.gpt.ge/v1",
        model: str = "gpt-5-codex",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Send a chat completion request. Returns content string or None."""
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")

        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    choices = body.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        return msg.get("content")
                    return None
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.max_retries:
                    wait = 2 ** (attempt + 1)
                    print(f"  [LLM] rate-limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"  [LLM] HTTP {e.code}: {e.reason}")
                return None
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                print(f"  [LLM] error: {e}")
                return None
        return None


# ── prompt construction ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a TensorFlow API documentation analyst. Your task is to extract \
precise rank (number of dimensions) and shape information for each parameter \
of a TF operation from its documentation text.

CRITICAL RULES:
1. "Rank" means the number of dimensions of a TENSOR (e.g., a 4-D tensor has rank 4).
2. Distinguish between:
   - PRIMARY data tensor (the main input being transformed, e.g., `input`, `x`, `value`)
   - WEIGHT/BIAS tensor (learned parameters, e.g., `filter`, `bias`)
   - AUXILIARY tensor (secondary inputs like `mean`, `variance`, `scale`)
   - SHAPE-CONTROL input (e.g., `shape` in Reshape — an int vector that specifies output shape)
   - INDEX input (e.g., `indices`, `perm` — integer tensors for indexing/permutation)
   - ATTRIBUTE (non-tensor params like `strides`, `padding`, `data_format`, `axis`)
3. When docs say "1-D tensor of length 4" for `strides`, that means strides is an \
   attribute-like int list, NOT a rank-1 data tensor.
4. When docs say "Flattens to a 2-D matrix", that's an INTERMEDIATE derivation, \
   NOT the input rank.
5. "Any number of dimensions" / "N-D" / "any rank" means the param accepts arbitrary rank.
6. For the API-level "primary_rank", ONLY use the rank of the PRIMARY data tensor(s).

Respond ONLY with a JSON object (no markdown fences, no explanation)."""

def build_user_prompt(
    api_name: str,
    doc_text: str,
    schema_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user prompt with doc text and optional schema context."""
    parts = []
    parts.append(f"API: {api_name}\n")

    # Include schema context if available
    if schema_info:
        tf_block = schema_info.get("tf") or {}
        op_def = tf_block.get("op_def") or {}
        input_args = op_def.get("input_args") or []
        attrs = op_def.get("attrs") or []

        if input_args:
            parts.append("Schema input_args:")
            for arg in input_args:
                name = arg.get("name", "?")
                type_name = arg.get("type_name") or arg.get("type_attr") or "?"
                desc = arg.get("description") or ""
                parts.append(f"  - {name}: type={type_name} desc=\"{desc[:100]}\"")

        if attrs:
            parts.append("Schema attrs:")
            for attr in attrs:
                name = attr.get("name", "?")
                atype = attr.get("type", "?")
                parts.append(f"  - {name}: type={atype}")
        parts.append("")

    parts.append("Documentation text:")
    parts.append("---")
    # Truncate very long docs to avoid token limits
    truncated = doc_text[:6000] if len(doc_text) > 6000 else doc_text
    parts.append(truncated)
    parts.append("---")
    parts.append("")

    parts.append("""\
Analyze the documentation and return a JSON object with this EXACT structure:
{
  "api_name": "<api_name>",
  "primary_rank": {
    "fixed_ranks": [<list of ints, empty if not fixed>],
    "rank_any": <true if accepts any rank, false otherwise>,
    "rank_min": <int or null, minimum rank if "rank N or higher">,
    "rank_max": <int or null>
  },
  "params": {
    "<param_name>": {
      "semantic_role": "<one of: data_tensor, weight_tensor, aux_tensor, index_input, shape_control, fixed_arity_list, layout_attr, scalar_attr, dtype_attr, meta>",
      "rank": <int or null, the tensor rank if applicable>,
      "rank_any": <true/false>,
      "rank_min": <int or null>,
      "description_summary": "<1-sentence summary of what this param does>"
    },
    ...
  },
  "reasoning": "<brief explanation of how you determined the primary rank>"
}

IMPORTANT:
- Include ALL parameters mentioned in the documentation.
- For non-tensor params (strides, padding, etc.), set rank=null and rank_any=false.
- The "primary_rank.fixed_ranks" should ONLY contain ranks of PRIMARY DATA tensors.
- Do NOT include ranks from weight tensors, attribute lists, or intermediate computations.""")

    return "\n".join(parts)


# ── response parsing ─────────────────────────────────────────────

def _extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from LLM response, handling markdown fences."""
    if not text:
        return None
    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_start = cleaned.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _parse_llm_response(
    llm_result: Dict[str, Any],
    api_name: str,
    input_arg_names: List[str],
) -> Dict[str, Any]:
    """
    Convert LLM JSON response into the standard rank.json format.
    """
    primary_info = llm_result.get("primary_rank") or {}
    params_info = llm_result.get("params") or {}

    # Extract primary rank
    fixed_ranks = primary_info.get("fixed_ranks") or []
    rank_any = bool(primary_info.get("rank_any", False))
    rank_min = primary_info.get("rank_min")
    rank_max = primary_info.get("rank_max")

    # Validate fixed_ranks
    valid_ranks = []
    for r in fixed_ranks:
        try:
            val = int(r)
            if 1 <= val <= 8:
                valid_ranks.append(val)
        except (ValueError, TypeError):
            pass

    # Build param_rank_details
    param_rank_details: Dict[str, Any] = {}
    for pname, pinfo in params_info.items():
        if not isinstance(pinfo, dict):
            continue
        sem_role = pinfo.get("semantic_role", "scalar_attr")
        p_rank = pinfo.get("rank")
        p_rank_any = bool(pinfo.get("rank_any", False))
        p_rank_min = pinfo.get("rank_min")

        # Map semantic role to coarse role for backward compatibility
        coarse_role = "attr"
        if sem_role in (SEMANTIC_ROLE_DATA_TENSOR,):
            coarse_role = "primary"
        elif sem_role in (SEMANTIC_ROLE_WEIGHT_TENSOR, SEMANTIC_ROLE_AUX_TENSOR):
            coarse_role = "aux"

        # Determine if included in api-level rank
        included = (coarse_role == "primary")

        param_rank_details[pname] = {
            "role": coarse_role,
            "semantic_role": sem_role,
            "fixed_ranks": [int(p_rank)] if p_rank is not None else [],
            "rank_min": int(p_rank_min) if p_rank_min is not None else None,
            "rank_any": p_rank_any,
            "included_in_api_rank": included,
            "description_summary": pinfo.get("description_summary", ""),
        }

    # Determine status
    has_rank_info = bool(valid_ranks) or rank_any or (rank_min is not None)
    status = "assigned" if has_rank_info else "unassigned"

    # Compute rank_max from candidates if not set
    if valid_ranks and rank_max is None:
        rank_max = max(valid_ranks)

    return {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "rank_candidates": sorted(set(valid_ranks)),
        "rank_any": rank_any,
        "rank_min": int(rank_min) if rank_min is not None else None,
        "rank_max": int(rank_max) if rank_max is not None else None,
        "marker": "__RANK_FROM_DOC__",
        "status": status,
        "param_rank_details": param_rank_details,
        "llm_reasoning": llm_result.get("reasoning", ""),
        "extraction_method": "llm",
    }


# ── regex fallback (simplified from extract_tf_doc_hints.py) ─────

RE_ND = re.compile(r"\b([1-9])\s*-?\s*D\b", re.IGNORECASE)
RE_RANK_EQ = re.compile(r"\brank\s*[:=]?\s*([1-9])\b", re.IGNORECASE)
RE_ANY_RANK = [
    re.compile(r"\bany\s+number\s+of\s+dimensions\b", re.IGNORECASE),
    re.compile(r"\barbitrary\s+rank\b", re.IGNORECASE),
    re.compile(r"\bN-?D\b"),
    re.compile(r"\bany\s+rank\b", re.IGNORECASE),
    re.compile(r"\bany\s+shape\b", re.IGNORECASE),
]


def _regex_fallback_extract(
    api_name: str,
    doc_text: str,
    schema_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Simplified regex fallback — delegates to extract_tf_doc_hints if available,
    otherwise does a bare-minimum extraction.
    """
    try:
        from extract_tf_doc_hints import extract_doc_rank_hints
        result = extract_doc_rank_hints(api_name, doc_text, schema_dir)
        result["extraction_method"] = "regex_fallback"
        return result
    except ImportError:
        pass

    # Bare minimum extraction
    all_ranks = []
    rank_any = False
    for m in RE_ND.finditer(doc_text):
        try:
            all_ranks.append(int(m.group(1)))
        except ValueError:
            pass
    for m in RE_RANK_EQ.finditer(doc_text):
        try:
            all_ranks.append(int(m.group(1)))
        except ValueError:
            pass
    for pat in RE_ANY_RANK:
        if pat.search(doc_text):
            rank_any = True
            break

    # Filter small ranks that are likely attr lengths
    candidates = sorted(set(r for r in all_ranks if r >= 2))

    return {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "rank_candidates": candidates,
        "rank_any": rank_any,
        "rank_min": None,
        "rank_max": max(candidates) if candidates else None,
        "marker": "__RANK_FROM_DOC__",
        "status": "assigned" if (candidates or rank_any) else "unassigned",
        "param_rank_details": {},
        "extraction_method": "regex_bare_fallback",
    }


# ── core extraction logic ────────────────────────────────────────

def extract_with_llm(
    api_name: str,
    doc_text: str,
    schema_dir: Optional[Path],
    llm_client: Optional[LLMClient],
) -> Dict[str, Any]:
    """
    Main extraction function: tries LLM first, falls back to regex.
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
            "extraction_method": "none",
        }

    # Load schema info if available
    schema_info = None
    if schema_dir:
        schema_path = schema_dir / f"{api_name.replace('.', '_')}_schema.json"
        if schema_path.exists():
            try:
                schema_info = load_json(schema_path)
            except Exception:
                pass

    input_arg_names = []
    if schema_info:
        tf_block = schema_info.get("tf") or {}
        op_def = tf_block.get("op_def") or {}
        input_args = op_def.get("input_args") or []
        input_arg_names = [a["name"] for a in input_args if isinstance(a, dict) and a.get("name")]

    # Try LLM extraction
    if llm_client is not None:
        user_prompt = build_user_prompt(api_name, doc_text, schema_info)
        raw_response = llm_client.chat(SYSTEM_PROMPT, user_prompt)

        if raw_response:
            parsed = _extract_json_from_response(raw_response)
            if parsed and isinstance(parsed, dict):
                try:
                    result = _parse_llm_response(parsed, api_name, input_arg_names)
                    result["llm_raw_response"] = raw_response[:2000]  # truncate for storage
                    return result
                except Exception as e:
                    print(f"  [LLM] parse error for {api_name}: {e}")
            else:
                print(f"  [LLM] could not extract JSON for {api_name}")
        else:
            print(f"  [LLM] no response for {api_name}")

    # Fallback to regex
    return _regex_fallback_extract(api_name, doc_text, schema_dir)


# ── batch processing with LLM (for efficiency) ──────────────────

def build_batch_user_prompt(
    apis: List[Tuple[str, str, Optional[Dict[str, Any]]]],
) -> str:
    """
    Build a prompt for batch processing multiple APIs at once.
    Each entry is (api_name, doc_text, schema_info).
    Max 5 APIs per batch to stay within token limits.
    """
    parts = []
    parts.append("Analyze the following TensorFlow APIs and return a JSON array with one entry per API.\n")

    for i, (api_name, doc_text, schema_info) in enumerate(apis):
        parts.append(f"=== API {i+1}: {api_name} ===")

        if schema_info:
            tf_block = schema_info.get("tf") or {}
            op_def = tf_block.get("op_def") or {}
            input_args = op_def.get("input_args") or []
            if input_args:
                parts.append("Schema input_args:")
                for arg in input_args:
                    name = arg.get("name", "?")
                    type_name = arg.get("type_name") or arg.get("type_attr") or "?"
                    parts.append(f"  - {name}: type={type_name}")

        parts.append("Doc:")
        truncated = doc_text[:3000] if len(doc_text) > 3000 else doc_text
        parts.append(truncated)
        parts.append("")

    parts.append("""\
Return a JSON array where each element has the structure:
{
  "api_name": "<api_name>",
  "primary_rank": {
    "fixed_ranks": [<ints>],
    "rank_any": <bool>,
    "rank_min": <int or null>,
    "rank_max": <int or null>
  },
  "params": {
    "<param_name>": {
      "semantic_role": "<role>",
      "rank": <int or null>,
      "rank_any": <bool>,
      "rank_min": <int or null>,
      "description_summary": "<summary>"
    }
  },
  "reasoning": "<brief>"
}

Semantic roles: data_tensor, weight_tensor, aux_tensor, index_input, shape_control, fixed_arity_list, layout_attr, scalar_attr, dtype_attr, meta

REMEMBER: primary_rank.fixed_ranks = ONLY primary data tensor ranks. NOT weight/attr/intermediate ranks.""")

    return "\n".join(parts)


def extract_batch_with_llm(
    api_entries: List[Tuple[str, str, Optional[Dict[str, Any]]]],
    llm_client: LLMClient,
    schema_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    Process a batch of APIs through the LLM.
    Returns a list of rank.json dicts, one per API.
    """
    if not api_entries:
        return []

    user_prompt = build_batch_user_prompt(api_entries)
    raw_response = llm_client.chat(SYSTEM_PROMPT, user_prompt)

    results = []
    if raw_response:
        parsed = _extract_json_from_response(raw_response)
        if isinstance(parsed, list):
            for i, (api_name, doc_text, schema_info) in enumerate(api_entries):
                input_arg_names = []
                if schema_info:
                    tf_block = schema_info.get("tf") or {}
                    op_def = tf_block.get("op_def") or {}
                    input_args = op_def.get("input_args") or []
                    input_arg_names = [a["name"] for a in input_args
                                       if isinstance(a, dict) and a.get("name")]
                if i < len(parsed) and isinstance(parsed[i], dict):
                    try:
                        result = _parse_llm_response(parsed[i], api_name, input_arg_names)
                        results.append(result)
                        continue
                    except Exception as e:
                        print(f"  [LLM-batch] parse error for {api_name}: {e}")
                # Fallback for this entry
                results.append(_regex_fallback_extract(api_name, doc_text, schema_dir))
        else:
            # Batch parse failed, fallback all
            for api_name, doc_text, _ in api_entries:
                results.append(_regex_fallback_extract(api_name, doc_text, schema_dir))
    else:
        for api_name, doc_text, _ in api_entries:
            results.append(_regex_fallback_extract(api_name, doc_text, schema_dir))

    return results


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
        description="LLM-assisted rank extraction from TF doc text files."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mapping_json",
                    help="json object: {api_name: [txt1, txt2, ...]}")
    g.add_argument("--api_list",
                    help="txt/json/pkl API list; requires --doc_dir")
    ap.add_argument("--doc_dir",
                    help="directory containing per-api txt docs")
    ap.add_argument("--schema_dir", default=None,
                    help="directory containing *_schema.json files")
    ap.add_argument("--out_dir", default="./tf_rank_hints",
                    help="where to write *.rank.json")

    # LLM configuration
    ap.add_argument("--llm_base_url", default="https://api.gpt.ge/v1/",
                    help="OpenAI-compatible API base URL (e.g., http://localhost:11434/v1)")
    ap.add_argument("--llm_model", default="gpt-5-codex",
                    help="model name for the LLM API")
    ap.add_argument("--llm_api_key", default="sk-WXtqOuBZPY096KTcDdE866275274464d88943d068aA7Ff5d",
                    help="API key (or set LLM_API_KEY env var)")
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm_max_tokens", type=int, default=4096)
    ap.add_argument("--llm_timeout", type=float, default=120.0)
    ap.add_argument("--no_llm", action="store_true",
                    help="disable LLM, use regex fallback only")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="number of APIs per LLM call (1=individual, 2-5=batch)")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="delay between LLM calls in seconds")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_dir = Path(args.schema_dir).resolve() if args.schema_dir else None

    # Build LLM client
    llm_client = None
    if not args.no_llm and args.llm_base_url:
        llm_client = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key or "",
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
        )
        print(f"[i] LLM enabled: {args.llm_base_url} model={args.llm_model}")
    else:
        print("[i] LLM disabled, using regex fallback only")

    # Build API → doc paths mapping
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

    # Process APIs
    batch_size = max(1, min(5, args.batch_size))
    count = 0
    llm_count = 0
    fallback_count = 0

    if batch_size > 1 and llm_client is not None:
        # Batch mode
        items = list(mapping.items())
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]

            # Prepare batch entries
            batch_entries: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
            for api_name, paths in batch:
                text = read_texts(paths)
                schema_info = None
                if schema_dir:
                    sp = schema_dir / f"{api_name.replace('.', '_')}_schema.json"
                    if sp.exists():
                        try:
                            schema_info = load_json(sp)
                        except Exception:
                            pass
                batch_entries.append((api_name, text, schema_info))

            results = extract_batch_with_llm(batch_entries, llm_client, schema_dir)

            for (api_name, _paths), result in zip(batch, results):
                out_path = out_dir / f"{safe_name(api_name)}.rank.json"
                dump_json(out_path, result)
                method = result.get("extraction_method", "?")
                rc = result.get("rank_candidates", [])
                if "llm" in method:
                    llm_count += 1
                else:
                    fallback_count += 1
                print(f"[+] {api_name} -> {out_path}  ranks={rc} method={method}")
                count += 1

            if args.delay > 0 and batch_start + batch_size < len(items):
                time.sleep(args.delay)
    else:
        # Individual mode
        for api_name, paths in mapping.items():
            text = read_texts(paths)
            result = extract_with_llm(api_name, text, schema_dir, llm_client)

            out_path = out_dir / f"{safe_name(api_name)}.rank.json"
            dump_json(out_path, result)

            method = result.get("extraction_method", "?")
            rc = result.get("rank_candidates", [])
            if "llm" in method:
                llm_count += 1
            else:
                fallback_count += 1
            print(f"[+] {api_name} -> {out_path}  ranks={rc} method={method}")
            count += 1

            if llm_client and args.delay > 0:
                time.sleep(args.delay)

    print(f"\n[done] wrote {count} rank hint files (llm={llm_count}, fallback={fallback_count})")


if __name__ == "__main__":
    main()
