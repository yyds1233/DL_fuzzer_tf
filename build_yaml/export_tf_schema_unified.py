#!/usr/bin/env python3
"""
export_tf_schema_unified.py  –  Stage A: unified schema export for BOTH
tf.raw_ops.* AND high-level APIs (tf.nn.*, tf.math.*, tf.*, etc.)

==========================================================================
WHAT THIS DOES DIFFERENTLY FROM export_tf_schema.py
==========================================================================

The original export_tf_schema.py:
  - Gets Python signature via inspect.signature()
  - Gets OpDef via op_def_registry.get(op_name) using the LAST part of the
    API name (e.g., "Conv2D" from "tf.raw_ops.Conv2D")

This unified version:
  - Still gets Python signature
  - Uses the TFApiResolver to find the underlying raw_op
  - If resolved: gets the OpDef via the resolved raw_op name
  - Stores the param_mapping so downstream stages can translate param names
  - For unresolved APIs: relies on Python signature + LLM classification

==========================================================================
USAGE
==========================================================================

  # Works for both raw_ops and high-level APIs
  python export_tf_schema_unified.py \
      --api_list apis.txt \
      --out_dir ./tf_api_schema

  # With LLM for unresolvable APIs
  python export_tf_schema_unified.py \
      --api_list apis.txt \
      --out_dir ./tf_api_schema \
      --llm_base_url http://localhost:11434/v1 \
      --llm_model qwen2.5:72b

  # Example api_list content:
  #   tf.raw_ops.Conv2D
  #   tf.nn.conv2d
  #   tf.math.reduce_sum
  #   tf.reshape

==========================================================================
OUTPUT FORMAT
==========================================================================

Same as original, but with additional fields:
  - resolve_info: { raw_op_name, param_mapping, inverse_mapping, ... }
  - api_category: "nn" | "math" | "raw_ops" | ...
  - api_module: "tf.nn" | "tf.math" | ...
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tf_schema_common import dump_json, load_api_list, decode_bytes_maybe
from tf_api_resolver import TFApiResolver, ResolveResult

# Reuse OpDef helpers from original
from export_tf_schema import (
    python_signature_to_dict,
    opdef_to_dict,
    get_tf_opdef,
)

GENERATOR_BLOCK = {
    "stage": "A-export-tf-schema-unified",
    "version": "2026-03-23-tf-v3",
}


# ── resolve helpers ──────────────────────────────────────────────

def resolve_obj_from_qualname(qualname: str):
    qualname = qualname.strip()
    if qualname.endswith("()"):
        qualname = qualname[:-2]
    module_name, _, attr_name = qualname.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid qualified name: {qualname}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


# ── main export logic ────────────────────────────────────────────

def export_tf_api_schema_unified(
    api_name: str,
    out_dir: Path,
    resolver: TFApiResolver,
) -> None:
    """
    Export schema for any TF API (raw_ops or high-level).
    """
    api_info: Dict[str, Any] = {
        "generator": GENERATOR_BLOCK,
        "api_name": api_name,
        "python_signature": None,
        "tf": None,
        "resolve_info": None,
        "api_category": "unknown",
        "api_module": "",
        "error": None,
    }

    # 1) Resolve the Python object and get signature
    obj = None
    try:
        obj = resolve_obj_from_qualname(api_name)
        sig = inspect.signature(obj)
        api_info["python_signature"] = python_signature_to_dict(sig)
    except Exception as e:
        api_info["error"] = f"inspect.signature failed: {e}"

    # 2) Resolve to raw_op
    resolve_result = resolver.resolve(api_name)
    api_info["resolve_info"] = resolve_result.to_dict()
    api_info["api_category"] = resolve_result.api_category
    api_info["api_module"] = resolve_result.api_module

    # 3) Get OpDef from the resolved raw_op
    raw_op_name = resolve_result.raw_op_name
    if raw_op_name:
        try:
            opdef = get_tf_opdef(raw_op_name)
            if opdef is not None:
                api_info["tf"] = {
                    "op_name": raw_op_name,
                    "raw_api_name": f"tf.raw_ops.{raw_op_name}",
                    "high_level_api_name": api_name,
                    "op_def": opdef_to_dict(opdef),
                    "param_mapping": resolve_result.param_mapping,
                    "inverse_mapping": resolve_result.inverse_mapping,
                }
            else:
                note = f"OpDef not found for resolved raw_op: {raw_op_name}"
                api_info["error"] = (
                    (api_info["error"] + " | " if api_info["error"] else "") + note
                )
        except Exception as e:
            note = f"OpDef load failed for {raw_op_name}: {e}"
            api_info["error"] = (
                (api_info["error"] + " | " if api_info["error"] else "") + note
            )
    else:
        # No raw_op found — this is OK for composite/unresolved APIs
        # The pipeline will rely on Python signature + LLM
        if resolve_result.strategy == "unresolved":
            api_info["error"] = (
                (api_info["error"] + " | " if api_info["error"] else "")
                + "no raw_op resolved (composite or unknown API)"
            )
        # For composite ops that have raw_op_names but no primary
        if resolve_result.raw_op_names:
            api_info["tf"] = {
                "op_name": None,
                "raw_api_name": None,
                "high_level_api_name": api_name,
                "op_def": None,
                "param_mapping": resolve_result.param_mapping,
                "inverse_mapping": resolve_result.inverse_mapping,
                "composite_ops": resolve_result.raw_op_names,
            }

    # 4) Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{api_name.replace('.', '_')}_schema.json"
    dump_json(out_path, api_info)
    status = "+" if raw_op_name else "~"
    print(
        f"[{status}] {api_name} → raw_op={raw_op_name or 'None'} "
        f"strategy={resolve_result.strategy} → {out_path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified TF API schema export (raw_ops + high-level)."
    )
    ap.add_argument("--api_list", required=True,
                    help="txt/json/pkl, each entry is a qualified TF API name")
    ap.add_argument("--out_dir", default="./tf_api_schema",
                    help="output dir for schema json files")
    ap.add_argument("--llm_base_url", default=None,
                    help="OpenAI-compatible API base URL for resolver fallback")
    ap.add_argument("--llm_model", default="gpt-5-codex",
                    help="model name for LLM resolver")
    ap.add_argument("--llm_api_key", default=None,
                    help="API key for LLM")
    args = ap.parse_args()

    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        raise SystemExit(f"TensorFlow import failed: {e}")

    api_list = load_api_list(args.api_list)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build LLM client if requested
    llm_client = None
    if args.llm_base_url:
        from llm_doc_rank_extractor import LLMClient
        llm_client = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key or "",
        )
        print(f"[i] LLM resolver enabled: {args.llm_base_url}")

    resolver = TFApiResolver(llm_client=llm_client)

    print(f"[i] loaded {len(api_list)} APIs")
    stats = {"total": 0, "resolved": 0, "unresolved": 0, "errors": 0}

    for api in api_list:
        try:
            export_tf_api_schema_unified(api, out_dir, resolver)
            stats["total"] += 1
            # Check if resolved
            resolve_path = out_dir / f"{api.replace('.', '_')}_schema.json"
            if resolve_path.exists():
                data = json.loads(resolve_path.read_text())
                ri = data.get("resolve_info") or {}
                if ri.get("raw_op_name"):
                    stats["resolved"] += 1
                else:
                    stats["unresolved"] += 1
        except Exception as e:
            stats["errors"] += 1
            out_path = out_dir / f"{api.replace('.', '_')}_schema.json"
            dump_json(out_path, {
                "generator": GENERATOR_BLOCK,
                "api_name": api,
                "python_signature": None,
                "tf": None,
                "resolve_info": None,
                "error": f"fatal export error: {e}",
            })
            print(f"[!] failed: {api}: {e}")

    print(f"\n[done] total={stats['total']}, resolved={stats['resolved']}, "
          f"unresolved={stats['unresolved']}, errors={stats['errors']}")


if __name__ == "__main__":
    main()
