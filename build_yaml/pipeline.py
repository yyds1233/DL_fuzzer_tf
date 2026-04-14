#!/usr/bin/env python3
"""
run_pipeline_unified.py  –  End-to-end pipeline runner for generating YAML
test specifications from TensorFlow APIs (both raw_ops and high-level).

==========================================================================
USAGE
==========================================================================

  # Full pipeline with LLM support
  python run_pipeline_unified.py \
      --api_list apis.txt \
      --doc_dir ./tf_docs \
      --work_dir ./pipeline_output \
      --llm_base_url http://localhost:11434/v1 \
      --llm_model qwen2.5:72b \
      --openai_base_url https://api.openai.com/v1 \
      --openai_model gpt-4o

  # Minimal (no LLM, heuristic only)
  python run_pipeline_unified.py \
      --api_list apis.txt \
      --work_dir ./pipeline_output \
      --no_llm

  # Run only specific stages
  python run_pipeline_unified.py \
      --api_list apis.txt \
      --work_dir ./pipeline_output \
      --stages A,B

==========================================================================
PIPELINE STAGES
==========================================================================

  A.resolve  : Resolve high-level APIs to raw_ops (tf_api_resolver.py)
  A.schema   : Export unified schema JSON (export_tf_schema_unified.py)
  A.doc      : Extract rank hints from docs (llm_doc_rank_extractor.py)
  B.skeleton : Build YAML skeletons (tf_schema2yaml_unified.py)
  B.normalize: Normalize YAML (normalize_yaml_skeleton.py)
  B.enrich   : Enrich with semantic roles (param_role_enricher_unified.py)
  C.shapes   : LLM shape materialization (tf_llm_patch_yaml.py)
  D.constrain: LLM constraint patch (tf_patch_constraints.py)
  E.material : Final materialization (tf_materialize_yaml.py)

==========================================================================
EXAMPLE api_list.txt
==========================================================================

  # Raw ops (original support)
  tf.raw_ops.Conv2D
  tf.raw_ops.MatMul
  tf.raw_ops.Relu

  # High-level APIs (NEW)
  tf.nn.conv2d
  tf.nn.relu
  tf.nn.softmax
  tf.math.reduce_sum
  tf.math.matmul
  tf.reshape
  tf.transpose
  tf.concat
  tf.gather
  tf.image.resize
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════
# Stage definitions
# ══════════════════════════════════════════════════════════════════

ALL_STAGES = ["A", "B", "C", "D", "E"]

STAGE_DESCRIPTIONS = {
    "A": "Schema export + resolve + doc rank extraction",
    "B": "YAML skeleton + normalize + enrich",
    "C": "LLM shape materialization",
    "D": "LLM constraint patch",
    "E": "Final materialization",
}


def run_cmd(cmd: List[str], desc: str, env: Optional[Dict[str, str]] = None) -> bool:
    """Run a subprocess command. Returns True on success."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=full_env)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage failed with code {result.returncode}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    schema_dir = work / "01_schema"
    resolved_dir = work / "01_resolved"
    rank_dir = work / "02_rank_hints"
    skeleton_dir = work / "03_skeleton"
    normalized_dir = work / "04_normalized"
    enriched_dir = work / "05_enriched"
    shaped_dir = work / "06_shaped"
    constrained_dir = work / "07_constrained"
    final_dir = work / "08_final"

    stages = set(args.stages.upper().split(",")) if args.stages else set(ALL_STAGES)

    python = sys.executable

    # Common LLM args
    llm_args = []
    if not args.no_llm and args.llm_base_url:
        llm_args = [
            "--llm_base_url", args.llm_base_url,
            "--llm_model", args.llm_model,
        ]
        if args.llm_api_key:
            llm_args += ["--llm_api_key", args.llm_api_key]

    # OpenAI env for Stages C/D
    openai_env = {}
    if args.openai_api_key:
        openai_env["OPENAI_API_KEY"] = args.openai_api_key
    elif os.environ.get("OPENAI_API_KEY"):
        openai_env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if args.openai_base_url:
        openai_env["OPENAI_BASE_URL"] = args.openai_base_url
    elif os.environ.get("OPENAI_BASE_URL"):
        openai_env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]

    success = True

    # ── Stage A ──────────────────────────────────────────────────
    if "A" in stages:
        # A.schema: Unified schema export (includes resolver)
        cmd = [
            python, "export_tf_schema_unified.py",
            "--api_list", args.api_list,
            "--out_dir", str(schema_dir),
        ] + llm_args
        if not run_cmd(cmd, "Stage A: Unified schema export"):
            success = False
            if not args.continue_on_error:
                return

        # A.doc: Rank extraction from docs (if doc_dir provided)
        if args.doc_dir:
            cmd = [
                python, "llm_doc_rank_extractor.py",
                "--api_list", args.api_list,
                "--doc_dir", args.doc_dir,
                "--schema_dir", str(schema_dir),
                "--out_dir", str(rank_dir),
            ]
            if not args.no_llm and args.llm_base_url:
                cmd += llm_args
            else:
                cmd += ["--no_llm"]
            if not run_cmd(cmd, "Stage A: Doc rank extraction"):
                success = False
                if not args.continue_on_error:
                    return

    # ── Stage B ──────────────────────────────────────────────────
    if "B" in stages:
        # B.skeleton: Unified YAML skeleton builder
        cmd = [
            python, "tf_schema2yaml_unified.py",
            "--schema_json", str(schema_dir),
            "--out_dir", str(skeleton_dir),
        ]
        if rank_dir.exists():
            cmd += ["--rank_index_dir", str(rank_dir)]
        if args.doc_dir:
            cmd += ["--doc_dir", args.doc_dir]
        cmd += llm_args
        if not run_cmd(cmd, "Stage B: Unified YAML skeleton"):
            success = False
            if not args.continue_on_error:
                return

        # B.normalize
        cmd = [
            python, "normalize_yaml_skeleton.py",
            "--yaml", str(skeleton_dir),
            "--out_dir", str(normalized_dir),
        ]
        if not run_cmd(cmd, "Stage B: Normalize YAML"):
            success = False
            if not args.continue_on_error:
                return

        # B.enrich: Unified enricher
        cmd = [
            python, "param_role_enricher_unified.py",
            "--yaml", str(normalized_dir),
            "--out_dir", str(enriched_dir),
        ]
        if rank_dir.exists():
            cmd += ["--rank_dir", str(rank_dir)]
        cmd += llm_args
        if not run_cmd(cmd, "Stage B: Unified enrichment"):
            success = False
            if not args.continue_on_error:
                return

    # ── Stage C ──────────────────────────────────────────────────
    if "C" in stages:
        if not openai_env.get("OPENAI_API_KEY"):
            print("[!] Stage C requires OPENAI_API_KEY — skipping")
        else:
            shaped_dir.mkdir(parents=True, exist_ok=True)
            yaml_files = sorted(enriched_dir.glob("*.yaml"))
            for yp in yaml_files:
                api_name = yp.stem  # e.g., tf_nn_conv2d
                # Find corresponding doc file
                doc_path = None
                if args.doc_dir:
                    doc_path = Path(args.doc_dir) / f"{api_name}.txt"
                    if not doc_path.exists():
                        doc_path = None

                if doc_path:
                    cmd = [
                        python, "llm_patch_yaml_new.py",
                        "--doc_txt", str(doc_path),
                        "--yaml_in", str(yp),
                        "--yaml_out_dir", str(shaped_dir),
                        "--model", args.openai_model or "gpt-5-codex",
                    ]
                    run_cmd(cmd, f"Stage C: Shape {api_name}", env=openai_env)
                else:
                    print(f"[!] No doc for {api_name}, copying as-is")
                    import shutil
                    shutil.copy2(yp, shaped_dir / yp.name)

    # ── Stage D ──────────────────────────────────────────────────
    if "D" in stages:
        if not openai_env.get("OPENAI_API_KEY"):
            print("[!] Stage D requires OPENAI_API_KEY — skipping")
        else:
            constrained_dir.mkdir(parents=True, exist_ok=True)
            src_dir = shaped_dir if shaped_dir.exists() else enriched_dir
            yaml_files = sorted(src_dir.glob("*.yaml"))
            for yp in yaml_files:
                if yp.suffix != ".yaml":
                    continue
                api_name = yp.stem
                doc_path = None
                if args.doc_dir:
                    doc_path = Path(args.doc_dir) / f"{api_name}.txt"
                    if not doc_path.exists():
                        doc_path = None

                out_path = constrained_dir / yp.name

                if doc_path:
                    cmd = [
                        python, "tf_patch_constraints.py",
                        "--doc_txt", str(doc_path),
                        "--yaml_in", str(yp),
                        "--yaml_out", str(out_path),
                        "--model", args.openai_model or "gpt-4o",
                    ]
                    run_cmd(cmd, f"Stage D: Constraints {api_name}", env=openai_env)
                else:
                    import shutil
                    shutil.copy2(yp, out_path)

    # ── Stage E ──────────────────────────────────────────────────
    if "E" in stages:
        src_dir = constrained_dir if constrained_dir.exists() else (
            shaped_dir if shaped_dir.exists() else enriched_dir
        )
        cmd = [
            python, "tf_materialize_yaml.py",
            "--yaml_dir", str(src_dir),
            "--out_dir", str(final_dir),
        ]
        if not run_cmd(cmd, "Stage E: Final materialization"):
            success = False

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE {'COMPLETE' if success else 'FINISHED WITH ERRORS'}")
    print(f"{'='*60}")
    print(f"  Work directory: {work}")
    for d in [schema_dir, rank_dir, skeleton_dir, normalized_dir,
              enriched_dir, shaped_dir, constrained_dir, final_dir]:
        if d.exists():
            count = len(list(d.glob("*.yaml"))) + len(list(d.glob("*.json")))
            print(f"  {d.name}: {count} files")
    if final_dir.exists():
        print(f"\n  Final output: {final_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Unified pipeline: TF API → YAML test spec "
                    "(raw_ops + high-level APIs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with local LLM
  python run_pipeline_unified.py \\
      --api_list apis.txt \\
      --doc_dir ./tf_docs \\
      --work_dir ./output \\
      --llm_base_url http://localhost:11434/v1 \\
      --llm_model qwen2.5:72b

  # Stages A+B only (no OpenAI needed)
  python run_pipeline_unified.py \\
      --api_list apis.txt \\
      --work_dir ./output \\
      --stages A,B \\
      --no_llm

  # Example API list:
  #   tf.raw_ops.Conv2D
  #   tf.nn.conv2d
  #   tf.math.reduce_sum
  #   tf.reshape
        """,
    )
    ap.add_argument("--api_list", required=True,
                    help="txt/json file of API names (raw_ops and/or high-level)")
    ap.add_argument("--doc_dir", default=None,
                    help="directory with per-API .txt documentation files")
    ap.add_argument("--work_dir", default="./pipeline_output",
                    help="working directory for all intermediate outputs")
    ap.add_argument("--stages", default=None,
                    help="comma-separated stages to run (e.g., A,B,C). Default: all")

    # LLM for resolver + rank extraction + enrichment
    ap.add_argument("--llm_base_url", default="https://api.gpt.ge/v1/",
                    help="OpenAI-compatible API for resolver/rank/enrich")
    ap.add_argument("--llm_model", default="gpt-5-codex",)
    ap.add_argument("--llm_api_key", default="sk-WXtqOuBZPY096KTcDdE866275274464d88943d068aA7Ff5d")
    ap.add_argument("--no_llm", action="store_true",
                    help="disable all LLM usage (heuristic only)")

    # OpenAI for Stages C/D
    ap.add_argument("--openai_base_url", default="https://api.gpt.ge/v1/",
                    help="OpenAI API base URL for shape/constraint LLM")
    ap.add_argument("--openai_model", default="gpt-5-codex",)
    ap.add_argument("--openai_api_key", default="sk-WXtqOuBZPY096KTcDdE866275274464d88943d068aA7Ff5d",
                    help="OpenAI API key (or set OPENAI_API_KEY env)")

    ap.add_argument("--continue_on_error", action="store_true",
                    help="continue pipeline even if a stage fails")

    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
