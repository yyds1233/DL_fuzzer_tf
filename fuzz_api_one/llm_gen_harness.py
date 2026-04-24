#!/usr/bin/env python3
"""
LLM-driven harness generator for DL_fuzzer_tf.

Reads:
  1) api yaml file
  2) api txt doc file
  3) Atheris doc / notes file

Then calls an OpenAI-compatible LLM endpoint to generate a standalone,
executable Atheris fuzz harness (.py), extracts Python code from the model
response, validates it, optionally asks the model to repair it, and writes
the final harness to disk.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: openai\n"
        "Install with: pip install openai pyyaml"
    ) from exc


DEFAULT_SYSTEM_PROMPT = """
# Identity
You are an expert Python fuzzing engineer specializing in coverage-guided fuzzing for TensorFlow APIs with Atheris.

# Mission
Generate one high-quality, executable Python Atheris harness for the target TensorFlow API.

# Priority Order
1. The harness must be executable and syntactically valid Python 3.
2. The harness must actually invoke the target TensorFlow API from the provided materials.
3. The harness should maximize reachable API parameter-space coverage and downstream code coverage.
4. The harness should use the YAML aggressively when it is plausible, but must not follow the YAML blindly if it appears inconsistent with the API text, TensorFlow semantics, or executability.

# Interpretation Rules
- Treat the YAML as a high-value but potentially imperfect intermediate representation.
- Prefer the YAML when it appears reasonable and internally consistent.
- If the YAML seems incomplete, noisy, or clearly inconsistent with the API text or likely TensorFlow calling semantics, use your best judgment and rely more on the API text and normal TensorFlow usage patterns.
- Never intentionally produce a harness that is obviously broken just to obey the YAML.
- When uncertain, favor a harness that is executable, exercises the real API, and explores diverse valid argument combinations.

# Coverage Goals
- Prefer harness logic that can explore broad combinations of dtypes, ranks, shapes, optional arguments, attributes, and boundary values over time.
- Do not collapse the harness into one narrow “mostly valid” input pattern.
- Bias toward semantically valid inputs, but preserve meaningful diversity so the fuzzer can reach as much code as possible.
- Where compatible parameters are required, enforce compatibility.
- Where the API admits multiple valid modes, branches, or attribute choices, expose those modes to fuzzing rather than hard-coding one mode.
- Prefer cheap/small tensors by default, but still allow controlled variation across ranks, dimensions, shapes, and values so different execution paths remain reachable.
- If the API supports multiple legal tensor ranks, dtypes, or shape patterns, expose those alternatives to fuzzing.
- If attributes or enum-like arguments alter execution paths, vary them rather than fixing a single constant.

# Exception Handling Rules
- Catch and ignore only expected benign exceptions from fuzzed inputs or normal TensorFlow input validation.
- Prefer narrow exceptions such as TypeError, ValueError, tf.errors.InvalidArgumentError, or other clearly input-related TensorFlow exceptions when appropriate.
- Never swallow KeyboardInterrupt, SystemExit, MemoryError, or AssertionError.
- Do not use broad exception handling unless fatal exceptions are explicitly re-raised first and the remaining catch is justified.

# Hard Requirements
# Hard Requirements
1. Output exactly one Python code block and nothing else.
2. The harness must be a complete standalone .py file.
3. Import `atheris` as the first import.
4. Do NOT use `with atheris.instrument_imports():`.
5. Do NOT call `atheris.instrument_all()`.
6. Add `@atheris.instrument_func` immediately before `TestOneInput`.
7. Define `TestOneInput(data: bytes)`.
8. Use `atheris.FuzzedDataProvider(data)` inside `TestOneInput` to decode bytes into arguments.
9. Call `atheris.Setup(sys.argv, TestOneInput)` before `atheris.Fuzz()`.
10. The harness must actually call the target TensorFlow API named in the provided files.
11. Prefer generating TensorFlow values that are valid often enough to reach deep code, while still exploring diverse argument combinations.
12. Avoid placeholder code, TODOs, pseudo code, markdown, and explanations.
13. Do not depend on any local project helper modules unless they are explicitly included in the provided files.
14. Use deterministic helper functions inside the same file when needed.
15. The output must be directly runnable with Python after dependencies are installed.

Exception-handling requirements:
- Use the following exception-swallowing template in `TestOneInput`.
- Do not invent a different exception policy.
- The harness should return on these expected fuzz-input-related exceptions:

except (
    tf.errors.InvalidArgumentError,
    tf.errors.UnimplementedError,
    tf.errors.InternalError,
    tf.errors.ResourceExhaustedError,
    ValueError,
    TypeError,
    RuntimeError,
    AssertionError,
    IndexError,
    NotImplementedError,
):
    return
except Exception:
    return

- Do not add comments or explanations around this policy unless they are inside Python comments in the generated file.

# Quality Bar
- The best answer is not the shortest harness.
- The best answer is the harness most likely to run successfully and cover a wide range of the target API’s parameter space.
""".strip()


PROMPT_TEMPLATE = """
# Task
Generate one executable Python Atheris harness for the TensorFlow API described below.

# Primary Goal
Produce a coverage-guided fuzz harness that is executable, actually reaches the target API, and explores as much of the API’s valid parameter space as reasonably possible.

# Output Contract
- Return exactly one fenced Python code block.
- No prose before or after the code block.
- The file must be directly runnable with `python harness.py` after dependencies are installed.

# Reliability And Source Priority
- The YAML is an intermediate representation and may be imperfect.
- Use the YAML as the primary structural hint for parameters, dtype relations, ranks, shapes, allowed values, and constraints when it appears plausible.
- However, do NOT treat the YAML as infallible.
- If the YAML appears clearly inconsistent, overly noisy, incomplete, or incompatible with the API txt / likely TensorFlow semantics / executability, then partially or fully override it using the API txt and your best judgment.
- If forced to choose, prefer:
  (a) an executable harness that correctly calls the target API and explores meaningful inputs
  over
  (b) strict obedience to a suspicious YAML detail.

# Target API
- API name: {api_name}
- Additional API markers you may reference if helpful: {api_markers}

# Harness Design Objectives
- Use the YAML as much as reasonably possible.
- Try to cover the full parameter space, not just one narrow valid corner.
- Maximize opportunities for code coverage by exposing:
  - different valid dtypes
  - different valid ranks
  - different compatible shapes
  - optional arguments when applicable
  - attribute/value alternatives
  - boundary-sized tensors and representative edge-case values
- Prefer structured decoding from `atheris.FuzzedDataProvider(data)` rather than ad hoc randomness.
- Enforce parameter compatibility when the API requires shared dtype/shape/rank relationships.
- If attributes have `allowed_values`, expose multiple allowed choices to fuzzing.
- If multiple argument construction strategies are valid, prefer the one that gives better reachable coverage while staying reasonably executable.
- Prefer small tensor sizes to keep execution cheap, but do not make shapes so trivial that most branches become unreachable.
- Include helper functions in the same file to decode booleans, ints, floats, strings, shapes, dtypes, tensors, lists, and optional values when useful.
- Avoid noisy logging/printing.
- Keep the harness self-contained.

# Validity Strategy
- The harness should be validity-biased, not validity-only.
- Generate inputs that are often valid enough to execute meaningful code paths.
- Still preserve diversity across parameter combinations so the fuzzer can mutate toward deeper coverage.
- Catch and ignore only expected benign exceptions caused by malformed fuzz inputs or normal TensorFlow input validation.
- Do not catch broad fatal exceptions such as `KeyboardInterrupt`, `SystemExit`, `MemoryError`, or `AssertionError`.

# Reasoning Policy For Conflicting Information
- When YAML and API text agree, follow them closely.
- When YAML is missing detail, infer reasonable TensorFlow-compatible behavior.
- When YAML conflicts with API text or appears obviously wrong, downweight the YAML and generate the best executable coverage-oriented harness you can.
- Do not mention this reasoning in the final output; only emit the Python file.

# YAML Summary Hints
Use these extracted hints as a compact aid, but verify them against the full materials:
<yaml_summary>
{yaml_summary}
</yaml_summary>

# Full Context
<api_yaml>
{yaml_text}
</api_yaml>

<api_txt>
{api_txt_text}
</api_txt>

<atheris_doc>
{atheris_doc_text}
</atheris_doc>
""".strip()


REPAIR_PROMPT_TEMPLATE = """
# Repair Task
Your previous output did not pass validation.

# Validation Error
{validation_error}

# Required Fixes
- Return exactly one fenced Python code block.
- The result must be syntactically valid Python 3.
- `atheris` must be the first import.
- The harness must contain `with atheris.instrument_imports():`.
- The harness must define `TestOneInput(data: bytes)`.
- The harness must call `atheris.Setup(sys.argv, TestOneInput)` before `atheris.Fuzz()`.
- The harness must actually call the target API from the original task.
- Preserve the original coverage-oriented intent.
- Do not add prose.

# Previous Raw Response
<bad_response>
{bad_response}
</bad_response>

# Extracted Code
<bad_code>
{bad_code}
</bad_code>

# Original Task
<original_task>
{original_prompt}
</original_task>
""".strip()


@dataclass
class GenConfig:
    yaml_path: Path
    api_txt_path: Path
    atheris_doc_path: Path
    out_path: Path
    model: str
    api_key: Optional[str]
    base_url: Optional[str]
    api_mode: str
    temperature: float
    max_output_tokens: int
    max_yaml_chars: int
    max_api_txt_chars: int
    max_atheris_chars: int
    repair_attempts: int
    save_raw_response: bool
    save_prompt: bool
    require_instrument_imports: bool
    require_api_marker: bool


class HarnessGenerationError(RuntimeError):
    pass


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imported_modules: list[str] = []
        self.from_imports: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported_modules.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.from_imports.append(node.module)
        self.generic_visit(node)


def parse_args() -> GenConfig:
    parser = argparse.ArgumentParser(
        description="Generate an Atheris fuzz harness from YAML + API txt + Atheris docs using an LLM."
    )
    parser.add_argument("--yaml", dest="yaml_path", required=True, help="Path to API YAML file")
    parser.add_argument("--api-txt", dest="api_txt_path", required=True, help="Path to API txt file")
    parser.add_argument(
        "--atheris-doc",
        dest="atheris_doc_path",
        required=True,
        help="Path to Atheris README/notes file",
    )
    parser.add_argument("--out", dest="out_path", required=True, help="Output harness .py path")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model name")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL (default: OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat", "auto"],
        default="auto",
        help="LLM API mode. auto = try responses first, then chat.",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=32000)
    parser.add_argument("--max-yaml-chars", type=int, default=300000)
    parser.add_argument("--max-api-txt-chars", type=int, default=500000)
    parser.add_argument("--max-atheris-chars", type=int, default=200000)
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=2,
        help="How many repair rounds to attempt after static validation fails.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Also save raw model response next to output file",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Also save final prompt next to output file",
    )
    parser.add_argument(
        "--no-require-instrument-imports",
        dest="require_instrument_imports",
        action="store_false",
        help="Do not fail validation if `with atheris.instrument_imports():` is absent.",
    )
    parser.add_argument(
        "--no-require-api-marker",
        dest="require_api_marker",
        action="store_false",
        help="Do not fail validation if no obvious API marker string is found in the generated code.",
    )
    parser.set_defaults(require_instrument_imports=True, require_api_marker=True)

    ns = parser.parse_args()
    return GenConfig(
        yaml_path=Path(ns.yaml_path),
        api_txt_path=Path(ns.api_txt_path),
        atheris_doc_path=Path(ns.atheris_doc_path),
        out_path=Path(ns.out_path),
        model=ns.model,
        api_key=ns.api_key,
        base_url=ns.base_url,
        api_mode=ns.api_mode,
        temperature=ns.temperature,
        max_output_tokens=ns.max_output_tokens,
        max_yaml_chars=ns.max_yaml_chars,
        max_api_txt_chars=ns.max_api_txt_chars,
        max_atheris_chars=ns.max_atheris_chars,
        repair_attempts=ns.repair_attempts,
        save_raw_response=ns.save_raw_response,
        save_prompt=ns.save_prompt,
        require_instrument_imports=ns.require_instrument_imports,
        require_api_marker=ns.require_api_marker,
    )


def read_text(path: Path, max_chars: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars - len(head)) :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n[... TRUNCATED {omitted} CHARS ...]\n\n{tail}"


def load_yaml_summary(yaml_text: str) -> Tuple[str, Dict[str, Any]]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise HarnessGenerationError("YAML root must be a mapping/object")
    api_name = str(data.get("api_name") or "UNKNOWN_API")
    return api_name, data


def _safe_get(container: Any, *keys: str, default: Any = None) -> Any:
    cur = container
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _compact(obj: Any, limit: int = 6000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [TRUNCATED {len(text) - limit} CHARS]"


def build_yaml_hint_summary(yaml_obj: Dict[str, Any]) -> str:
    tf_meta = yaml_obj.get("tf") if isinstance(yaml_obj.get("tf"), dict) else {}
    params = yaml_obj.get("params") if isinstance(yaml_obj.get("params"), dict) else {}
    summary = {
        "api_name": yaml_obj.get("api_name"),
        "category": yaml_obj.get("category"),
        "api_category": yaml_obj.get("api_category"),
        "api_module": yaml_obj.get("api_module"),
        "primary_param": yaml_obj.get("primary_param"),
        "test_ranks": yaml_obj.get("test_ranks"),
        "test_dtype_choices": yaml_obj.get("test_dtype_choices"),
        "constraints": yaml_obj.get("constraints"),
        "shape_vars": yaml_obj.get("shape_vars"),
        "rank_hints": yaml_obj.get("rank_hints"),
        "resolve_info": yaml_obj.get("resolve_info"),
        "tf": {
            "op_name": tf_meta.get("op_name"),
            "raw_api_name": tf_meta.get("raw_api_name"),
            "high_level_api_name": tf_meta.get("high_level_api_name"),
            "is_stateful": tf_meta.get("is_stateful"),
            "input_meta": tf_meta.get("input_meta"),
            "attr_meta": tf_meta.get("attr_meta"),
            "output_meta": tf_meta.get("output_meta"),
        },
        "params": params,
    }
    return _compact(summary, limit=12000)


def build_api_markers(yaml_obj: Dict[str, Any], api_name: str) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        value = value.strip()
        if not value or value in seen:
            return
        seen.add(value)
        markers.append(value)

    add(api_name)
    add(yaml_obj.get("api_module"))
    add(yaml_obj.get("category"))
    add(_safe_get(yaml_obj, "tf", "op_name"))
    add(_safe_get(yaml_obj, "tf", "raw_api_name"))
    add(_safe_get(yaml_obj, "tf", "high_level_api_name"))

    raw_names = _safe_get(yaml_obj, "resolve_info", "raw_op_names", default=[])
    if isinstance(raw_names, list):
        for item in raw_names:
            add(item)

    return markers


def build_prompt(
    api_name: str,
    api_markers: Iterable[str],
    yaml_summary: str,
    yaml_text: str,
    api_txt_text: str,
    atheris_doc_text: str,
) -> str:
    return PROMPT_TEMPLATE.format(
        api_name=api_name,
        api_markers=", ".join(api_markers) if api_markers else "(none)",
        yaml_summary=yaml_summary,
        yaml_text=yaml_text,
        api_txt_text=api_txt_text,
        atheris_doc_text=atheris_doc_text,
    )


def create_client(cfg: GenConfig) -> OpenAI:
    kwargs = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)


def call_model_responses(client: OpenAI, cfg: GenConfig, system_prompt: str, user_prompt: str) -> str:
    resp = client.responses.create(
        model=cfg.model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
    )
    text = getattr(resp, "output_text", None)
    if text:
        return text
    try:
        dumped = resp.model_dump()
        if isinstance(dumped, dict) and isinstance(dumped.get("output_text"), str):
            return dumped["output_text"]
        return json.dumps(dumped, ensure_ascii=False, indent=2)
    except Exception:  # pragma: no cover
        return str(resp)


def call_model_chat(client: OpenAI, cfg: GenConfig, system_prompt: str, user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
    )
    choice = completion.choices[0]
    content = choice.message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def call_model(client: OpenAI, cfg: GenConfig, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    errors = []
    modes = [cfg.api_mode] if cfg.api_mode != "auto" else ["responses", "chat"]
    for mode in modes:
        try:
            if mode == "responses":
                return call_model_responses(client, cfg, system_prompt, user_prompt), mode
            if mode == "chat":
                return call_model_chat(client, cfg, system_prompt, user_prompt), mode
            raise ValueError(f"Unsupported api mode: {mode}")
        except Exception as exc:
            errors.append(f"{mode}: {exc}")
    raise HarnessGenerationError("LLM call failed in all modes:\n" + "\n".join(errors))


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                code = obj.get("code") or obj.get("python")
                if isinstance(code, str) and code.strip():
                    return code.strip() + "\n"
        except Exception:
            pass

    return stripped + ("\n" if stripped else "")


def _first_import_is_atheris(tree: ast.AST) -> bool:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Import):
            return any(alias.name == "atheris" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return False
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue  # module docstring
        return False
    return False


def _find_function(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _string_markers_present(code: str, api_markers: Iterable[str]) -> bool:
    lowered = code.lower()
    for marker in api_markers:
        marker = marker.strip()
        if marker and marker.lower() in lowered:
            return True
    return False


def static_validate_python(
    code: str,
    *,
    api_markers: Iterable[str],
    require_instrument_imports: bool,
    require_api_marker: bool,
) -> Tuple[bool, str]:
    problems = []

    if not code.strip():
        return False, "Empty code generated."

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})"

    if not _first_import_is_atheris(tree):
        problems.append("`atheris` must be the first import in the file.")

    if require_instrument_imports and "with atheris.instrument_imports():" not in code:
        problems.append("Missing required construct: `with atheris.instrument_imports():`")

    if "atheris.Setup" not in code:
        problems.append("Missing required marker: atheris.Setup")
    if "atheris.Fuzz" not in code:
        problems.append("Missing required marker: atheris.Fuzz")
    if "FuzzedDataProvider" not in code:
        problems.append("Missing required marker: FuzzedDataProvider")

    test_fn = _find_function(tree, "TestOneInput")
    if test_fn is None:
        problems.append("Missing required function: TestOneInput")
    else:
        if len(test_fn.args.args) != 1:
            problems.append("TestOneInput must take exactly one argument.")
        else:
            arg_name = test_fn.args.args[0].arg
            if arg_name != "data":
                problems.append("TestOneInput argument should be named `data`.")

    if require_api_marker and not _string_markers_present(code, api_markers):
        problems.append(
            "No obvious target API marker found in generated code; the harness may not call the intended API."
        )

    visitor = _ImportVisitor()
    visitor.visit(tree)
    imported = set(visitor.imported_modules) | set(visitor.from_imports)
    if not any(mod == "tensorflow" or mod.startswith("tensorflow") for mod in imported):
        problems.append("The generated harness does not appear to import TensorFlow.")

    return len(problems) == 0, "\n".join(problems) if problems else "OK"


def build_repair_prompt(
    original_prompt: str,
    bad_response: str,
    bad_code: str,
    validation_error: str,
) -> str:
    return REPAIR_PROMPT_TEMPLATE.format(
        validation_error=validation_error,
        bad_response=bad_response,
        bad_code=bad_code,
        original_prompt=original_prompt,
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    cfg = parse_args()

    yaml_text = read_text(cfg.yaml_path, cfg.max_yaml_chars)
    api_txt_text = read_text(cfg.api_txt_path, cfg.max_api_txt_chars)
    atheris_doc_text = read_text(cfg.atheris_doc_path, cfg.max_atheris_chars)

    api_name, yaml_obj = load_yaml_summary(yaml_text)
    api_markers = build_api_markers(yaml_obj, api_name)
    yaml_summary = build_yaml_hint_summary(yaml_obj)
    prompt = build_prompt(api_name, api_markers, yaml_summary, yaml_text, api_txt_text, atheris_doc_text)

    if cfg.save_prompt:
        write_text(cfg.out_path.with_suffix(cfg.out_path.suffix + ".prompt.txt"), prompt)

    client = create_client(cfg)
    raw_response, used_mode = call_model(client, cfg, DEFAULT_SYSTEM_PROMPT, prompt)
    code = extract_python_code(raw_response)
    ok, validation_msg = static_validate_python(
        code,
        api_markers=api_markers,
        require_instrument_imports=cfg.require_instrument_imports,
        require_api_marker=cfg.require_api_marker,
    )

    repair_round = 0
    while not ok and repair_round < cfg.repair_attempts:
        repair_round += 1
        repair_prompt = build_repair_prompt(prompt, raw_response, code, validation_msg)
        raw_response, used_mode = call_model(client, cfg, DEFAULT_SYSTEM_PROMPT, repair_prompt)
        code = extract_python_code(raw_response)
        ok, validation_msg = static_validate_python(
            code,
            api_markers=api_markers,
            require_instrument_imports=cfg.require_instrument_imports,
            require_api_marker=cfg.require_api_marker,
        )

    if cfg.save_raw_response:
        write_text(cfg.out_path.with_suffix(cfg.out_path.suffix + ".raw.txt"), raw_response)

    if not ok:
        raise HarnessGenerationError(
            "Generated harness failed validation after repair attempts.\n"
            f"Last validation error:\n{validation_msg}"
        )

    banner = textwrap.dedent(
        f"""
        # Auto-generated by llm_harness_codegen.py
        # model={cfg.model}
        # api_mode={used_mode}
        # source_yaml={cfg.yaml_path}
        # source_api_txt={cfg.api_txt_path}
        # source_atheris_doc={cfg.atheris_doc_path}
        # repair_attempts_used={repair_round}

        """
    ).lstrip()

    final_code = banner + code
    write_text(cfg.out_path, final_code)

    print(f"[OK] Wrote harness: {cfg.out_path}")
    print(f"[OK] API name: {api_name}")
    print(f"[OK] LLM mode used: {used_mode}")
    print(f"[OK] Repair rounds used: {repair_round}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
