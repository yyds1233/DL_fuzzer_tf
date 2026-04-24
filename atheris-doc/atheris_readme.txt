#!/usr/bin/env python3
"""
LLM-driven harness generator for DL_fuzzer_tf.

Reads:
  1) api yaml file
  2) api txt doc file
  3) Atheris doc / notes file

Then calls an OpenAI-compatible LLM endpoint to generate a standalone,
executable Atheris fuzz harness (.py), extracts Python code from the model
response, validates it, optionally asks the model to repair it once, and
writes the final harness to disk.

Example:
  python llm_harness_codegen.py \
    --yaml build_yaml/out/tf.bitwise.bitwise_xor.yaml \
    --api-txt api_txt_50/tf.bitwise.bitwise_xor.txt \
    --atheris-doc docs/atheris_readme.txt \
    --out fuzz_output/tf.bitwise.bitwise_xor_harness.py \
    --model gpt-5.4

Environment variables:
  OPENAI_API_KEY      API key for OpenAI-compatible endpoint.
  OPENAI_BASE_URL     Optional base url for OpenAI-compatible endpoint.
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
from typing import Optional, Tuple

import yaml

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: openai\n"
        "Install with: pip install openai pyyaml"
    ) from exc


DEFAULT_SYSTEM_PROMPT = """
You are an expert Python fuzzing engineer.
You generate high-quality, executable Atheris harnesses for TensorFlow APIs.

Hard requirements:
1. Output exactly one Python code block and nothing else.
2. The harness must be a complete standalone .py file.
3. Import atheris first.
4. Use `with atheris.instrument_imports():` for modules that should be instrumented.
5. Define `TestOneInput(data: bytes)`.
6. Call `atheris.Setup(sys.argv, TestOneInput)` before `atheris.Fuzz()`.
7. Use `atheris.FuzzedDataProvider(data)` to decode bytes into arguments.
8. The harness must actually call the target TensorFlow API named in the provided files.
9. Prefer generating valid TensorFlow tensors/values that respect the YAML constraints.
10. Swallow only expected/benign input-validation exceptions. Never swallow KeyboardInterrupt, SystemExit, MemoryError, or AssertionError.
11. Avoid placeholder code, TODOs, pseudo code, markdown, and explanations.
12. Do not depend on any local project helper modules unless they are explicitly included in the provided files.
13. Use deterministic helper functions inside the same file when needed.
14. The output must be syntactically valid Python 3.
""".strip()


PROMPT_TEMPLATE = """
Generate one executable Python Atheris harness for the TensorFlow API described below.

Goal:
- Produce a coverage-guided fuzz harness.
- Use the YAML as the source of truth for parameters/constraints/dtypes/ranks.
- Use the API txt as semantic/API behavior context.
- Use the Atheris doc as the source of truth for harness structure.

Output contract:
- Return exactly one fenced Python code block.
- No prose before or after the code block.
- The file must be directly runnable with `python harness.py` after dependencies are installed.

Implementation guidance:
- Target API name: {api_name}
- Prefer small tensor sizes to keep execution cheap, unless the YAML requires otherwise.
- Respect dtype relations, rank hints, and shape variables from YAML.
- If multiple parameters must share dtype/shape compatibility, enforce that.
- If the API has attributes with allowed_values, sample only from those values.
- If an argument is optional and the docs do not require it, it may be omitted.
- Use helper functions inside the harness to decode booleans, ints, floats, strings, shapes, dtypes, and tensors from FuzzedDataProvider.
- The harness should avoid obviously invalid values when YAML/doc hints give a valid domain.
- Expected user/input validation exceptions may be caught and ignored so fuzzing can continue.
- The harness should not print noisy logs on every iteration.
- Add a small `main()` section or module tail that starts Atheris.

==== API YAML BEGIN ====
{yaml_text}
==== API YAML END ====

==== API TXT BEGIN ====
{api_txt_text}
==== API TXT END ====

==== ATHERIS DOC BEGIN ====
{atheris_doc_text}
==== ATHERIS DOC END ====
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


class HarnessGenerationError(RuntimeError):
    pass


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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--max-yaml-chars", type=int, default=80000)
    parser.add_argument("--max-api-txt-chars", type=int, default=120000)
    parser.add_argument("--max-atheris-chars", type=int, default=120000)
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
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
    return (
        f"{head}\n\n"
        f"[... TRUNCATED {omitted} CHARS ...]\n\n"
        f"{tail}"
    )


def load_yaml_summary(yaml_text: str) -> Tuple[str, dict]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise HarnessGenerationError("YAML root must be a mapping/object")
    api_name = str(data.get("api_name") or "UNKNOWN_API")
    return api_name, data


def build_prompt(api_name: str, yaml_text: str, api_txt_text: str, atheris_doc_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        api_name=api_name,
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

    # fallback extraction for odd gateway/proxy behavior
    try:
        return json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
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

    # Sometimes the model returns JSON like {"code": "..."}
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

    # Last fallback: use the whole response as code.
    return stripped + ("\n" if stripped else "")


EXPECTED_MARKERS = [
    "import atheris",
    "TestOneInput",
    "atheris.Setup",
    "atheris.Fuzz",
]


def static_validate_python(code: str) -> Tuple[bool, str]:
    problems = []

    if not code.strip():
        return False, "Empty code generated."

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})"

    for marker in EXPECTED_MARKERS:
        if marker not in code:
            problems.append(f"Missing required marker: {marker}")

    return len(problems) == 0, "\n".join(problems) if problems else "OK"


def build_repair_prompt(
    original_prompt: str,
    bad_response: str,
    bad_code: str,
    validation_error: str,
) -> str:
    return textwrap.dedent(
        f"""
        Your previous output did not pass validation.

        Validation error:
        {validation_error}

        Previous raw response:
        ===== BEGIN BAD RESPONSE =====
        {bad_response}
        ===== END BAD RESPONSE =====

        Extracted code:
        ===== BEGIN BAD CODE =====
        {bad_code}
        ===== END BAD CODE =====

        Please fix the harness.

        Requirements again:
        - Return exactly one fenced Python code block.
        - Must be syntactically valid Python 3.
        - Must import atheris first.
        - Must define TestOneInput(data: bytes).
        - Must call atheris.Setup(sys.argv, TestOneInput) before atheris.Fuzz().
        - Must target the API described in the original prompt.

        Original task again:
        ===== BEGIN ORIGINAL TASK =====
        {original_prompt}
        ===== END ORIGINAL TASK =====
        """
    ).strip()


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

    api_name, _yaml_obj = load_yaml_summary(yaml_text)
    prompt = build_prompt(api_name, yaml_text, api_txt_text, atheris_doc_text)

    if cfg.save_prompt:
        write_text(cfg.out_path.with_suffix(cfg.out_path.suffix + ".prompt.txt"), prompt)

    client = create_client(cfg)
    raw_response, used_mode = call_model(client, cfg, DEFAULT_SYSTEM_PROMPT, prompt)
    code = extract_python_code(raw_response)
    ok, validation_msg = static_validate_python(code)

    repair_round = 0
    while not ok and repair_round < cfg.repair_attempts:
        repair_round += 1
        repair_prompt = build_repair_prompt(prompt, raw_response, code, validation_msg)
        raw_response, used_mode = call_model(client, cfg, DEFAULT_SYSTEM_PROMPT, repair_prompt)
        code = extract_python_code(raw_response)
        ok, validation_msg = static_validate_python(code)

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

        """
    ).lstrip()

    final_code = banner + code
    write_text(cfg.out_path, final_code)

    print(f"[OK] Wrote harness: {cfg.out_path}")
    print(f"[OK] API name: {api_name}")
    print(f"[OK] LLM mode used: {used_mode}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
