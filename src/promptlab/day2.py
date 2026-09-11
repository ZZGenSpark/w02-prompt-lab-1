"""Day 2 runner: baseline summarization on both configured local models."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings

CASE_IDS = tuple(f"S{index:02d}" for index in range(1, 13))
MAX_OUTPUT_TOKENS = 1024
CASES_PATH = PROJECT_ROOT / "cases" / "summarization.jsonl"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"
DOCUMENT_PLACEHOLDER = "{document_text}"


def load_cases(path: Path, case_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted = set(case_ids)
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            continue
        case_id = case["id"]
        if case_id in wanted:
            found[str(case_id)] = case
    missing = [case_id for case_id in case_ids if case_id not in found]
    if missing:
        raise KeyError(f"Missing summarization cases: {', '.join(missing)}")
    return [found[case_id] for case_id in case_ids]


def build_request(
    case: Mapping[str, Any],
    prompt_template: str,
    settings: Settings,
) -> CompletionRequest:
    if DOCUMENT_PLACEHOLDER not in prompt_template:
        raise ValueError("Prompt template is missing {document_text} placeholder")
    document_text = str(case["source"])
    filled = prompt_template.replace(DOCUMENT_PLACEHOLDER, document_text)
    system, separator, remainder = filled.partition("<document>")
    user_content = f"{separator}{remainder}" if separator else filled
    return CompletionRequest(
        task="summarization",
        case_id=str(case["id"]),
        prompt_id="baseline",
        prompt_version="v0",
        system=system.strip(),
        user_content=user_content.strip(),
        temperature=settings.temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def main() -> None:
    settings = Settings.from_env()
    run_id = str(uuid.uuid4())
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    cases = load_cases(CASES_PATH, CASE_IDS)
    adapters = [
        OllamaAdapter(model_id=config.model_id) for config in settings.models.values()
    ]

    print(f"run_id={run_id}")
    for case in cases:
        request = build_request(case, prompt_template, settings)
        for adapter in adapters:
            result = adapter.complete(request, run_id)
            print(
                f"{request.case_id} model_id={adapter.model_id} "
                f"succeeded={result.succeeded} error_type={result.error_type}"
            )


if __name__ == "__main__":
    main()
