"""Run Day 4 triage v1 and v2 under one run_id."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.prompts import load, render_user
from promptlab.records import OutputRecord, append_record
from promptlab.schemas import (
    TriageOutput,
    TriageOutputWithAnalysis,
    schema_description,
)
from promptlab.scoring import score_triage_case
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord

CASES_PATH = PROJECT_ROOT / "cases" / "triage.jsonl"
GOLD_PATH = PROJECT_ROOT / "cases" / "gold" / "triage.jsonl"
DOCS_RUN_PATH = PROJECT_ROOT / "docs" / "day4-run.jsonl"
DOCS_SCORE_PATH = PROJECT_ROOT / "docs" / "day4-scores.jsonl"
MAX_OUTPUT_TOKENS = 1024


class CountingAdapter:
    """Count adapter.complete calls so the runner can record repair attempts."""

    provider = "ollama"

    def __init__(self, inner: ModelAdapter) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.calls = 0
        self.records: list[CallRecord] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        result = self._inner.complete(request, run_id)
        self.records.extend(result.records)
        return result

    def reset(self) -> None:
        self.calls = 0
        self.records = []


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in load_jsonl(path)}


def run_case(
    *,
    adapter: CountingAdapter,
    case: dict[str, Any],
    prompt_version: str,
    schema: type[BaseModel],
    run_id: str,
    model_name: str,
    temperature: float,
    max_repairs: int,
) -> OutputRecord:
    adapter.reset()
    template = load("triage", prompt_version)
    case_id = str(case["id"])
    request = CompletionRequest(
        task="triage",
        case_id=case_id,
        prompt_id=template.prompt_id,
        prompt_version=template.version,
        system=template.system,
        user_content=render_user(
            template,
            variables={
                "case_id": case_id,
                "schema_description": schema_description(schema),
            },
            untrusted=str(case["source"]),
        ),
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    error: str | None = None
    output: dict[str, Any] | None = None
    succeeded = False
    try:
        parsed = complete_structured(
            adapter,
            request,
            schema,
            run_id,
            max_repairs=max_repairs,
        )
        output = parsed.model_dump()
        succeeded = True
    except (ValidationError, ValueError, RuntimeError) as exc:
        error = str(exc)

    return OutputRecord(
        run_id=run_id,
        task="triage",
        case_id=case_id,
        model_name=model_name,
        model_id=adapter.model_id,
        prompt_version=prompt_version,
        succeeded=succeeded,
        repairs=max(adapter.calls - 1, 0),
        output=output,
        error=error,
    )


def main() -> None:
    settings = Settings.from_env()
    model = settings.models["mistral"]
    run_id = str(uuid.uuid4())
    adapter = CountingAdapter(OllamaAdapter(model.model_id))
    cases = load_jsonl(CASES_PATH)
    gold = load_gold(GOLD_PATH)

    jobs: list[tuple[str, type[BaseModel]]] = [
        ("v1", TriageOutput),
        ("v2", TriageOutputWithAnalysis),
    ]

    if DOCS_RUN_PATH.exists():
        DOCS_RUN_PATH.unlink()
    if DOCS_SCORE_PATH.exists():
        DOCS_SCORE_PATH.unlink()

    print(f"run_id={run_id} model={model.model_id} temperature={settings.temperature}")
    for prompt_version, schema in jobs:
        for case in cases:
            record = run_case(
                adapter=adapter,
                case=case,
                prompt_version=prompt_version,
                schema=schema,
                run_id=run_id,
                model_name=model.logical_name,
                temperature=settings.temperature,
                max_repairs=settings.max_schema_repairs,
            )
            append_record(DOCS_RUN_PATH, record)
            for score in score_triage_case(record, gold[record.case_id]):
                append_record(DOCS_SCORE_PATH, score)
            status = "ok" if record.succeeded else "fail"
            print(
                f"{record.prompt_version} {record.case_id} {status} "
                f"repairs={record.repairs}",
                flush=True,
            )


if __name__ == "__main__":
    main()
