"""Run Day 3 structured summarization and extraction under one run_id."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.records import OutputRecord, append_record
from promptlab.schemas import PolicyExtraction, SummarizationOutput, TaskName, schema_description
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord

CASES_DIR = PROJECT_ROOT / "cases"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
DOCS_RUN_PATH = PROJECT_ROOT / "docs" / "day3-run.jsonl"
MAX_OUTPUT_TOKENS = 1024
HEADING_LINE = re.compile(r"^(?:\d+\.\s+\S.*|#{1,6}\s+\S.*)$")
EXAMPLE_ONLY_STRINGS: tuple[str, ...] = (
    "Northglass",
    "Norwyn",
    "Bellwater",
    "Redhaven",
    "East Kestrel",
)


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


def section_headings(source: str) -> frozenset[str]:
    """Return exact section-heading lines from a source document."""
    headings: set[str] = set()
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not HEADING_LINE.match(line):
            continue
        headings.add(line.lstrip("#").strip())
        headings.add(line)
    return frozenset(headings)


def citation_is_heading(citation: str | None, headings: frozenset[str]) -> bool:
    """True when every citation part equals a real heading, not a bare number."""
    if citation is None or not citation.strip():
        return False
    parts = [part.strip() for part in citation.split(";") if part.strip()]
    return bool(parts) and all(part in headings for part in parts)


def present_citation_failures(output: dict[str, Any], source: str) -> int:
    """Count status=present evidence fields whose citation is not a heading."""
    headings = section_headings(source)
    failures = 0
    for value in output.values():
        if not isinstance(value, dict) or value.get("status") != "present":
            continue
        citation = value.get("citation")
        if not isinstance(citation, str) or not citation_is_heading(citation, headings):
            failures += 1
    return failures


def example_leakage_count(output: dict[str, Any] | None) -> int:
    """Count distinctive extract.v2 example strings found in one output."""
    if output is None:
        return 0
    blob = json.dumps(output)
    return sum(1 for marker in EXAMPLE_ONLY_STRINGS if marker in blob)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def render_prompt(template: str, document_text: str, schema: type[BaseModel]) -> str:
    return template.replace("{document_text}", document_text, 1).replace(
        "{schema_description}", schema_description(schema), 1
    )


def run_case(
    *,
    adapter: CountingAdapter,
    case: dict[str, Any],
    task: TaskName,
    prompt_id: str,
    prompt_version: str,
    template: str,
    schema: type[BaseModel],
    run_id: str,
    model_name: str,
    temperature: float,
    max_repairs: int,
) -> OutputRecord:
    adapter.reset()
    request = CompletionRequest(
        task=task,
        case_id=str(case["id"]),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        system="",
        user_content=render_prompt(template, str(case["source"]), schema),
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
        task=task,
        case_id=str(case["id"]),
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
    summarize_template = (PROMPTS_DIR / "summarize.v1.md").read_text(encoding="utf-8")
    extract_template = (PROMPTS_DIR / "extract.v2.md").read_text(encoding="utf-8")

    if DOCS_RUN_PATH.exists():
        DOCS_RUN_PATH.unlink()

    jobs: list[tuple[TaskName, str, str, str, type[BaseModel]]] = [
        (
            "summarization",
            "summarize",
            "v1",
            summarize_template,
            SummarizationOutput,
        ),
        (
            "extraction",
            "extract",
            "v2",
            extract_template,
            PolicyExtraction,
        ),
    ]

    print(f"run_id={run_id} model={model.model_id} temperature={settings.temperature}")
    summarization_repairs = 0
    extraction_repairs = 0
    summarization_ok = 0
    extraction_ok = 0
    leakage = 0
    citation_failures = 0
    for task, prompt_id, prompt_version, template, schema in jobs:
        cases = load_cases(CASES_DIR / f"{task}.jsonl")
        for case in cases:
            record = run_case(
                adapter=adapter,
                case=case,
                task=task,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                template=template,
                schema=schema,
                run_id=run_id,
                model_name=model.logical_name,
                temperature=settings.temperature,
                max_repairs=settings.max_schema_repairs,
            )
            append_record(DOCS_RUN_PATH, record)
            if task == "summarization":
                summarization_repairs += int(record.repairs > 0)
                summarization_ok += int(record.succeeded)
            else:
                extraction_repairs += int(record.repairs > 0)
                extraction_ok += int(record.succeeded)
                leakage += example_leakage_count(record.output)
            if record.output is not None:
                citation_failures += present_citation_failures(
                    record.output, str(case["source"])
                )
            status = "ok" if record.succeeded else "fail"
            print(
                f"{record.case_id} {status} repairs={record.repairs}",
                flush=True,
            )
    print(
        "summary "
        f"summarization_ok={summarization_ok}/12 "
        f"summarization_repair_rate={summarization_repairs}/12 "
        f"extraction_ok={extraction_ok}/12 "
        f"extraction_repair_rate={extraction_repairs}/12 "
        f"example_leakage={leakage} "
        f"citation_failures={citation_failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
