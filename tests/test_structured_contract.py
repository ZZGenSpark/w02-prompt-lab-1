from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.structured import complete_structured


def _day3_run() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "day3_run.py"
    spec = importlib.util.spec_from_file_location("day3_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinySchema(BaseModel):
    value: str


class RepairingStubAdapter:
    provider = "ollama"
    model_id = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        assert run_id == "fixture-run"
        self.calls += 1
        self.requests.append(request)

        text = '{"wrong":"shape"}' if self.calls == 1 else '{"value":"fixed"}'
        return CompletionResult(
            succeeded=True,
            text=text,
            error_type=None,
            records=[],
        )


def _request() -> CompletionRequest:
    return CompletionRequest(
        task="summarization",
        case_id="S00",
        prompt_id="summarize",
        prompt_version="v1",
        system="",
        user_content="Summarize the supplied procedure.",
        temperature=0.0,
        max_output_tokens=128,
    )


def test_complete_structured_repairs_once() -> None:
    adapter = RepairingStubAdapter()

    result = complete_structured(
        adapter,
        _request(),
        TinySchema,
        "fixture-run",
        max_repairs=1,
    )

    assert result == TinySchema(value="fixed")
    assert adapter.calls == 2


def test_repair_request_carries_validation_context() -> None:
    adapter = RepairingStubAdapter()

    complete_structured(
        adapter,
        _request(),
        TinySchema,
        "fixture-run",
        max_repairs=1,
    )

    repair_text = adapter.requests[1].user_content.lower()
    assert "validation" in repair_text or "field required" in repair_text
    assert "value" in repair_text


HEADING_SOURCE = (
    "1. Document Control\n"
    "Title: Card Dispute Intake Procedure.\n"
    "2. Purpose\n"
    "This procedure standardizes intake.\n"
)


def test_citation_must_be_full_heading_not_numeral() -> None:
    runner = _day3_run()
    headings = runner.section_headings(HEADING_SOURCE)

    assert "1. Document Control" in headings
    assert "1" not in headings
    assert runner.citation_is_heading("1. Document Control", headings)
    assert not runner.citation_is_heading("1", headings)


def test_present_fields_fail_when_citation_is_only_a_number() -> None:
    runner = _day3_run()
    output = {
        "document_status": "valid",
        "title": {
            "value": "Card Dispute Intake Procedure",
            "status": "present",
            "citation": "1",
        },
        "version": {"value": None, "status": "absent", "citation": None},
        "purpose": {
            "value": "This procedure standardizes intake.",
            "status": "present",
            "citation": "2. Purpose",
        },
        "required_steps": {"value": None, "status": "absent", "citation": None},
        "exceptions": {"value": None, "status": "absent", "citation": None},
        "effective_date": {"value": None, "status": "absent", "citation": None},
    }

    assert runner.present_citation_failures(output, HEADING_SOURCE) == 1


def test_example_leakage_detects_few_shot_proper_nouns() -> None:
    runner = _day3_run()
    assert runner.example_leakage_count({"policy_name": {"value": "Northglass"}}) == 1
    assert runner.example_leakage_count({"policy_name": {"value": "Pennsylvania"}}) == 0
