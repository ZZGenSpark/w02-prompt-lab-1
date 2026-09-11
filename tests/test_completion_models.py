from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.usage import CallRecord

TASKS = ("triage", "summarization", "extraction")


def _request_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "summarization",
        "case_id": "S01",
        "prompt_id": "baseline",
        "prompt_version": "v0",
        "system": "You are a helpful assistant.",
        "user_content": "Summarize this procedure.",
        "temperature": 0.0,
        "max_output_tokens": 256,
    }
    payload.update(overrides)
    return payload


def _call_record() -> CallRecord:
    return CallRecord(
        record_id="00000000-0000-4000-8000-000000000001",
        run_id="completion-models-test",
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id="fixture-model",
        task="summarization",
        case_id="S01",
        prompt_id="baseline",
        prompt_version="v0",
        attempt=1,
        temperature=0.0,
        max_output_tokens=256,
        input_tokens=10,
        output_tokens=4,
        cached_input_tokens=None,
        latency_ms=12,
        cost_usd=0.0,
        stop_reason="stop",
        error_type=None,
        response_text="ok",
    )


@pytest.mark.parametrize("task", TASKS)
def test_completion_request_accepts_each_configured_task(task: str) -> None:
    request = CompletionRequest(**_request_kwargs(task=task))

    assert request.task == task
    assert request.case_id == "S01"
    assert request.user_content == "Summarize this procedure."
    assert request.temperature == 0.0
    assert request.max_output_tokens == 256


def test_completion_result_accepts_successful_call_records() -> None:
    record = _call_record()
    result = CompletionResult(
        succeeded=True,
        text="ok",
        error_type=None,
        records=[record],
    )

    assert result.succeeded is True
    assert result.text == "ok"
    assert result.error_type is None
    assert result.records == [record]
    assert isinstance(result.records[0], CallRecord)


def test_completion_request_rejects_invalid_task() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(**_request_kwargs(task="classification"))


@pytest.mark.parametrize("missing_field", ["user_content", "max_output_tokens", "task", "case_id"])
def test_completion_request_rejects_missing_required_field(missing_field: str) -> None:
    payload = _request_kwargs()
    del payload[missing_field]

    with pytest.raises(ValidationError):
        CompletionRequest(**payload)


def test_completion_result_rejects_missing_succeeded() -> None:
    with pytest.raises(ValidationError):
        CompletionResult.model_validate({"text": "ok", "error_type": None, "records": []})
