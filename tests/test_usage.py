from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from promptlab.config import Settings
from promptlab.errors import UnknownModelError
from promptlab.usage import CallRecord, append_record, compute_cost


def mistral_id() -> str:
    return str(Settings.from_env().models["mistral"].model_id)


def qwen_id() -> str:
    return str(Settings.from_env().models["qwen"].model_id)


def make_record(*, model_id: str | None = None, run_id: str = "usage-test") -> CallRecord:
    return CallRecord(
        record_id="00000000-0000-4000-8000-000000000001",
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id or mistral_id(),
        task="extraction",
        case_id="E12",
        prompt_id="baseline",
        prompt_version="v0",
        attempt=1,
        temperature=0.0,
        max_output_tokens=256,
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=None,
        latency_ms=250,
        cost_usd=0.0,
        stop_reason="stop",
        error_type=None,
        response_text="example",
    )


def test_compute_cost_mistral_is_zero() -> None:
    assert compute_cost(mistral_id(), input_tokens=1234, output_tokens=567) == pytest.approx(0.0)


def test_compute_cost_qwen_is_zero() -> None:
    assert compute_cost(qwen_id(), input_tokens=1234, output_tokens=567) == pytest.approx(0.0)


def test_compute_cost_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        compute_cost("not-a-configured-model", input_tokens=10, output_tokens=10)


def test_call_record_accepts_mistral_extraction() -> None:
    record = make_record()
    assert record.provider == "ollama"
    assert record.task == "extraction"
    assert record.model_id == mistral_id()


def test_call_record_rejects_invalid_provider() -> None:
    with pytest.raises(ValidationError):
        CallRecord.model_validate(
            {
                "record_id": "x",
                "run_id": "x",
                "timestamp": datetime.now(UTC),
                "provider": "openai",
                "model_id": mistral_id(),
                "task": "extraction",
                "case_id": "E12",
                "prompt_id": "baseline",
                "prompt_version": "v0",
                "attempt": 1,
                "temperature": 0.0,
                "max_output_tokens": 256,
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_input_tokens": None,
                "latency_ms": 1,
                "cost_usd": 0.0,
                "stop_reason": None,
                "error_type": None,
                "response_text": None,
            }
        )


def test_call_record_rejects_invalid_task() -> None:
    with pytest.raises(ValidationError):
        CallRecord.model_validate(
            {
                "record_id": "x",
                "run_id": "x",
                "timestamp": datetime.now(UTC),
                "provider": "ollama",
                "model_id": mistral_id(),
                "task": "classification",
                "case_id": "E12",
                "prompt_id": "baseline",
                "prompt_version": "v0",
                "attempt": 1,
                "temperature": 0.0,
                "max_output_tokens": 256,
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_input_tokens": None,
                "latency_ms": 1,
                "cost_usd": 0.0,
                "stop_reason": None,
                "error_type": None,
                "response_text": None,
            }
        )


def test_append_record_creates_and_appends_utc_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    record = make_record()

    append_record(record, "usage-test")
    append_record(record, "usage-test")

    path = tmp_path / "runs" / "usage-test.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["case_id"] == "E12"
    assert first["model_id"] == mistral_id()
    timestamp = first["timestamp"]
    assert timestamp.endswith("Z") or timestamp.endswith("+00:00")


def test_append_record_does_not_rewrite_other_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    first = make_record(run_id="run-a")
    second = make_record(run_id="run-b")

    append_record(first, "run-a")
    append_record(second, "run-b")

    run_a = (tmp_path / "runs" / "run-a.jsonl").read_text(encoding="utf-8").splitlines()
    run_b = (tmp_path / "runs" / "run-b.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(run_a) == 1
    assert len(run_b) == 1
    assert json.loads(run_a[0])["run_id"] == "run-a"
    assert json.loads(run_b[0])["run_id"] == "run-b"
