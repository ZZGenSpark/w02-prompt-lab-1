from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from promptlab.config import Settings  # type: ignore[import-untyped]
from promptlab.day1 import (  # type: ignore[import-untyped]
    CASES_PATH,
    TRUNCATION_OUTPUT_TOKENS,
    call_ollama,
    demonstrate_truncation,
    load_cases,
    main,
    record_from_payload,
)


def mistral_id() -> str:
    return str(Settings.from_env().models["mistral"].model_id)


def settings() -> Settings:
    return Settings.from_env()


def ollama_payload(
    *,
    prompt_eval_count: int = 228,
    eval_count: int = 50,
    done_reason: object = "stop",
    response: object = "ok",
) -> dict[str, Any]:
    return {
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "done_reason": done_reason,
        "response": response,
    }


def test_load_cases_returns_selected_ids_in_order() -> None:
    cases = load_cases(CASES_PATH, ("E12", "E07", "E11"))
    assert [str(case["id"]) for case in cases] == ["E12", "E07", "E11"]
    assert all("source" in case for case in cases)


def test_load_cases_missing_id_raises() -> None:
    with pytest.raises(KeyError, match="Missing extraction cases"):
        load_cases(CASES_PATH, ("E12", "E99"))


def test_load_cases_skips_blank_and_non_object_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        '{"id":"E12","source":"short"}\n'
        "[1, 2]\n"
        '{"id":"E07","source":"middle"}\n',
        encoding="utf-8",
    )
    cases = load_cases(path, ("E12", "E07"))
    assert [str(case["id"]) for case in cases] == ["E12", "E07"]


def test_call_ollama_posts_mistral_generate_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = ollama_payload()
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr("promptlab.day1.httpx.post", mock_post)

    cfg = settings()
    result, latency_ms = call_ollama(cfg, mistral_id(), "hello", 256)

    assert result == payload
    assert isinstance(latency_ms, int)
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == f"{cfg.ollama_base_url}/api/generate"
    assert kwargs["json"] == {
        "model": mistral_id(),
        "prompt": "hello",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }


def test_call_ollama_rejects_non_dict_json(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = ["not", "an", "object"]
    monkeypatch.setattr("promptlab.day1.httpx.post", MagicMock(return_value=mock_response))

    with pytest.raises(TypeError, match="JSON object"):
        call_ollama(settings(), mistral_id(), "hello", 256)


def test_call_ollama_propagates_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "http://example.test/api/generate")
    response = httpx.Response(500, request=request)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=request, response=response
    )
    monkeypatch.setattr("promptlab.day1.httpx.post", MagicMock(return_value=mock_response))

    with pytest.raises(httpx.HTTPStatusError):
        call_ollama(settings(), mistral_id(), "hello", 256)


def test_record_from_payload_maps_mistral_success() -> None:
    record = record_from_payload(
        run_id="run-1",
        model_id=mistral_id(),
        case_id="E12",
        payload=ollama_payload(),
        latency_ms=1972,
    )
    uuid.UUID(record.record_id)
    assert record.run_id == "run-1"
    assert record.provider == "ollama"
    assert record.model_id == mistral_id()
    assert record.task == "extraction"
    assert record.case_id == "E12"
    assert record.prompt_id == "baseline"
    assert record.prompt_version == "v0"
    assert record.input_tokens == 228
    assert record.output_tokens == 50
    assert record.stop_reason == "stop"
    assert record.response_text == "ok"
    assert record.error_type is None
    assert record.cached_input_tokens is None
    assert record.cost_usd == pytest.approx(0.0)
    assert record.timestamp.tzinfo is not None
    assert record.timestamp.utcoffset() is not None


def test_record_from_payload_coerces_non_string_optional_fields() -> None:
    record = record_from_payload(
        run_id="run-1",
        model_id=mistral_id(),
        case_id="E12",
        payload=ollama_payload(done_reason=1, response=None),
        latency_ms=10,
    )
    assert record.stop_reason is None
    assert record.response_text is None


def test_record_from_payload_missing_token_counts_raises() -> None:
    with pytest.raises(KeyError):
        record_from_payload(
            run_id="run-1",
            model_id=mistral_id(),
            case_id="E12",
            payload={"response": "ok"},
            latency_ms=10,
        )


def test_demonstrate_truncation_records_length_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "promptlab.day1.call_ollama",
        lambda *args, **kwargs: (ollama_payload(done_reason="length", eval_count=8), 12),
    )

    demonstrate_truncation(settings(), mistral_id(), "prompt", "run-1")

    demo_path = tmp_path / "runs" / "run-1-truncation.jsonl"
    evidence_path = tmp_path / "runs" / "run-1.jsonl"
    record = json.loads(demo_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["error_type"] == "TruncatedResponseError"
    assert record["stop_reason"] == "length"
    assert record["max_output_tokens"] == TRUNCATION_OUTPUT_TOKENS
    assert record["model_id"] == mistral_id()
    assert not evidence_path.exists()


def test_demonstrate_truncation_stop_has_no_error_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "promptlab.day1.call_ollama",
        lambda *args, **kwargs: (ollama_payload(done_reason="stop"), 12),
    )

    demonstrate_truncation(settings(), mistral_id(), "prompt", "run-1")

    record = json.loads(
        (tmp_path / "runs" / "run-1-truncation.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["error_type"] is None
    assert record["stop_reason"] == "stop"


def test_main_writes_three_mistral_records_and_separate_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_call(
        _settings: Settings, _model_id: str, _prompt: str, max_output_tokens: int
    ) -> tuple[dict[str, Any], int]:
        if max_output_tokens == TRUNCATION_OUTPUT_TOKENS:
            return ollama_payload(done_reason="length", eval_count=8), 10
        return ollama_payload(), 25

    monkeypatch.setattr("promptlab.day1.call_ollama", fake_call)

    main()
    captured = capsys.readouterr().out
    assert f"model_id={mistral_id()}" in captured
    run_id = next(
        line.split("=", 1)[1].split()[0]
        for line in captured.splitlines()
        if line.startswith("run_id=")
    )

    evidence = (tmp_path / "runs" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    truncation = (
        (tmp_path / "runs" / f"{run_id}-truncation.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert [json.loads(line)["case_id"] for line in evidence] == ["E12", "E07", "E11"]
    assert all(json.loads(line)["model_id"] == mistral_id() for line in evidence)
    assert all(json.loads(line)["error_type"] is None for line in evidence)
    assert len(truncation) == 1
    assert json.loads(truncation[0])["error_type"] == "TruncatedResponseError"
