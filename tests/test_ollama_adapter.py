from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)


def _model_id(name: str) -> str:
    return Settings.from_env().models[name].model_id


def _request() -> CompletionRequest:
    return CompletionRequest(
        task="summarization",
        case_id="S01",
        prompt_id="baseline",
        prompt_version="v0",
        system="You are a helpful assistant.",
        user_content="Summarize this procedure.",
        temperature=0.0,
        max_output_tokens=64,
    )


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "ok",
        prompt_tokens: int = 11,
        output_tokens: int = 4,
        done_reason: str = "stop",
        include_response: bool = True,
    ) -> None:
        self.status_code = status_code
        self.text = text
        payload: dict[str, object] = {
            "message": {"content": text},
            "prompt_eval_count": prompt_tokens,
            "eval_count": output_tokens,
            "done_reason": done_reason,
        }
        if include_response:
            payload["response"] = text
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


def test_success_maps_generate_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(
            text="mapped summary",
            prompt_tokens=40,
            output_tokens=9,
            done_reason="stop",
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OllamaAdapter(model_id=_model_id("mistral")).complete(_request(), "map-run")

    assert calls == 1
    assert result.succeeded is True
    assert result.text == "mapped summary"
    assert result.error_type is None
    assert len(result.records) == 1

    record = result.records[0]
    assert record.provider == "ollama"
    assert record.model_id == _model_id("mistral")
    assert record.case_id == "S01"
    assert record.task == "summarization"
    assert record.attempt == 1
    assert record.input_tokens == 40
    assert record.output_tokens == 9
    assert record.stop_reason == "stop"
    assert record.response_text == "mapped summary"
    assert record.cost_usd == pytest.approx(0.0)
    assert record.error_type is None


def test_success_falls_back_to_message_content_when_response_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(text="from chat message", include_response=False)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = OllamaAdapter(model_id=_model_id("qwen")).complete(_request(), "chat-shape-run")

    assert result.succeeded is True
    assert result.text == "from chat message"
    assert result.records[0].response_text == "from chat message"
    assert result.records[0].stop_reason == "stop"


def test_http_500_then_success_retries_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0
    slept: list[float] = []

    def flaky_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse(status_code=500, text="server error")
        return FakeResponse(text="recovered")

    monkeypatch.setattr(httpx, "post", flaky_post)
    monkeypatch.setattr("time.sleep", lambda seconds: slept.append(float(seconds)))

    result = OllamaAdapter(model_id=_model_id("mistral")).complete(_request(), "http500-run")

    assert calls == 2
    assert slept != []
    assert result.succeeded is True
    assert result.text == "recovered"
    assert [record.attempt for record in result.records] == [1, 2]
    assert result.records[0].error_type == TransientProviderError.__name__
    assert result.records[1].error_type is None


def test_timeout_is_transient_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0

    def flaky_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.TimeoutException("timed out")
        return FakeResponse(text="after timeout")

    monkeypatch.setattr(httpx, "post", flaky_post)
    monkeypatch.setattr("time.sleep", lambda _: None)

    result = OllamaAdapter(model_id=_model_id("mistral")).complete(_request(), "timeout-run")

    assert calls == 2
    assert result.succeeded is True
    assert [record.attempt for record in result.records] == [1, 2]
    assert result.records[0].error_type == TransientProviderError.__name__
    assert result.records[0].response_text is None
    assert result.records[1].error_type is None


def test_three_transient_failures_stop_at_attempt_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0

    def always_down(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("still down")

    monkeypatch.setattr(httpx, "post", always_down)
    monkeypatch.setattr("time.sleep", lambda _: None)

    result = OllamaAdapter(model_id=_model_id("mistral")).complete(_request(), "cap-run")

    assert calls == 3
    assert result.succeeded is False
    assert result.error_type == TransientProviderError.__name__
    assert [record.attempt for record in result.records] == [1, 2, 3]
    assert all(record.error_type == TransientProviderError.__name__ for record in result.records)


def test_http_400_is_permanent_and_does_not_sleep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0
    slept: list[float] = []

    def bad_request(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(status_code=400, text="bad request")

    monkeypatch.setattr(httpx, "post", bad_request)
    monkeypatch.setattr("time.sleep", lambda seconds: slept.append(float(seconds)))

    result = OllamaAdapter(model_id=_model_id("mistral")).complete(_request(), "http400-run")

    assert calls == 1
    assert slept == []
    assert result.succeeded is False
    assert result.error_type == PermanentProviderError.__name__
    assert result.records[0].attempt == 1
    assert result.records[0].error_type == PermanentProviderError.__name__


def test_truncation_keeps_partial_text_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0

    def truncated(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(
            text="partial output",
            prompt_tokens=21,
            output_tokens=64,
            done_reason="length",
        )

    monkeypatch.setattr(httpx, "post", truncated)

    result = OllamaAdapter(model_id=_model_id("qwen")).complete(_request(), "length-run")

    assert calls == 1
    assert result.succeeded is False
    assert result.text == "partial output"
    assert result.error_type == TruncatedResponseError.__name__
    record = result.records[0]
    assert record.attempt == 1
    assert record.stop_reason == "length"
    assert record.input_tokens == 21
    assert record.output_tokens == 64
    assert record.response_text == "partial output"
    assert record.error_type == TruncatedResponseError.__name__


def test_unknown_model_is_recorded_once_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0

    def unexpected_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(text="should not be called")

    monkeypatch.setattr(httpx, "post", unexpected_post)
    monkeypatch.setattr("time.sleep", lambda _: None)

    result = OllamaAdapter(model_id="not-a-configured-model").complete(
        _request(),
        "unknown-model-run",
    )

    assert calls == 0
    assert result.succeeded is False
    assert result.error_type == UnknownModelError.__name__
    assert len(result.records) == 1
    assert result.records[0].attempt == 1
    assert result.records[0].model_id == "not-a-configured-model"
    assert result.records[0].error_type == UnknownModelError.__name__
