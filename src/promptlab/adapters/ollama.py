from __future__ import annotations

import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)
from promptlab.usage import CallRecord, append_record, compute_cost

MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 180.0


class OllamaAdapter:
    provider = "ollama"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._base_url = Settings.from_env().ollama_base_url

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        if not self._model_is_configured():
            record = self._record(
                request=request,
                run_id=run_id,
                attempt=1,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                stop_reason=None,
                error_type=UnknownModelError.__name__,
                response_text=None,
                cost_usd=0.0,
            )
            append_record(record, run_id)
            return CompletionResult(
                succeeded=False,
                text=None,
                error_type=UnknownModelError.__name__,
                records=[record],
            )

        records: list[CallRecord] = []
        last_text: str | None = None
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record, error, text = self._one_attempt(request, run_id, attempt)
            records.append(record)
            append_record(record, run_id)
            last_text = text
            last_error = error
            if error is None:
                return CompletionResult(
                    succeeded=True,
                    text=text,
                    error_type=None,
                    records=records,
                )
            if not isinstance(error, TransientProviderError) or attempt == MAX_ATTEMPTS:
                return CompletionResult(
                    succeeded=False,
                    text=text,
                    error_type=type(error).__name__,
                    records=records,
                )
            time.sleep((2 ** (attempt - 1)) + random.uniform(0, 1))

        return CompletionResult(
            succeeded=False,
            text=last_text,
            error_type=type(last_error).__name__ if last_error is not None else None,
            records=records,
        )

    def _model_is_configured(self) -> bool:
        try:
            compute_cost(self.model_id, 0, 0)
        except UnknownModelError:
            return False
        return True

    def _one_attempt(
        self,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
    ) -> tuple[CallRecord, Exception | None, str | None]:
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": f"{request.system}\n\n{request.user_content}",
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_output_tokens,
                    },
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.TransportError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            error: Exception = TransientProviderError(str(exc))
            return (
                self._record_from_failure(request, run_id, attempt, latency_ms, error),
                error,
                None,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        status_code = response.status_code
        if status_code == 429 or status_code >= 500:
            error = TransientProviderError(f"HTTP {status_code}")
            return (
                self._record_from_failure(request, run_id, attempt, latency_ms, error),
                error,
                None,
            )
        if 400 <= status_code < 500:
            error = PermanentProviderError(f"HTTP {status_code}")
            return (
                self._record_from_failure(request, run_id, attempt, latency_ms, error),
                error,
                None,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            error = PermanentProviderError(str(exc))
            return (
                self._record_from_failure(request, run_id, attempt, latency_ms, error),
                error,
                None,
            )
        if not isinstance(payload, dict):
            error = PermanentProviderError("Ollama response was not a JSON object")
            return (
                self._record_from_failure(request, run_id, attempt, latency_ms, error),
                error,
                None,
            )

        text = _extract_text(payload)
        stop_reason = _extract_stop_reason(payload)
        token_counts = _extract_token_counts(payload)
        if token_counts is None:
            error = PermanentProviderError("Ollama response was missing token counts")
            return (
                self._record_from_failure(request, run_id, attempt, latency_ms, error),
                error,
                text,
            )
        input_tokens, output_tokens = token_counts
        if stop_reason == "length":
            error = TruncatedResponseError("output token ceiling reached")
            return (
                self._record(
                    request=request,
                    run_id=run_id,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stop_reason=stop_reason,
                    error_type=type(error).__name__,
                    response_text=text,
                ),
                error,
                text,
            )
        return (
            self._record(
                request=request,
                run_id=run_id,
                attempt=attempt,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=stop_reason,
                error_type=None,
                response_text=text,
            ),
            None,
            text,
        )

    def _record_from_failure(
        self,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
        latency_ms: int,
        error: Exception,
    ) -> CallRecord:
        return self._record(
            request=request,
            run_id=run_id,
            attempt=attempt,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            stop_reason=None,
            error_type=type(error).__name__,
            response_text=None,
        )

    def _record(
        self,
        *,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str | None,
        error_type: str | None,
        response_text: str | None,
        cost_usd: float | None = None,
    ) -> CallRecord:
        if cost_usd is None:
            cost_usd = compute_cost(self.model_id, input_tokens, output_tokens)
        return CallRecord(
            record_id=str(uuid.uuid4()),
            run_id=run_id,
            timestamp=datetime.now(UTC),
            provider="ollama",
            model_id=self.model_id,
            task=request.task,
            case_id=request.case_id,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            attempt=attempt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            stop_reason=stop_reason,
            error_type=error_type,
            response_text=response_text,
        )


def _extract_text(payload: dict[str, Any]) -> str | None:
    response_text = payload.get("response")
    if isinstance(response_text, str):
        return response_text
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def _extract_stop_reason(payload: dict[str, Any]) -> str | None:
    done_reason = payload.get("done_reason")
    return done_reason if isinstance(done_reason, str) else None


def _extract_token_counts(payload: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(payload["prompt_eval_count"]), int(payload["eval_count"])
    except (KeyError, TypeError, ValueError):
        return None
