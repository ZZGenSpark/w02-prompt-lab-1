from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, ModelAdapter

_FENCE_PREFIX = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_SUFFIX = re.compile(r"\s*```$")


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_PREFIX.sub("", stripped, count=1)
        stripped = _FENCE_SUFFIX.sub("", stripped, count=1)
    return stripped.strip()


def _validate_text[T: BaseModel](schema: type[T], text: str | None) -> T:
    if text is None or not text.strip():
        raise ValueError("validation failed: empty model response")
    try:
        payload = json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"validation failed: response is not valid JSON: {exc}") from exc
    return schema.model_validate(payload)


def _repair_request(
    request: CompletionRequest,
    previous_text: str,
    error: Exception,
) -> CompletionRequest:
    user_content = (
        f"{request.user_content}\n\n"
        "The previous response failed validation.\n"
        f"Previous response:\n{previous_text}\n\n"
        f"Validation error:\n{error}\n\n"
        "Correct only what the validation error concerns. "
        "Return only the corrected JSON object."
    )
    return request.model_copy(update={"user_content": user_content})


def complete_structured[T: BaseModel](
    adapter: ModelAdapter,
    request: CompletionRequest,
    schema: type[T],
    run_id: str,
    max_repairs: int = 1,
) -> T:
    """Return a schema-validated completion with a bounded semantic repair loop.

    Transport retry remains inside the adapter.
    Schema/content repair belongs here.

    On validation failure, send the validation error text back to the model and
    instruct it to correct only what the error concerns. Do not perform more
    than max_repairs semantic repair attempts.
    """

    current = request
    last_text = ""
    for repair_attempt in range(max_repairs + 1):
        result = adapter.complete(current, run_id)
        last_text = result.text or ""
        try:
            if not result.succeeded:
                raise ValueError(
                    f"validation failed: adapter call did not succeed ({result.error_type})"
                )
            return _validate_text(schema, result.text)
        except (ValidationError, ValueError) as error:
            if repair_attempt >= max_repairs:
                raise
            current = _repair_request(request, last_text, error)

    raise RuntimeError("structured completion exhausted without a result")
