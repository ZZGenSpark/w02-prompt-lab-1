from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, ModelAdapter


def _load_json(text: str) -> Any:
    """Parse a JSON object, allowing a Markdown fence or leading commentary."""
    stripped = text.strip()
    if stripped.startswith("```"):
        rest = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        fence = rest.rfind("```")
        if fence != -1:
            rest = rest[:fence]
        stripped = rest.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _validate_text[T: BaseModel](schema: type[T], text: str | None) -> T:
    if text is None or not text.strip():
        raise ValueError("validation failed: empty model response")
    try:
        payload = _load_json(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"validation failed: response is not valid JSON: {exc}") from exc
    return schema.model_validate(payload)


def _repair_request(request: CompletionRequest, error: Exception) -> CompletionRequest:
    user_content = (
        f"{request.user_content}\n\n"
        "Your previous response failed validation with the following error. "
        "Return only a corrected JSON instance of the required object. "
        "Do not return a JSON Schema. Do not include $defs, properties, $ref, "
        "additionalProperties, required, or type as top-level keys. "
        "Do not wrap the object in Markdown fences. Do not add commentary. "
        "Do not change any field the error does not concern.\n"
        f"<error>\n{error}\n</error>"
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
    for repair_attempt in range(max_repairs + 1):
        result = adapter.complete(current, run_id)
        if not result.succeeded:
            raise ValueError(
                f"validation failed: adapter call did not succeed ({result.error_type})"
            )
        try:
            return _validate_text(schema, result.text)
        except (ValidationError, ValueError) as error:
            if repair_attempt >= max_repairs:
                raise
            current = _repair_request(request, error)

    raise RuntimeError("structured completion exhausted without a result")
