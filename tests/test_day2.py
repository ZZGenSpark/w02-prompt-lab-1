from __future__ import annotations

from pathlib import Path

import pytest

from promptlab.config import Settings
from promptlab.day2 import (
    CASE_IDS,
    CASES_PATH,
    MAX_OUTPUT_TOKENS,
    PROMPT_PATH,
    build_request,
    load_cases,
)

EXPECTED_CASE_IDS = tuple(f"S{index:02d}" for index in range(1, 13))
SHARED_REQUEST_FIELDS = (
    "task",
    "case_id",
    "prompt_id",
    "prompt_version",
    "temperature",
    "max_output_tokens",
)


def test_case_ids_cover_all_twelve_summarization_cases() -> None:
    assert CASE_IDS == EXPECTED_CASE_IDS


def test_max_output_tokens_is_a_shared_ceiling_both_models_can_finish_under() -> None:
    assert MAX_OUTPUT_TOKENS == 548


def test_load_cases_returns_all_summarization_ids_in_order() -> None:
    cases = load_cases(CASES_PATH, CASE_IDS)

    assert [str(case["id"]) for case in cases] == list(EXPECTED_CASE_IDS)
    assert all("source" in case for case in cases)


def test_build_request_fills_document_placeholder() -> None:
    settings = Settings.from_env()
    case = load_cases(CASES_PATH, CASE_IDS)[0]
    template = PROMPT_PATH.read_text(encoding="utf-8")

    request = build_request(case, template, settings)
    combined = f"{request.system}\n{request.user_content}"

    assert "{document_text}" not in combined
    assert str(case["source"]) in combined
    assert request.task == "summarization"
    assert request.case_id == str(case["id"])
    assert request.prompt_id == "baseline"
    assert request.prompt_version == "v0"
    assert request.temperature == settings.temperature
    assert request.max_output_tokens == MAX_OUTPUT_TOKENS


def test_request_fields_are_identical_across_configured_models() -> None:
    settings = Settings.from_env()
    case = load_cases(CASES_PATH, CASE_IDS)[0]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    model_ids = [config.model_id for config in settings.models.values()]

    assert len(model_ids) == 2
    assert model_ids[0] != model_ids[1]

    requests = [build_request(case, template, settings) for _ in model_ids]
    first = requests[0]
    for other in requests[1:]:
        for field in SHARED_REQUEST_FIELDS:
            assert getattr(other, field) == getattr(first, field)


def test_day2_source_has_no_model_identifier_literals() -> None:
    from promptlab import day2

    source = Path(day2.__file__).read_text(encoding="utf-8")
    assert "mistral:7b" not in source
    assert "qwen3:8b" not in source


def test_load_cases_missing_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "summarization.jsonl"
    path.write_text('{"id":"S01","source":"only one case"}\n', encoding="utf-8")

    with pytest.raises(KeyError, match="Missing"):
        load_cases(path, ("S01", "S99"))


def test_build_request_rejects_unfilled_placeholder() -> None:
    settings = Settings.from_env()
    case = {"id": "S01", "source": "Card dispute intake procedure."}

    with pytest.raises(ValueError, match="document_text"):
        build_request(case, "Summarize the document with no placeholder.", settings)
