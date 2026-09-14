from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from promptlab.schemas import (  # type: ignore[import-untyped]
    OUTPUT_SCHEMAS,
    EvidenceField,
    PolicyExtraction,
    ProcedureSummary,
    schema_description,
)

REQUIRED_EVIDENCE_FIELDS = (
    "version",
    "effective_date",
    "superseded_status",
    "scope",
    "required_analyst_actions",
    "evidence_to_gather",
    "deadlines",
    "out_of_scope_path",
)


def _evidence(
    *,
    status: str = "present",
    value: str | list[str] | None = "1.0",
    citation: str | None = "Document Control",
) -> dict[str, object]:
    if status != "present":
        return {"value": None, "status": status, "citation": None}
    return {"value": value, "status": status, "citation": citation}


def _procedure_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_status": "valid",
        **{name: _evidence() for name in REQUIRED_EVIDENCE_FIELDS},
    }
    payload.update(overrides)
    return payload


def test_procedure_summary_accepts_complete_payload() -> None:
    summary = ProcedureSummary.model_validate(
        _procedure_payload(
            required_analyst_actions=_evidence(
                value=["Create a review item", "Route to Card Disputes"],
                citation="Required Steps",
            ),
            deadlines=_evidence(status="absent"),
        )
    )

    assert summary.document_status == "valid"
    assert summary.version.value == "1.0"
    assert summary.required_analyst_actions.status == "present"
    assert summary.deadlines.status == "absent"
    assert summary.deadlines.value is None


def test_procedure_summary_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ProcedureSummary.model_validate(_procedure_payload(title="not allowed"))


def test_procedure_summary_rejects_unknown_document_status() -> None:
    with pytest.raises(ValidationError):
        ProcedureSummary.model_validate(_procedure_payload(document_status="draft"))


def test_procedure_summary_requires_all_assignment_fields() -> None:
    payload = _procedure_payload()
    del payload["scope"]

    with pytest.raises(ValidationError):
        ProcedureSummary.model_validate(payload)


def test_procedure_summary_evidence_fields_covers_evidence_only() -> None:
    summary = ProcedureSummary.model_validate(_procedure_payload())
    fields = summary.evidence_fields()

    assert set(fields) == set(REQUIRED_EVIDENCE_FIELDS)
    assert all(isinstance(field, EvidenceField) for field in fields.values())
    assert "document_status" not in fields


def test_summarization_schema_is_procedure_summary() -> None:
    assert OUTPUT_SCHEMAS["summarization"] is ProcedureSummary


def test_schema_description_is_pretty_printed_json_schema() -> None:
    description = schema_description(ProcedureSummary)
    parsed = json.loads(description)

    assert description == json.dumps(ProcedureSummary.model_json_schema(), indent=2)
    assert parsed["title"] == "ProcedureSummary"
    assert parsed["additionalProperties"] is False
    for name in ("document_status", *REQUIRED_EVIDENCE_FIELDS):
        assert name in parsed["properties"]
        assert name in parsed["required"]


def test_schema_description_differs_by_model() -> None:
    summary_schema = schema_description(ProcedureSummary)
    extraction_schema = schema_description(PolicyExtraction)

    assert summary_schema != extraction_schema
    assert "required_analyst_actions" in summary_schema
    assert "beneficial_ownership_threshold" in extraction_schema
    assert "beneficial_ownership_threshold" not in summary_schema
