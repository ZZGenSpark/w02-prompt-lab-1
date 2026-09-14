from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from promptlab.schemas import (
    OUTPUT_SCHEMAS,
    EvidenceField,
    PolicyExtraction,
    SummarizationOutput,
    schema_description,
)

SUMMARIZATION_EVIDENCE_FIELDS = (
    "title",
    "version",
    "effective_date",
    "purpose",
    "required_steps",
    "exceptions",
)

EXTRACTION_EVIDENCE_FIELDS = (
    "policy_name",
    "version",
    "effective_date",
    "jurisdictions",
    "beneficial_ownership_threshold",
    "review_frequency",
    "required_documents",
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


def _payload(field_names: tuple[str, ...], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_status": "valid",
        **{name: _evidence() for name in field_names},
    }
    payload.update(overrides)
    return payload


def test_summarization_output_accepts_complete_payload() -> None:
    summary = SummarizationOutput.model_validate(
        _payload(
            SUMMARIZATION_EVIDENCE_FIELDS,
            required_steps=_evidence(
                value=["Create a review item", "Route to Card Disputes"],
                citation="Required Steps",
            ),
            exceptions=_evidence(status="absent"),
        )
    )

    assert summary.document_status == "valid"
    assert summary.title.citation == "Document Control"
    assert summary.required_steps.status == "present"
    assert summary.exceptions.status == "absent"
    assert summary.exceptions.value is None
    assert summary.exceptions.citation is None


def test_summarization_output_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SummarizationOutput.model_validate(
            _payload(SUMMARIZATION_EVIDENCE_FIELDS, scope="not allowed")
        )


def test_summarization_output_rejects_unknown_document_status() -> None:
    with pytest.raises(ValidationError):
        SummarizationOutput.model_validate(
            _payload(SUMMARIZATION_EVIDENCE_FIELDS, document_status="draft")
        )


def test_summarization_output_requires_existing_fields() -> None:
    payload = _payload(SUMMARIZATION_EVIDENCE_FIELDS)
    del payload["purpose"]

    with pytest.raises(ValidationError):
        SummarizationOutput.model_validate(payload)


def test_summarization_evidence_fields_covers_evidence_only() -> None:
    summary = SummarizationOutput.model_validate(_payload(SUMMARIZATION_EVIDENCE_FIELDS))
    fields = summary.evidence_fields()

    assert set(fields) == set(SUMMARIZATION_EVIDENCE_FIELDS)
    assert all(isinstance(field, EvidenceField) for field in fields.values())
    assert "document_status" not in fields


def test_evidence_field_uses_citation_not_section() -> None:
    field = EvidenceField.model_validate(
        {"value": "1.0", "status": "present", "citation": "Document Control"}
    )

    assert field.citation == "Document Control"
    assert "section" not in EvidenceField.model_fields
    with pytest.raises(ValidationError):
        EvidenceField.model_validate(
            {
                "value": "1.0",
                "status": "present",
                "citation": "Document Control",
                "section": "Document Control",
            }
        )


def test_policy_extraction_accepts_complete_payload() -> None:
    extraction = PolicyExtraction.model_validate(
        _payload(
            EXTRACTION_EVIDENCE_FIELDS,
            beneficial_ownership_threshold=_evidence(status="absent"),
        )
    )

    assert extraction.document_status == "valid"
    assert extraction.policy_name.citation == "Document Control"
    assert extraction.beneficial_ownership_threshold.status == "absent"


def test_output_schemas_use_existing_models() -> None:
    assert OUTPUT_SCHEMAS["summarization"] is SummarizationOutput
    assert OUTPUT_SCHEMAS["extraction"] is PolicyExtraction


def test_schema_description_is_pretty_printed_json_schema() -> None:
    description = schema_description(SummarizationOutput)
    parsed = json.loads(description)

    assert description == json.dumps(SummarizationOutput.model_json_schema(), indent=2)
    assert parsed["title"] == "SummarizationOutput"
    assert parsed["additionalProperties"] is False
    for name in ("document_status", *SUMMARIZATION_EVIDENCE_FIELDS):
        assert name in parsed["properties"]
        assert name in parsed["required"]


def test_schema_description_includes_citation_on_evidence() -> None:
    parsed = json.loads(schema_description(EvidenceField))

    assert "citation" in parsed["properties"]
    assert "section" not in parsed["properties"]


def test_schema_description_differs_by_model() -> None:
    summary_schema = schema_description(SummarizationOutput)
    extraction_schema = schema_description(PolicyExtraction)

    assert summary_schema != extraction_schema
    assert "required_steps" in summary_schema
    assert "beneficial_ownership_threshold" in extraction_schema
    assert "beneficial_ownership_threshold" not in summary_schema
    assert '"citation"' in extraction_schema
