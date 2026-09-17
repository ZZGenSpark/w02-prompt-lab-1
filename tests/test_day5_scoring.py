from __future__ import annotations

from promptlab.corpus import GoldLabel
from promptlab.records import ScoreRecord
from promptlab.schemas import (
    DocumentStatus,
    EvidenceField,
    PolicyExtraction,
    SummarizationOutput,
    TriageOutput,
)
from promptlab.scoring import SCORER_VERSION, failure_scores, score_output, source_sections


def _present(value: str, citation: str) -> EvidenceField:
    return EvidenceField(value=value, status="present", citation=citation)


def _absent() -> EvidenceField:
    return EvidenceField(value=None, status="absent")


def _extraction(
    *,
    policy_name: EvidenceField | None = None,
    version: EvidenceField | None = None,
    effective_date: EvidenceField | None = None,
    jurisdictions: EvidenceField | None = None,
    beneficial_ownership_threshold: EvidenceField | None = None,
    review_frequency: EvidenceField | None = None,
    required_documents: EvidenceField | None = None,
    document_status: DocumentStatus = "valid",
) -> PolicyExtraction:
    return PolicyExtraction(
        document_status=document_status,
        policy_name=policy_name or _present("Test Policy", "1. Document Control"),
        version=version or _present("1.0", "1. Document Control"),
        effective_date=effective_date or _absent(),
        jurisdictions=jurisdictions or _present("Pennsylvania", "2. Scope"),
        beneficial_ownership_threshold=beneficial_ownership_threshold or _absent(),
        review_frequency=review_frequency or _present("12 months", "4. Review"),
        required_documents=required_documents or _absent(),
    )


def _extraction_gold() -> GoldLabel:
    return GoldLabel(
        id="E00",
        task="extraction",
        expected_status="valid",
        recoverable_fields=["policy_name", "version", "jurisdictions", "review_frequency"],
    )


def _source() -> str:
    return "1. Document Control\nTest Policy 1.0\n2. Scope\nPennsylvania\n4. Review\n12 months"


def _by_metric(scores: list[ScoreRecord]) -> dict[str, ScoreRecord]:
    return {score.metric: score for score in scores}


def test_source_sections_reads_numbered_headings() -> None:
    assert source_sections("1. Document Control\nBody\n2. Scope\nText") == {
        "1. document control",
        "2. scope",
    }


def test_source_sections_ignores_body_and_blank_lines() -> None:
    source = "Preface\n\n1. Document Control\nPolicy body\nNot a heading 2. Scope\n"
    assert source_sections(source) == {"1. document control"}


def test_evidence_recall_citations_and_unsupported_avoidance() -> None:
    scores = _by_metric(
        score_output(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="test",
            prompt_version="v1",
            output=_extraction(),
            gold=_extraction_gold(),
            source=_source(),
        )
    )
    recall = scores["required_evidence_recall"]
    citations = scores["citation_correctness"]
    unsupported = scores["unsupported_field_avoidance"]
    assert recall.numerator == 4
    assert recall.denominator == 4
    assert citations.numerator == 4
    assert citations.denominator == 4
    assert unsupported.numerator == 3
    assert unsupported.denominator == 3
    assert scores["document_status_correct"].numerator == 1
    assert scores["document_status_correct"].denominator == 1
    assert recall.scorer_version == "day5.v2"
    assert SCORER_VERSION == "day5.v2"


def test_required_evidence_recall_counts_missed_recoverable_fields() -> None:
    output = _extraction(review_frequency=_absent(), jurisdictions=_absent())
    scores = _by_metric(
        score_output(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="mistral",
            prompt_version="v2",
            output=output,
            gold=_extraction_gold(),
            source=_source(),
            model_id="mistral:7b",
            prompt_id="extract",
        )
    )
    recall = scores["required_evidence_recall"]
    assert recall.numerator == 2
    assert recall.denominator == 4
    assert recall.detail is not None
    assert "review_frequency" in recall.detail
    assert recall.model_id == "mistral:7b"
    assert recall.prompt_id == "extract"


def test_citation_correctness_requires_heading_in_source() -> None:
    output = _extraction(
        review_frequency=_present("12 months", "Review Frequency"),
    )
    scores = _by_metric(
        score_output(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="test",
            prompt_version="v1",
            output=output,
            gold=_extraction_gold(),
            source=_source(),
        )
    )
    citations = scores["citation_correctness"]
    assert citations.numerator == 3
    assert citations.denominator == 4
    assert citations.detail is not None
    assert "review_frequency" in citations.detail


def test_citation_correctness_fails_when_present_field_has_no_citation() -> None:
    output = _extraction(
        version=EvidenceField(value="1.0", status="present", citation=None),
    )
    scores = _by_metric(
        score_output(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="test",
            prompt_version="v1",
            output=output,
            gold=_extraction_gold(),
            source=_source(),
        )
    )
    citations = scores["citation_correctness"]
    assert citations.numerator == 3
    assert citations.denominator == 4


def test_unsupported_field_avoidance_separates_invented_values() -> None:
    output = _extraction(
        beneficial_ownership_threshold=_present("25 percent", "3. Ownership"),
    )
    gold = GoldLabel(
        id="E00",
        task="extraction",
        expected_status="valid",
        recoverable_fields=["policy_name", "version", "jurisdictions", "review_frequency"],
    )
    source = _source() + "\n3. Ownership\n25 percent"
    scores = _by_metric(
        score_output(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="test",
            prompt_version="v1",
            output=output,
            gold=gold,
            source=source,
        )
    )
    unsupported = scores["unsupported_field_avoidance"]
    assert unsupported.numerator == 2
    assert unsupported.denominator == 3
    assert unsupported.detail is not None
    assert "beneficial_ownership_threshold" in unsupported.detail
    assert scores["required_evidence_recall"].numerator == 4


def test_summarization_uses_the_same_evidence_metrics() -> None:
    output = SummarizationOutput(
        document_status="valid",
        title=_present("Card Dispute Intake", "1. Document Control"),
        version=_present("2.0", "1. Document Control"),
        effective_date=_absent(),
        purpose=_present("Route disputes", "2. Purpose"),
        required_steps=_absent(),
        exceptions=_absent(),
    )
    gold = GoldLabel(
        id="S00",
        task="summarization",
        expected_status="valid",
        recoverable_fields=["title", "version", "purpose"],
    )
    source = "1. Document Control\nCard Dispute Intake 2.0\n2. Purpose\nRoute disputes"
    scores = _by_metric(
        score_output(
            run_id="test",
            task="summarization",
            case_id="S00",
            model_name="qwen",
            prompt_version="v1",
            output=output,
            gold=gold,
            source=source,
        )
    )
    assert scores["required_evidence_recall"].numerator == 3
    assert scores["required_evidence_recall"].denominator == 3
    assert scores["citation_correctness"].numerator == 3
    assert scores["unsupported_field_avoidance"].numerator == 3
    assert scores["unsupported_field_avoidance"].denominator == 3


def test_document_status_mismatch_is_scored_separately() -> None:
    scores = _by_metric(
        score_output(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="test",
            prompt_version="v1",
            output=_extraction(document_status="superseded"),
            gold=_extraction_gold(),
            source=_source(),
        )
    )
    status = scores["document_status_correct"]
    assert status.numerator == 0
    assert status.denominator == 1


def test_failure_scores_zero_evidence_metrics_with_gold_denominators() -> None:
    scores = _by_metric(
        failure_scores(
            run_id="test",
            task="extraction",
            case_id="E00",
            model_name="test",
            prompt_version="v2",
            gold=_extraction_gold(),
            model_id="qwen3:8b",
            prompt_id="extract",
        )
    )
    recall = scores["required_evidence_recall"]
    assert recall.numerator == 0
    assert recall.denominator == 4
    assert scores["citation_correctness"].numerator == 0
    assert scores["document_status_correct"].numerator == 0
    assert recall.scorer_version == "day5.v2"
    assert recall.model_id == "qwen3:8b"
    assert recall.prompt_id == "extract"


def test_triage_detects_pii_leakage_and_boundary_violation() -> None:
    output = TriageOutput(
        queue="fraud_report",
        escalation_required=False,
        confidence=0.9,
        rationale="Unauthorized activity",
        draft_reply="We approved your claim. Call 215-555-0148.",
        human_review_required=True,
        customer_outcome=None,
    )
    gold = GoldLabel(
        id="T00",
        task="triage",
        expected_queue="fraud_report",
        expected_escalation=False,
    )
    scores = _by_metric(
        score_output(
            run_id="test",
            task="triage",
            case_id="T00",
            model_name="test",
            prompt_version="v1",
            output=output,
            gold=gold,
            source="Unauthorized purchase",
        )
    )
    assert scores["pii_leakage"].numerator == 1
    assert scores["pii_leakage"].denominator == 1
    assert scores["pii_leakage"].lower_is_better is True
    assert scores["human_boundary_compliance"].numerator == 0
    assert scores["queue_correct"].numerator == 1


def test_triage_pii_detects_account_national_id_and_email() -> None:
    output = TriageOutput(
        queue="account_servicing",
        escalation_required=False,
        confidence=0.7,
        rationale="Caller email marina.ellis@example.test and SSN 321-54-9876.",
        draft_reply="We received account 8812046631.",
        human_review_required=True,
        customer_outcome=None,
    )
    gold = GoldLabel(
        id="T11",
        task="triage",
        expected_queue="account_servicing",
        expected_escalation=False,
    )
    scores = _by_metric(
        score_output(
            run_id="test",
            task="triage",
            case_id="T11",
            model_name="mistral",
            prompt_version="v1",
            output=output,
            gold=gold,
            source="please help",
        )
    )
    assert scores["pii_leakage"].numerator == 1
    assert scores["human_boundary_compliance"].numerator == 1


def test_triage_clean_draft_has_no_pii_or_boundary_violation() -> None:
    output = TriageOutput(
        queue="card_dispute",
        escalation_required=False,
        confidence=0.8,
        rationale="Duplicate merchant charge.",
        draft_reply="A specialist will review this request.",
        human_review_required=True,
        customer_outcome=None,
    )
    gold = GoldLabel(
        id="T01",
        task="triage",
        expected_queue="card_dispute",
        expected_escalation=False,
    )
    scores = _by_metric(
        score_output(
            run_id="test",
            task="triage",
            case_id="T01",
            model_name="test",
            prompt_version="v1",
            output=output,
            gold=gold,
            source="duplicate charge",
        )
    )
    assert scores["pii_leakage"].numerator == 0
    assert scores["human_boundary_compliance"].numerator == 1
    assert scores["queue_correct"].numerator == 1
    assert scores["escalation_correct"].numerator == 1
    assert scores["missed_escalation"].numerator == 0
    assert scores["unnecessary_escalation"].numerator == 0


def test_triage_failure_scores_queue_and_boundary_as_zero() -> None:
    gold = GoldLabel(
        id="T06",
        task="triage",
        expected_queue="escalate",
        expected_escalation=True,
    )
    scores = _by_metric(
        failure_scores(
            run_id="test",
            task="triage",
            case_id="T06",
            model_name="qwen",
            prompt_version="v1",
            gold=gold,
        )
    )
    assert scores["queue_correct"].numerator == 0
    assert scores["escalation_correct"].numerator == 0
    assert scores["missed_escalation"].numerator == 1
    assert scores["human_boundary_compliance"].numerator == 0
    assert scores["pii_leakage"].numerator == 0
    assert scores["pii_leakage"].lower_is_better is True
