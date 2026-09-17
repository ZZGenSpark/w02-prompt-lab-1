from __future__ import annotations

from datetime import date

from promptlab.config import DEFAULT_MODEL_A
from promptlab.corpus import GoldLabel
from promptlab.rules import VersionCandidate
from promptlab.schemas import DocumentStatus, EvidenceField, PolicyExtraction, SummarizationOutput
from promptlab.scoring import (
    SCORER_VERSION,
    candidate_from_output,
    score_version_selection,
)


def _present(value: str, citation: str = "1. Document Control") -> EvidenceField:
    return EvidenceField(value=value, status="present", citation=citation)


def _absent() -> EvidenceField:
    return EvidenceField(value=None, status="absent")


def _extraction(
    *,
    version: str,
    effective_date: str,
    case_status: DocumentStatus = "valid",
) -> PolicyExtraction:
    return PolicyExtraction(
        document_status=case_status,
        policy_name=_present("Small Business Periodic KYC Review Policy"),
        version=_present(version),
        effective_date=_present(effective_date),
        jurisdictions=_present("Pennsylvania", "2. Scope"),
        beneficial_ownership_threshold=_present("25 percent", "3. Beneficial Ownership"),
        review_frequency=_present("24 months", "4. Review Frequency"),
        required_documents=_present("formation documents", "5. Required Documents"),
    )


def _group_labels() -> list[GoldLabel]:
    return [
        GoldLabel(
            id="E01",
            task="extraction",
            expected_status="superseded",
            version_group="small-business-periodic-kyc",
            expected_current_case_id="E02",
            as_of="2025-06-01",
        ),
        GoldLabel(
            id="E02",
            task="extraction",
            expected_status="valid",
            version_group="small-business-periodic-kyc",
            expected_current_case_id="E02",
            as_of="2025-06-01",
        ),
        GoldLabel(id="E03", task="extraction", expected_status="valid"),
    ]


def test_candidate_from_output_uses_extracted_evidence_fields() -> None:
    candidate = candidate_from_output(
        "E02",
        _extraction(version="2.0", effective_date="2025-01-01"),
    )
    assert candidate == VersionCandidate(
        case_id="E02",
        version="2.0",
        effective_date=date(2025, 1, 1),
    )


def test_candidate_from_output_rejects_absent_or_invalid_dates() -> None:
    missing_date = _extraction(version="2.0", effective_date="2025-01-01")
    missing_date = missing_date.model_copy(update={"effective_date": _absent()})
    assert candidate_from_output("E02", missing_date) is None

    bad_date = _extraction(version="2.0", effective_date="next-year")
    assert candidate_from_output("E02", bad_date) is None


def test_version_selection_scores_select_current_version_not_model_opinion() -> None:
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="mistral",
        prompt_version="v2",
        labels=_group_labels(),
        outputs={
            "E01": _extraction(
                version="1.0", effective_date="2024-01-01", case_status="superseded"
            ),
            "E02": _extraction(version="2.0", effective_date="2025-01-01"),
        },
        model_id=DEFAULT_MODEL_A,
        prompt_id="extract",
    )
    assert len(scores) == 1
    score = scores[0]
    assert score.metric == "version_selection_accuracy"
    assert score.case_id == "version:small-business-periodic-kyc"
    assert score.numerator == 1
    assert score.denominator == 1
    assert score.detail == "expected=E02; selected=E02"
    assert score.scorer_version == "day5.v2"
    assert SCORER_VERSION == "day5.v2"
    assert score.model_id == DEFAULT_MODEL_A
    assert score.prompt_id == "extract"


def test_effective_date_equal_to_as_of_is_scored_current() -> None:
    labels = [
        GoldLabel(
            id="old",
            task="extraction",
            version_group="boundary",
            expected_current_case_id="equal",
            as_of="2025-06-01",
        ),
        GoldLabel(
            id="equal",
            task="extraction",
            version_group="boundary",
            expected_current_case_id="equal",
            as_of="2025-06-01",
        ),
    ]
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="mistral",
        prompt_version="v2",
        labels=labels,
        outputs={
            "old": _extraction(version="1.0", effective_date="2025-01-01"),
            "equal": _extraction(version="2.0", effective_date="2025-06-01"),
        },
    )
    assert scores[0].numerator == 1
    assert scores[0].detail == "expected=equal; selected=equal"


def test_future_extracted_version_is_not_selected() -> None:
    labels = [
        GoldLabel(
            id="current",
            task="extraction",
            version_group="g",
            expected_current_case_id="current",
            as_of="2025-06-01",
        ),
        GoldLabel(
            id="future",
            task="extraction",
            version_group="g",
            expected_current_case_id="current",
            as_of="2025-06-01",
        ),
    ]
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="mistral",
        prompt_version="v2",
        labels=labels,
        outputs={
            "current": _extraction(version="1.0", effective_date="2025-01-01"),
            "future": _extraction(version="2.0", effective_date="2026-01-01"),
        },
    )
    assert scores[0].numerator == 1
    assert scores[0].detail == "expected=current; selected=current"


def test_equal_effective_dates_score_as_unresolved() -> None:
    labels = [
        GoldLabel(
            id="a",
            task="extraction",
            version_group="tie",
            expected_current_case_id="a",
            as_of="2025-06-01",
        ),
        GoldLabel(
            id="b",
            task="extraction",
            version_group="tie",
            expected_current_case_id="a",
            as_of="2025-06-01",
        ),
    ]
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="mistral",
        prompt_version="v2",
        labels=labels,
        outputs={
            "a": _extraction(version="1.0", effective_date="2025-01-01"),
            "b": _extraction(version="2.0", effective_date="2025-01-01"),
        },
    )
    assert scores[0].numerator == 0
    assert scores[0].detail == "expected=a; selected=none"


def test_bad_extraction_is_scored_as_selection_failure() -> None:
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="qwen",
        prompt_version="v2",
        labels=_group_labels(),
        outputs={
            "E01": _extraction(
                version="1.0", effective_date="2024-01-01", case_status="superseded"
            ),
            "E02": _extraction(version="2.0", effective_date="2023-01-01"),
        },
    )
    assert scores[0].numerator == 0
    assert scores[0].detail == "expected=E02; selected=E01"


def test_missing_extraction_does_not_invent_a_current_document() -> None:
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="mistral",
        prompt_version="v2",
        labels=_group_labels(),
        outputs={
            "E01": _extraction(
                version="1.0", effective_date="2024-01-01", case_status="superseded"
            ),
        },
    )
    assert scores[0].numerator == 0
    assert scores[0].detail == "expected=E02; selected=E01"


def test_summarization_version_group_uses_the_same_rule() -> None:
    labels = [
        GoldLabel(
            id="S01",
            task="summarization",
            version_group="card-dispute-intake",
            expected_current_case_id="S02",
            as_of="2025-06-01",
        ),
        GoldLabel(
            id="S02",
            task="summarization",
            version_group="card-dispute-intake",
            expected_current_case_id="S02",
            as_of="2025-06-01",
        ),
    ]
    outputs = {
        "S01": SummarizationOutput(
            document_status="superseded",
            title=_present("Card Dispute Intake"),
            version=_present("1.0"),
            effective_date=_present("2024-01-01"),
            purpose=_present("Route disputes", "2. Purpose"),
            required_steps=_absent(),
            exceptions=_absent(),
        ),
        "S02": SummarizationOutput(
            document_status="valid",
            title=_present("Card Dispute Intake"),
            version=_present("2.0"),
            effective_date=_present("2025-01-01"),
            purpose=_present("Route disputes", "2. Purpose"),
            required_steps=_absent(),
            exceptions=_absent(),
        ),
    }
    scores = score_version_selection(
        run_id="run-day5",
        task="summarization",
        model_name="mistral",
        prompt_version="v1",
        labels=labels,
        outputs=outputs,
    )
    assert scores[0].numerator == 1
    assert scores[0].case_id == "version:card-dispute-intake"


def test_ungrouped_labels_are_not_scored() -> None:
    scores = score_version_selection(
        run_id="run-day5",
        task="extraction",
        model_name="mistral",
        prompt_version="v2",
        labels=[GoldLabel(id="E03", task="extraction", expected_status="valid")],
        outputs={"E03": _extraction(version="1.5", effective_date="2025-02-01")},
    )
    assert scores == []
