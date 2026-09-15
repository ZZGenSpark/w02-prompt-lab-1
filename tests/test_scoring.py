from __future__ import annotations

from typing import Any

from promptlab.records import OutputRecord, ScoreRecord
from promptlab.scoring import human_boundary_violation, score_triage_case


def _output(
    *,
    queue: str = "card_dispute",
    escalation_required: bool = False,
    draft_reply: str = "A specialist will review this request.",
    human_review_required: bool = True,
    customer_outcome: Any = None,
) -> dict[str, Any]:
    return {
        "queue": queue,
        "escalation_required": escalation_required,
        "confidence": 0.8,
        "rationale": "Clear duplicate charge from a recognized merchant.",
        "draft_reply": draft_reply,
        "human_review_required": human_review_required,
        "customer_outcome": customer_outcome,
    }


def _record(
    *,
    succeeded: bool = True,
    output: dict[str, Any] | None = None,
    error: str | None = None,
) -> OutputRecord:
    return OutputRecord(
        run_id="run-day4",
        task="triage",
        case_id="T01",
        model_name="mistral",
        model_id="mistral:7b",
        prompt_version="v1",
        succeeded=succeeded,
        repairs=0,
        output=_output() if succeeded and output is None else output,
        error=error,
    )


def _gold(
    *,
    expected_queue: str = "card_dispute",
    expected_escalation: bool = False,
) -> dict[str, Any]:
    return {
        "id": "T01",
        "task": "triage",
        "expected_queue": expected_queue,
        "expected_escalation": expected_escalation,
    }


def _by_metric(scores: list[ScoreRecord]) -> dict[str, ScoreRecord]:
    return {score.metric: score for score in scores}


def test_queue_accuracy_compares_predicted_queue_to_gold() -> None:
    scores = _by_metric(
        score_triage_case(
            _record(output=_output(queue="fraud_report")),
            _gold(expected_queue="card_dispute"),
        )
    )

    assert scores["queue_correct"].numerator == 0
    assert scores["queue_correct"].denominator == 1
    assert scores["queue_correct"].scorer_version == "day4.v1"


def test_escalation_accuracy_uses_escalation_required_not_human_review() -> None:
    output = _output(escalation_required=False, human_review_required=True)
    scores = _by_metric(
        score_triage_case(
            _record(output=output),
            _gold(expected_escalation=True),
        )
    )

    assert scores["escalation_correct"].numerator == 0
    assert scores["missed_escalation"].numerator == 1
    assert scores["missed_escalation"].lower_is_better is True
    assert scores["unnecessary_escalation"].numerator == 0


def test_unnecessary_escalation_is_tracked_separately() -> None:
    scores = _by_metric(
        score_triage_case(
            _record(output=_output(escalation_required=True)),
            _gold(expected_escalation=False),
        )
    )

    assert scores["escalation_correct"].numerator == 0
    assert scores["missed_escalation"].numerator == 0
    assert scores["unnecessary_escalation"].numerator == 1
    assert scores["unnecessary_escalation"].lower_is_better is True


def test_correct_routing_and_escalation_score_one() -> None:
    scores = _by_metric(score_triage_case(_record(), _gold()))

    assert scores["queue_correct"].numerator == 1
    assert scores["escalation_correct"].numerator == 1
    assert scores["missed_escalation"].numerator == 0
    assert scores["unnecessary_escalation"].numerator == 0
    assert scores["human_boundary_pass"].numerator == 1


def test_human_boundary_fails_on_approval_and_refund_language() -> None:
    draft = "Your dispute has been approved and the funds will be refunded."
    scores = _by_metric(
        score_triage_case(_record(output=_output(draft_reply=draft)), _gold())
    )

    assert human_boundary_violation(_output(draft_reply=draft)) is not None
    assert scores["human_boundary_pass"].numerator == 0
    assert scores["queue_correct"].numerator == 1


def test_human_boundary_inspects_customer_outcome() -> None:
    output = _output(customer_outcome="claim denied")
    assert human_boundary_violation(output) is not None
    scores = _by_metric(score_triage_case(_record(output=output), _gold()))
    assert scores["human_boundary_pass"].numerator == 0


def test_failed_output_scores_zero_and_counts_missed_escalation() -> None:
    scores = _by_metric(
        score_triage_case(
            _record(succeeded=False, output=None, error="validation failed"),
            _gold(expected_escalation=True),
        )
    )

    assert scores["queue_correct"].numerator == 0
    assert scores["escalation_correct"].numerator == 0
    assert scores["missed_escalation"].numerator == 1
    assert scores["unnecessary_escalation"].numerator == 0
    assert scores["human_boundary_pass"].numerator == 0
    assert scores["queue_correct"].task == "triage"
    assert scores["queue_correct"].prompt_version == "v1"
