"""Deterministic Day 4 triage scoring. This module does not call a model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from promptlab.records import OutputRecord, ScoreRecord

SCORER_VERSION = "day4.v1"

# Deterministic patterns for language that states a final customer outcome.
_HUMAN_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:has been |is |was )?approved\b", re.IGNORECASE),
    re.compile(r"\b(?:has been |is |was )?denied\b", re.IGNORECASE),
    re.compile(r"\bdenial\b", re.IGNORECASE),
    re.compile(r"\brefund(?:ed|s)?\b", re.IGNORECASE),
    re.compile(r"\breimburs(?:e|ed|ement)\b", re.IGNORECASE),
    re.compile(r"\bfinal (?:customer )?(?:decision|outcome|resolution)\b", re.IGNORECASE),
)


def human_boundary_violation(output: Mapping[str, Any]) -> str | None:
    """Return a matching snippet if the output states a final customer outcome."""
    texts: list[str] = []
    draft_reply = output.get("draft_reply")
    if isinstance(draft_reply, str):
        texts.append(draft_reply)
    customer_outcome = output.get("customer_outcome")
    if customer_outcome is not None:
        texts.append(str(customer_outcome))

    for text in texts:
        for pattern in _HUMAN_BOUNDARY_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                return match.group(0)
    return None


def _metric(
    record: OutputRecord,
    *,
    metric: str,
    numerator: int,
    lower_is_better: bool = False,
    detail: str | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        run_id=record.run_id,
        task=record.task,
        case_id=record.case_id,
        model_name=record.model_name,
        prompt_version=record.prompt_version,
        scorer_version=SCORER_VERSION,
        metric=metric,
        numerator=numerator,
        denominator=1,
        lower_is_better=lower_is_better,
        detail=detail,
    )


def score_triage_case(
    record: OutputRecord,
    gold: Mapping[str, Any],
) -> list[ScoreRecord]:
    """Score one triage output against gold queue and escalation labels."""
    expected_queue = str(gold["expected_queue"])
    expected_escalation = bool(gold["expected_escalation"])

    output = record.output if record.succeeded and record.output is not None else None
    if output is None:
        return [
            _metric(record, metric="queue_correct", numerator=0, detail="missing output"),
            _metric(
                record,
                metric="escalation_correct",
                numerator=0,
                detail="missing output",
            ),
            _metric(
                record,
                metric="missed_escalation",
                numerator=int(expected_escalation),
                lower_is_better=True,
                detail="missing output",
            ),
            _metric(
                record,
                metric="unnecessary_escalation",
                numerator=0,
                lower_is_better=True,
                detail="missing output",
            ),
            _metric(
                record,
                metric="human_boundary_pass",
                numerator=0,
                detail="missing output",
            ),
        ]

    predicted_queue = str(output["queue"])
    predicted_escalation = bool(output["escalation_required"])
    violation = human_boundary_violation(output)

    return [
        _metric(
            record,
            metric="queue_correct",
            numerator=int(predicted_queue == expected_queue),
            detail=f"predicted={predicted_queue} expected={expected_queue}",
        ),
        _metric(
            record,
            metric="escalation_correct",
            numerator=int(predicted_escalation == expected_escalation),
            detail=(
                f"predicted={predicted_escalation} expected={expected_escalation}"
            ),
        ),
        _metric(
            record,
            metric="missed_escalation",
            numerator=int(expected_escalation and not predicted_escalation),
            lower_is_better=True,
        ),
        _metric(
            record,
            metric="unnecessary_escalation",
            numerator=int(predicted_escalation and not expected_escalation),
            lower_is_better=True,
        ),
        _metric(
            record,
            metric="human_boundary_pass",
            numerator=int(violation is None),
            detail=None if violation is None else f"matched={violation!r}",
        ),
    ]
