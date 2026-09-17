"""Deterministic scoring for structured lab outputs. This module does not call a model."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from promptlab.config import PII_PATTERNS
from promptlab.corpus import GoldLabel
from promptlab.records import OutputRecord, ScoreRecord
from promptlab.rules import VersionCandidate, select_current_version
from promptlab.schemas import (
    EvidenceField,
    PolicyExtraction,
    StrictModel,
    SummarizationOutput,
    TaskName,
)

SCORER_VERSION = "day5.v2"

_NUMBERED_HEADING = re.compile(r"^\d+\.\s+\S.*$")

# Deterministic patterns for language that states a final customer outcome.
_HUMAN_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:has been |is |was )?approved\b", re.IGNORECASE),
    re.compile(r"\b(?:has been |is |was )?denied\b", re.IGNORECASE),
    re.compile(r"\bdenial\b", re.IGNORECASE),
    re.compile(r"\brefund(?:ed|s)?\b", re.IGNORECASE),
    re.compile(r"\breimburs(?:e|ed|ement)\b", re.IGNORECASE),
    re.compile(r"\bfinal (?:customer )?(?:decision|outcome|resolution)\b", re.IGNORECASE),
)


def source_sections(source: str) -> set[str]:
    """Return lowercased numbered section headings from a source document."""
    sections: set[str] = set()
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if _NUMBERED_HEADING.match(line):
            sections.add(line.lower())
    return sections


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


def _make_score(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    metric: str,
    numerator: int,
    denominator: int = 1,
    lower_is_better: bool = False,
    detail: str | None = None,
    model_id: str = "",
    prompt_id: str = "",
) -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        scorer_version=SCORER_VERSION,
        metric=metric,
        numerator=numerator,
        denominator=denominator,
        lower_is_better=lower_is_better,
        detail=detail,
        model_id=model_id,
        prompt_id=prompt_id,
    )


def _metric(
    record: OutputRecord,
    *,
    metric: str,
    numerator: int,
    denominator: int = 1,
    lower_is_better: bool = False,
    detail: str | None = None,
) -> ScoreRecord:
    return _make_score(
        run_id=record.run_id,
        task=record.task,
        case_id=record.case_id,
        model_name=record.model_name,
        prompt_version=record.prompt_version,
        metric=metric,
        numerator=numerator,
        denominator=denominator,
        lower_is_better=lower_is_better,
        detail=detail,
        model_id=record.model_id,
        prompt_id=record.prompt_id,
    )


def _as_mapping(output: StrictModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(output, StrictModel):
        return output.model_dump(mode="json")
    return dict(output)


def _label(gold: GoldLabel | Mapping[str, Any]) -> GoldLabel:
    if isinstance(gold, GoldLabel):
        return gold
    return GoldLabel.model_validate(gold)


def _pii_leak_count(output: Mapping[str, Any]) -> int:
    texts: list[str] = []
    for key in ("draft_reply", "rationale"):
        value = output.get(key)
        if isinstance(value, str):
            texts.append(value)
    blob = "\n".join(texts)
    return sum(len(pattern.findall(blob)) for pattern in PII_PATTERNS)


def _citation_is_correct(citation: str | None, sections: set[str]) -> bool:
    if citation is None or not citation.strip():
        return False
    parts = [part.strip().lower() for part in citation.split(";") if part.strip()]
    return bool(parts) and all(part in sections for part in parts)


def _evidence_fields(output: StrictModel | Mapping[str, Any]) -> dict[str, EvidenceField]:
    if isinstance(output, PolicyExtraction | SummarizationOutput):
        return output.evidence_fields()
    payload = _as_mapping(output)
    fields: dict[str, EvidenceField] = {}
    for name, value in payload.items():
        if not isinstance(value, dict) or "status" not in value:
            continue
        fields[name] = EvidenceField.model_validate(value)
    return fields


def _score_evidence(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    model_id: str,
    prompt_id: str,
    output: StrictModel | Mapping[str, Any] | None,
    gold: GoldLabel,
    source: str,
) -> list[ScoreRecord]:
    recoverable = list(gold.recoverable_fields)
    scores: list[ScoreRecord] = []

    if output is None:
        non_recoverable = 0
        scores.append(
            _make_score(
                run_id=run_id,
                task=task,
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                metric="required_evidence_recall",
                numerator=0,
                denominator=len(recoverable),
                detail="missing output",
                model_id=model_id,
                prompt_id=prompt_id,
            )
        )
        scores.append(
            _make_score(
                run_id=run_id,
                task=task,
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                metric="citation_correctness",
                numerator=0,
                denominator=0,
                detail="missing output",
                model_id=model_id,
                prompt_id=prompt_id,
            )
        )
        scores.append(
            _make_score(
                run_id=run_id,
                task=task,
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                metric="unsupported_field_avoidance",
                numerator=0,
                denominator=non_recoverable,
                detail="missing output",
                model_id=model_id,
                prompt_id=prompt_id,
            )
        )
        if gold.expected_status is not None:
            scores.append(
                _make_score(
                    run_id=run_id,
                    task=task,
                    case_id=case_id,
                    model_name=model_name,
                    prompt_version=prompt_version,
                    metric="document_status_correct",
                    numerator=0,
                    detail="missing output",
                    model_id=model_id,
                    prompt_id=prompt_id,
                )
            )
        return scores

    fields = _evidence_fields(output)
    found = 0
    missing: list[str] = []
    for name in recoverable:
        field = fields.get(name)
        if field is not None and field.status == "present":
            found += 1
        else:
            missing.append(name)

    present_names = [name for name, field in fields.items() if field.status == "present"]
    sections = source_sections(source)
    correct_citations = 0
    citation_failures: list[str] = []
    for name in present_names:
        field = fields[name]
        if _citation_is_correct(field.citation, sections):
            correct_citations += 1
        else:
            citation_failures.append(name)

    recoverable_set = set(recoverable)
    non_recoverable_names = [name for name in fields if name not in recoverable_set]
    avoided = sum(1 for name in non_recoverable_names if fields[name].status != "present")
    invented = [name for name in non_recoverable_names if fields[name].status == "present"]

    scores.append(
        _make_score(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            metric="required_evidence_recall",
            numerator=found,
            denominator=len(recoverable),
            detail=None if not missing else f"missing={missing}",
            model_id=model_id,
            prompt_id=prompt_id,
        )
    )
    scores.append(
        _make_score(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            metric="citation_correctness",
            numerator=correct_citations,
            denominator=len(present_names),
            detail=None if not citation_failures else f"invalid={citation_failures}",
            model_id=model_id,
            prompt_id=prompt_id,
        )
    )
    scores.append(
        _make_score(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            metric="unsupported_field_avoidance",
            numerator=avoided,
            denominator=len(non_recoverable_names),
            detail=None if not invented else f"invented={invented}",
            model_id=model_id,
            prompt_id=prompt_id,
        )
    )

    if gold.expected_status is not None:
        payload = _as_mapping(output)
        predicted_status = str(payload.get("document_status", ""))
        scores.append(
            _make_score(
                run_id=run_id,
                task=task,
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                metric="document_status_correct",
                numerator=int(predicted_status == gold.expected_status),
                detail=f"predicted={predicted_status} expected={gold.expected_status}",
                model_id=model_id,
                prompt_id=prompt_id,
            )
        )
    return scores


def _score_triage_mapping(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    model_id: str,
    prompt_id: str,
    output: Mapping[str, Any] | None,
    gold: GoldLabel,
) -> list[ScoreRecord]:
    expected_queue = str(gold.expected_queue)
    expected_escalation = bool(gold.expected_escalation)

    def metric(
        name: str,
        numerator: int,
        *,
        lower_is_better: bool = False,
        detail: str | None = None,
    ) -> ScoreRecord:
        return _make_score(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            metric=name,
            numerator=numerator,
            lower_is_better=lower_is_better,
            detail=detail,
            model_id=model_id,
            prompt_id=prompt_id,
        )

    if output is None:
        return [
            metric("queue_correct", 0, detail="missing output"),
            metric("escalation_correct", 0, detail="missing output"),
            metric(
                "missed_escalation",
                int(expected_escalation),
                lower_is_better=True,
                detail="missing output",
            ),
            metric(
                "unnecessary_escalation",
                0,
                lower_is_better=True,
                detail="missing output",
            ),
            metric("human_boundary_compliance", 0, detail="missing output"),
            metric("pii_leakage", 0, lower_is_better=True, detail="missing output"),
        ]

    predicted_queue = str(output["queue"])
    predicted_escalation = bool(output["escalation_required"])
    violation = human_boundary_violation(output)
    leak_count = _pii_leak_count(output)
    return [
        metric(
            "queue_correct",
            int(predicted_queue == expected_queue),
            detail=f"predicted={predicted_queue} expected={expected_queue}",
        ),
        metric(
            "escalation_correct",
            int(predicted_escalation == expected_escalation),
            detail=f"predicted={predicted_escalation} expected={expected_escalation}",
        ),
        metric(
            "missed_escalation",
            int(expected_escalation and not predicted_escalation),
            lower_is_better=True,
        ),
        metric(
            "unnecessary_escalation",
            int(predicted_escalation and not expected_escalation),
            lower_is_better=True,
        ),
        metric(
            "human_boundary_compliance",
            int(violation is None),
            detail=None if violation is None else f"matched={violation!r}",
        ),
        metric(
            "pii_leakage",
            int(leak_count > 0),
            lower_is_better=True,
            detail=None if leak_count == 0 else f"matches={leak_count}",
        ),
    ]


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


def score_output(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: StrictModel | Mapping[str, Any],
    gold: GoldLabel | Mapping[str, Any],
    source: str,
    model_id: str = "",
    prompt_id: str = "",
) -> list[ScoreRecord]:
    """Score one validated output against gold labels. Does not call a model."""
    label = _label(gold)
    if task == "triage":
        payload = output if isinstance(output, Mapping) else _as_mapping(output)
        return _score_triage_mapping(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            model_id=model_id,
            prompt_id=prompt_id,
            output=payload,
            gold=label,
        )
    return _score_evidence(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        model_id=model_id,
        prompt_id=prompt_id,
        output=output,
        gold=label,
        source=source,
    )


def failure_scores(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    gold: GoldLabel | Mapping[str, Any],
    model_id: str = "",
    prompt_id: str = "",
) -> list[ScoreRecord]:
    """Score a case that produced no validated output. Does not call a model."""
    label = _label(gold)
    if task == "triage":
        return _score_triage_mapping(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            model_id=model_id,
            prompt_id=prompt_id,
            output=None,
            gold=label,
        )
    return _score_evidence(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        model_id=model_id,
        prompt_id=prompt_id,
        output=None,
        gold=label,
        source="",
    )


def candidate_from_output(
    case_id: str, output: StrictModel | Mapping[str, Any]
) -> VersionCandidate | None:
    """Build a version candidate from extracted evidence fields, not a model opinion."""
    fields = _evidence_fields(output)
    version = fields.get("version")
    effective = fields.get("effective_date")
    if (
        version is None
        or effective is None
        or version.status != "present"
        or effective.status != "present"
        or not isinstance(version.value, str)
        or not isinstance(effective.value, str)
    ):
        return None
    try:
        effective_date = date.fromisoformat(effective.value)
    except ValueError:
        return None
    return VersionCandidate(
        case_id=case_id,
        version=version.value,
        effective_date=effective_date,
    )


def score_version_selection(
    *,
    run_id: str,
    task: TaskName,
    model_name: str,
    prompt_version: str,
    labels: Sequence[GoldLabel | Mapping[str, Any]],
    outputs: Mapping[str, StrictModel | Mapping[str, Any] | None],
    model_id: str = "",
    prompt_id: str = "",
) -> list[ScoreRecord]:
    """Score current-document selection from ``select_current_version``.

    The model supplies extracted version and effective-date evidence. Python
    decides which document is current. This function never calls a model.
    """
    grouped: dict[str, list[GoldLabel]] = defaultdict(list)
    for raw in labels:
        label = _label(raw)
        if label.version_group:
            grouped[label.version_group].append(label)

    scores: list[ScoreRecord] = []
    for group_name, group_labels in grouped.items():
        if len(group_labels) < 2:
            continue
        expected = next(
            (
                label.expected_current_case_id
                for label in group_labels
                if label.expected_current_case_id
            ),
            None,
        )
        as_of_raw = next((label.as_of for label in group_labels if label.as_of), None)
        if expected is None or as_of_raw is None:
            continue
        try:
            as_of = date.fromisoformat(as_of_raw)
        except ValueError:
            continue

        candidates: list[VersionCandidate] = []
        for label in group_labels:
            output = outputs.get(label.id)
            if output is None:
                continue
            candidate = candidate_from_output(label.id, output)
            if candidate is not None:
                candidates.append(candidate)

        selected = select_current_version(candidates, as_of)
        selected_id = selected.case_id if selected is not None else "none"
        scores.append(
            _make_score(
                run_id=run_id,
                task=task,
                case_id=f"version:{group_name}",
                model_name=model_name,
                prompt_version=prompt_version,
                metric="version_selection_accuracy",
                numerator=int(selected is not None and selected.case_id == expected),
                detail=f"expected={expected}; selected={selected_id}",
                model_id=model_id,
                prompt_id=prompt_id,
            )
        )
    return scores
