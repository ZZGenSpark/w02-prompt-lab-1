"""Reporting for the Week 2 model-comparison lab.

The reporting layer consumes the existing UsageRecord, OutputRecord, and
ScoreRecord objects.  It does not rescore model output and it does not call an
LLM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any, cast

from promptlab.prompts import prompt_label
from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.schemas import TaskName

_ConfigKey = tuple[str, str, str]  # task, model_name, prompt_version
_ALL_TASKS: tuple[str, ...] = ("triage", "summarization", "extraction")
_ALL_MODELS: tuple[str, ...] = ("mistral", "qwen")


def _key(record: Any) -> _ConfigKey:
    return (
        str(record.task),
        str(record.model_name),
        str(record.prompt_version),
    )


def _for_run(records: Sequence[Any], run_id: str) -> list[Any]:
    return [record for record in records if str(record.run_id) == run_id]


def _fmt_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _aggregate_scores(
    records: Sequence[ScoreRecord],
) -> dict[str, tuple[int, int, bool | None]]:
    """Aggregate compatible score counts without averaging percentages."""

    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.metric)].append(record)

    result: dict[str, tuple[int, int, bool | None]] = {}

    for metric, rows in sorted(grouped.items()):
        numerator = sum(int(row.numerator) for row in rows)
        denominator = sum(int(row.denominator) for row in rows)
        directions = {
            bool(value)
            for value in (getattr(row, "lower_is_better", None) for row in rows)
            if value is not None
        }
        lower_is_better = next(iter(directions)) if len(directions) == 1 else None
        result[metric] = (numerator, denominator, lower_is_better)

    return result


def _metric_text(records: Sequence[ScoreRecord]) -> str:
    metrics = _aggregate_scores(records)
    if not metrics:
        return "—"

    rendered: list[str] = []
    for metric, (numerator, denominator, lower_is_better) in metrics.items():
        suffix = " ↓" if lower_is_better else ""
        rendered.append(f"{metric}: {numerator}/{denominator}{suffix}")

    return "<br>".join(rendered)


def _usage_summary(
    records: Sequence[UsageRecord],
    case_count: int,
) -> tuple[str, str, str, str, str, str]:
    """Return token, latency, observation, and retry summaries."""

    if not records:
        return "—", "—", "—", "—", "0", "0"

    prompt_tokens = sum(int(getattr(row, "prompt_tokens", 0) or 0) for row in records)
    completion_tokens = sum(
        int(getattr(row, "completion_tokens", 0) or 0) for row in records
    )

    latencies = [
        float(row.latency_ms)
        for row in records
        if getattr(row, "latency_ms", None) is not None
    ]

    if latencies:
        median_latency = f"{_fmt_number(float(median(latencies)))} ms"
        max_latency = f"{_fmt_number(float(max(latencies)))} ms"
    else:
        median_latency = "—"
        max_latency = "—"

    retry_attempts = sum(
        1
        for row in records
        if str(getattr(row, "kind", "")).lower() in {"transport_retry", "repair_retry"}
    )

    per_case = case_count if case_count > 0 else 1
    return (
        _fmt_number(prompt_tokens / per_case),
        _fmt_number(completion_tokens / per_case),
        median_latency,
        max_latency,
        str(len(latencies)),
        str(retry_attempts),
    )


def _case_summary(records: Sequence[OutputRecord]) -> tuple[str, str, str]:
    """Median/max wall time for a whole case (all attempts summed)."""
    latencies = [
        float(row.case_latency_ms)
        for row in records
        if getattr(row, "case_latency_ms", None) is not None
    ]
    if not latencies:
        return "—", "—", "0"
    return (
        f"{_fmt_number(float(median(latencies)))} ms",
        f"{_fmt_number(float(max(latencies)))} ms",
        str(len(latencies)),
    )


def _output_summary(
    records: Sequence[OutputRecord],
) -> tuple[str, str, str]:
    if not records:
        return "0/0", "0/0", "0"

    total = len(records)
    succeeded = sum(1 for row in records if bool(row.succeeded))
    repairs_needed = sum(
        1 for row in records if int(getattr(row, "repairs", 0) or 0) > 0
    )
    failures = total - succeeded

    return (
        f"{succeeded}/{total}",
        f"{repairs_needed}/{total}",
        str(failures),
    )


def _all_config_keys(
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
) -> list[_ConfigKey]:
    keys = {_key(row) for row in usage}
    keys.update(_key(row) for row in outputs)
    keys.update(_key(row) for row in scores)
    return sorted(keys)


def _display_prompt(task: str, model_name: str, prompt_version: str) -> str:
    return prompt_label(cast(TaskName, task), model_name, prompt_version)


def _metric_pair(records: Sequence[ScoreRecord]) -> str:
    if not records:
        return "—"
    numerator = sum(int(row.numerator) for row in records)
    denominator = sum(int(row.denominator) for row in records)
    return f"{numerator}/{denominator}"


def _human_boundary_lines(
    scores: Sequence[ScoreRecord],
    outputs: Sequence[OutputRecord],
) -> list[str]:
    triage_scores = [row for row in scores if str(row.task) == "triage"]
    if not any(row.metric == "human_boundary_compliance" for row in triage_scores):
        return []

    lines: list[str] = [
        "## Human-boundary re-verification",
        "",
        "Triage `draft_reply` was scored against the Day 4 human-boundary rule "
        "for both models in this run.",
        "",
        "| Model | Prompt | human_boundary_compliance | pii_leakage ↓ | Valid outputs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    keys = sorted(
        {(str(row.model_name), str(row.prompt_version)) for row in triage_scores}
    )
    for model_name, prompt_version in keys:
        boundary = [
            row
            for row in triage_scores
            if row.model_name == model_name
            and row.prompt_version == prompt_version
            and row.metric == "human_boundary_compliance"
        ]
        leakage = [
            row
            for row in triage_scores
            if row.model_name == model_name
            and row.prompt_version == prompt_version
            and row.metric == "pii_leakage"
        ]
        task_outputs = [
            row
            for row in outputs
            if str(row.task) == "triage"
            and str(row.model_name) == model_name
            and str(row.prompt_version) == prompt_version
        ]
        prompt = _display_prompt("triage", model_name, prompt_version)
        valid, _repairs, _failures = _output_summary(task_outputs)
        lines.append(
            f"| {model_name} | {prompt} | {_metric_pair(boundary)} | "
            f"{_metric_pair(leakage)} | {valid} |"
        )

    lines.extend(
        [
            "",
            "A missing-output human-boundary miss is a failed structured call, "
            "not a prohibited phrase in a committed `draft_reply`.",
            "",
        ]
    )
    return lines


def _untested_lines(keys: Sequence[_ConfigKey], models: Sequence[str]) -> list[str]:
    evaluated = {(task, model) for task, model, _prompt in keys}
    configured_models = [str(model) for model in models] or list(_ALL_MODELS)
    missing: list[str] = []
    for task in _ALL_TASKS:
        for model in configured_models:
            if (task, model) not in evaluated:
                missing.append(f"`{task}` × {model}")
    if not missing:
        return ["- Every configured task/model pair in this run produced records."]
    return [f"- Untested combinations in this run: {', '.join(missing)}."]


def _write_report(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
) -> None:
    lines: list[str] = [
        "# Model Comparison",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Counts are reported with their denominators. "
        "Call latency is one HTTP attempt. Case latency sums every attempt "
        "for that case, including repairs and transport retries. "
        "Latency uses median and maximum rather than mean. "
        "Local Ollama provider cost is `$0.00`.",
        "",
        "Prompt-transfer rows are labeled `transfer`. They measure that prompt "
        "on the second model, not the model's best adapted performance.",
        "",
    ]

    keys = _all_config_keys(usage, outputs, scores)
    tasks = sorted({task for task, _model, _prompt in keys})

    if not tasks:
        lines.extend(
            [
                "No records were supplied for this run.",
                "",
            ]
        )

    for task in tasks:
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Model | Prompt | Quality | Valid outputs | Input tokens/case | "
                "Output tokens/case | Median call latency | Max call latency | "
                "n calls | Median case latency | Max case latency | n cases | "
                "Repairs | Retries | Final failures |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        task_keys = [key for key in keys if key[0] == task]

        for key in task_keys:
            _task, model_name, prompt_version = key

            u = [row for row in usage if _key(row) == key]
            o = [row for row in outputs if _key(row) == key]
            s = [row for row in scores if _key(row) == key]
            case_count = len({str(row.case_id) for row in o}) or len(
                {str(row.case_id) for row in u}
            )

            (
                input_tokens,
                output_tokens,
                median_call_latency,
                max_call_latency,
                n_calls,
                retries,
            ) = _usage_summary(u, case_count)
            median_case_latency, max_case_latency, n_cases = _case_summary(o)

            valid_outputs, repairs, failures = _output_summary(o)
            metric_text = _metric_text(s)
            prompt = _display_prompt(_task, model_name, prompt_version)

            lines.append(
                "| "
                f"{model_name} | {prompt} | {metric_text} | {valid_outputs} | "
                f"{input_tokens} | {output_tokens} | "
                f"{median_call_latency} | {max_call_latency} | {n_calls} | "
                f"{median_case_latency} | {max_case_latency} | {n_cases} | "
                f"{repairs} | {retries} | {failures} |"
            )

        lines.append("")

    lines.extend(_human_boundary_lines(scores, outputs))

    lines.extend(
        [
            "## Limits",
            "",
            "- There are only 12 cases per task. Results are directional, not "
            "production-scale estimates.",
            "- A one-case gap such as 11/12 versus 10/12 is not a universal model ranking.",
            "- Prompt-transfer rows are identified in the Prompt column.",
            *_untested_lines(keys, models),
            "- No production-volume reliability claim is being made.",
            "- Local Ollama latency depends on lab hardware and is not a cloud SLA.",
            "- Local provider/API charge is `$0.00`; input tokens, output tokens, "
            "per-call and per-case median/max latency, observation counts, repair "
            "rate, and retry/failure counts are the operational measurements.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision_scaffold(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    decision_path: Path,
) -> None:
    """Write an evidence scaffold, not an invented model recommendation."""

    keys = _all_config_keys(usage, outputs, scores)

    lines: list[str] = [
        "# Model Decision Record",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Use this file to record the task-level decision after reviewing the measured "
        "comparison. Do not select one universal model solely because it leads on a "
        "different task. Do not rewrite earlier decision constraints after seeing "
        "these results.",
        "",
        "## Evaluated models",
        "",
    ]

    evaluated_models = sorted(
        {model for _task, model, _prompt in keys} | {str(model) for model in models}
    )
    if evaluated_models:
        for model in evaluated_models:
            lines.append(f"- {model}")
    else:
        lines.append("- None")

    lines.extend(["", "## Evaluated configurations", ""])

    if keys:
        for task, model, prompt in keys:
            label = _display_prompt(task, model, prompt)
            lines.append(f"- `{task}` — {model} — `{label}`")
    else:
        lines.append("- No configurations supplied.")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "Fill one row per measured configuration. Every row must name the prompt "
            "version, including `transfer` when the prompt was not adapted.",
            "",
            "## Task decisions",
            "",
            "For each task, complete:",
            "",
            "- selected model",
            "- prompt version",
            "- measured reason",
            "- rejected alternative(s)",
            "- condition that would reopen the decision",
            "",
            "## Review triggers",
            "",
            "- A new prompt version is measured.",
            "- The 12-case sample is replaced or expanded.",
            "- Human-boundary or PII leakage fails on a previously selected model.",
            "",
        ]
    )

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    if decision_path.exists() and "selected model:" in decision_path.read_text(
        encoding="utf-8"
    ).lower():
        return
    decision_path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    decision_path: Path,
) -> None:
    """Generate the comparison report and decision scaffold for one run.

    Only records whose ``run_id`` matches the requested run are included.
    """

    run_usage = _for_run(usage, run_id)
    run_outputs = _for_run(outputs, run_id)
    run_scores = _for_run(scores, run_id)

    _write_report(
        run_id=run_id,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        report_path=Path(report_path),
    )

    _write_decision_scaffold(
        run_id=run_id,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        decision_path=Path(decision_path),
    )
