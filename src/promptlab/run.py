"""Day 5 evaluation runner.

run.py -> prompt registry -> complete_structured -> OllamaAdapter -> Mistral/Qwen
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, ModelConfig, Settings
from promptlab.corpus import Case, GoldLabel, load_cases, validate_corpus
from promptlab.prompts import load, prompt_label, render_user, task_prompt
from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.records import append_record as append_jsonl
from promptlab.report import write_reports
from promptlab.schemas import OUTPUT_SCHEMAS, StrictModel, TaskName, schema_description
from promptlab.scoring import failure_scores, score_output, score_version_selection
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
MAX_OUTPUT_TOKENS = 1024
FULL_EVAL_COUNT = 72
UsageKind = Literal["primary", "transport_retry", "repair", "repair_retry"]
UsageStatus = Literal["success", "schema_invalid", "transport_error"]


class CountingAdapter:
    """Count adapter.complete calls so repairs can be distinguished from transport retries."""

    provider = "ollama"

    def __init__(self, inner: ModelAdapter) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.calls = 0
        self.records: list[CallRecord] = []
        self._call_index: dict[str, int] = {}

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        result = self._inner.complete(request, run_id)
        for record in result.records:
            self.records.append(record)
            self._call_index[record.record_id] = self.calls
        return result

    def reset(self) -> None:
        self.calls = 0
        self.records = []
        self._call_index = {}

    def call_index(self, record: CallRecord) -> int:
        return self._call_index.get(record.record_id, 1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local two-model prompt comparison")
    parser.add_argument("--run-id", help="Stable identifier for this run")
    parser.add_argument("--task", choices=["triage", "summarization", "extraction"])
    parser.add_argument("--model", choices=["mistral", "qwen"])
    parser.add_argument("--limit", type=int, help="Limit cases per task for a smoke run")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and corpus without calling Ollama",
    )
    return parser


def usage_kind(call_index: int, attempt: int) -> UsageKind:
    if call_index <= 1:
        return "primary" if attempt <= 1 else "transport_retry"
    return "repair" if attempt <= 1 else "repair_retry"


def usage_from_call_record(
    record: CallRecord,
    *,
    model_name: str,
    call_index: int,
) -> UsageRecord:
    status: UsageStatus = "success" if record.error_type is None else "transport_error"
    return UsageRecord(
        run_id=record.run_id,
        task=record.task,
        case_id=record.case_id,
        model_name=model_name,
        model_id=record.model_id,
        prompt_version=record.prompt_version,
        attempt=record.attempt,
        kind=usage_kind(call_index, record.attempt),
        status=status,
        prompt_tokens=record.input_tokens,
        completion_tokens=record.output_tokens,
        latency_ms=float(record.latency_ms),
        cost_usd=Decimal(str(record.cost_usd)),
        error=record.error_type,
    )


def _write_jsonl(path: Path, records: Sequence[CallRecord | ScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def write_day5_evidence(
    *,
    outputs: Sequence[OutputRecord],
    calls: Sequence[CallRecord],
    scores: Sequence[ScoreRecord],
    run_path: Path,
    scores_path: Path,
) -> None:
    """Write docs evidence only after a completed 72-eval run."""
    if len(outputs) != FULL_EVAL_COUNT:
        return
    _write_jsonl(run_path, calls)
    _write_jsonl(scores_path, scores)


def evaluate_case(
    *,
    adapter: CountingAdapter,
    settings: Settings,
    run_id: str,
    task: TaskName,
    case: Case,
    gold: GoldLabel,
    model: ModelConfig,
) -> tuple[
    OutputRecord,
    list[ScoreRecord],
    list[UsageRecord],
    list[CallRecord],
    StrictModel | None,
]:
    adapter.reset()
    prompt_id, prompt_version = task_prompt(task)
    template = load(prompt_id, prompt_version)
    schema = OUTPUT_SCHEMAS[task]
    request = CompletionRequest(
        task=task,
        case_id=case.id,
        prompt_id=template.prompt_id,
        prompt_version=template.version,
        system=template.system,
        user_content=render_user(
            template,
            variables={
                "case_id": case.id,
                "schema_description": schema_description(schema),
            },
            untrusted=case.document_text,
        ),
        temperature=settings.temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    parsed: StrictModel | None = None
    error: str | None = None
    try:
        parsed = complete_structured(
            adapter,
            request,
            schema,
            run_id,
            max_repairs=settings.max_schema_repairs,
        )
    except (ValidationError, ValueError, RuntimeError) as exc:
        error = str(exc)

    usage_records = [
        usage_from_call_record(
            record, model_name=model.logical_name, call_index=adapter.call_index(record)
        )
        for record in adapter.records
    ]
    output_record = OutputRecord(
        run_id=run_id,
        task=task,
        case_id=case.id,
        model_name=model.logical_name,
        model_id=model.model_id,
        prompt_version=prompt_version,
        prompt_id=prompt_id,
        succeeded=parsed is not None,
        repairs=max(adapter.calls - 1, 0),
        output=None if parsed is None else parsed.model_dump(mode="json"),
        error=error,
    )
    if parsed is None:
        scores = failure_scores(
            run_id=run_id,
            task=task,
            case_id=case.id,
            model_name=model.logical_name,
            prompt_version=prompt_version,
            gold=gold,
            model_id=model.model_id,
            prompt_id=prompt_id,
        )
    else:
        scores = score_output(
            run_id=run_id,
            task=task,
            case_id=case.id,
            model_name=model.logical_name,
            prompt_version=prompt_version,
            output=parsed,
            gold=gold,
            source=case.document_text,
            model_id=model.model_id,
            prompt_id=prompt_id,
        )
    return output_record, scores, usage_records, list(adapter.records), parsed


def main() -> None:
    args = _parser().parse_args()
    counts = validate_corpus()
    if args.validate_only:
        print("Corpus valid: " + ", ".join(f"{task}={count}" for task, count in counts.items()))
        return

    run_id = cast(str | None, args.run_id)
    if run_id is None or not RUN_ID_PATTERN.fullmatch(run_id):
        raise SystemExit("--run-id is required and must use letters, numbers, '.', '_' or '-'")
    limit = cast(int | None, args.limit)
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be at least 1")

    selected_tasks: list[TaskName]
    if args.task:
        selected_tasks = [cast(TaskName, args.task)]
    else:
        selected_tasks = ["triage", "summarization", "extraction"]

    settings = Settings.from_env()
    selected_models = [cast(str, args.model)] if args.model else list(settings.models)
    run_dir = PROJECT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    usage_path = run_dir / "usage.jsonl"
    outputs_path = run_dir / "outputs.jsonl"
    scores_path = run_dir / "scores.jsonl"
    docs_run_path = PROJECT_ROOT / "docs" / "day5-run.jsonl"
    docs_scores_path = PROJECT_ROOT / "docs" / "day5-scores.jsonl"

    all_usage: list[UsageRecord] = []
    all_outputs: list[OutputRecord] = []
    all_scores: list[ScoreRecord] = []
    all_calls: list[CallRecord] = []
    total_cost = Decimal("0")
    validated_by_task_model: dict[tuple[TaskName, str], dict[str, StrictModel]] = defaultdict(dict)
    labels_by_task: dict[TaskName, list[GoldLabel]] = defaultdict(list)

    adapters: dict[str, CountingAdapter] = {
        model_name: CountingAdapter(OllamaAdapter(settings.models[model_name].model_id))
        for model_name in selected_models
    }

    for task in selected_tasks:
        pairs = load_cases(task)
        if limit is not None:
            pairs = pairs[:limit]
        labels_by_task[task] = [gold for _case, gold in pairs]
        prompt_id, prompt_version = task_prompt(task)
        for model_name in selected_models:
            model = settings.models[model_name]
            adapter = adapters[model_name]
            for case, gold in pairs:
                if total_cost >= settings.per_run_cap_usd:
                    raise SystemExit(
                        f"Per-run cost cap reached before {task}/{model_name}/{case.id}"
                    )
                output_record, case_scores, usage_records, call_records, parsed = evaluate_case(
                    adapter=adapter,
                    settings=settings,
                    run_id=run_id,
                    task=task,
                    case=case,
                    gold=gold,
                    model=model,
                )
                for usage_record, call_record in zip(usage_records, call_records, strict=True):
                    append_jsonl(usage_path, usage_record)
                    all_usage.append(usage_record)
                    total_cost += usage_record.cost_usd
                    all_calls.append(call_record)
                if parsed is not None:
                    validated_by_task_model[(task, model_name)][case.id] = parsed
                append_jsonl(outputs_path, output_record)
                all_outputs.append(output_record)
                for score in case_scores:
                    append_jsonl(scores_path, score)
                    all_scores.append(score)
                print(
                    f"{task:13} {model_name:8} {case.id:5} "
                    f"{prompt_label(task, model_name, prompt_version):22} "
                    f"{'ok' if output_record.succeeded else 'failed'}"
                    f" repairs={output_record.repairs}",
                    flush=True,
                )

    for task in selected_tasks:
        if task == "triage":
            continue
        prompt_id, prompt_version = task_prompt(task)
        for model_name in selected_models:
            model = settings.models[model_name]
            version_scores = score_version_selection(
                run_id=run_id,
                task=task,
                model_name=model_name,
                prompt_version=prompt_version,
                labels=labels_by_task[task],
                outputs=validated_by_task_model[(task, model_name)],
                model_id=model.model_id,
                prompt_id=prompt_id,
            )
            for score in version_scores:
                append_jsonl(scores_path, score)
                all_scores.append(score)

    write_day5_evidence(
        outputs=all_outputs,
        calls=all_calls,
        scores=all_scores,
        run_path=docs_run_path,
        scores_path=docs_scores_path,
    )

    write_reports(
        run_id=run_id,
        models=selected_models,
        usage=all_usage,
        outputs=all_outputs,
        scores=all_scores,
        report_path=PROJECT_ROOT / "reports" / "comparison.md",
        decision_path=PROJECT_ROOT / "docs" / "model-decision.md",
    )
    print(f"Report: {PROJECT_ROOT / 'reports' / 'comparison.md'}")
    print("Recorded provider cost: $0.00")


if __name__ == "__main__":
    main()
