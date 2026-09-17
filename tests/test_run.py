from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.config import DEFAULT_MODEL_A, DEFAULT_MODEL_B, ModelConfig, Settings
from promptlab.corpus import Case, GoldLabel
from promptlab.prompts import prompt_label, task_prompt
from promptlab.records import OutputRecord, ScoreRecord
from promptlab.run import (
    FULL_EVAL_COUNT,
    MAX_OUTPUT_TOKENS,
    CountingAdapter,
    evaluate_case,
    usage_from_call_record,
    usage_kind,
    write_day5_evidence,
)
from promptlab.usage import CallRecord


def _call_record(*, attempt: int = 1, error_type: str | None = None) -> CallRecord:
    return CallRecord(
        record_id=str(uuid4()),
        run_id="run-day5",
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id="configured-a",
        task="triage",
        case_id="T01",
        prompt_id="triage",
        prompt_version="v1",
        attempt=attempt,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=None,
        latency_ms=100,
        cost_usd=0.0,
        stop_reason="stop",
        error_type=error_type,
        response_text="{}",
    )


class FakeAdapter:
    provider = "ollama"

    def __init__(self, texts: list[str], model_id: str = "configured-a") -> None:
        self.model_id = model_id
        self._texts = texts
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        index = len(self.requests)
        self.requests.append(request)
        text = self._texts[min(index, len(self._texts) - 1)]
        record = _call_record(attempt=1)
        return CompletionResult(
            succeeded=True,
            text=text,
            error_type=None,
            records=[record],
        )


VALID_TRIAGE = (
    '{"queue":"card_dispute","escalation_required":false,"confidence":0.8,'
    '"rationale":"Duplicate merchant charge.",'
    '"draft_reply":"A specialist will review this request.",'
    '"human_review_required":true,"customer_outcome":null}'
)


def test_task_prompt_map_uses_day5_versions() -> None:
    assert task_prompt("summarization") == ("summarize", "v1")
    assert task_prompt("extraction") == ("extract", "v2")
    assert task_prompt("triage") == ("triage", "v1")


def test_prompt_label_marks_qwen_as_transfer() -> None:
    assert prompt_label("summarization", "mistral", "v1") == "summarize.v1"
    assert prompt_label("summarization", "qwen", "v1") == "summarize.v1 transfer"
    assert prompt_label("extraction", "qwen", "v2") == "extract.v2 transfer"


def test_usage_kind_separates_transport_retries_from_repairs() -> None:
    assert usage_kind(1, 1) == "primary"
    assert usage_kind(1, 2) == "transport_retry"
    assert usage_kind(2, 1) == "repair"
    assert usage_kind(2, 2) == "repair_retry"


def test_usage_from_call_record_preserves_zero_cost_and_join_fields() -> None:
    record = _call_record(attempt=2, error_type="TransientProviderError")
    usage = usage_from_call_record(record, model_name="mistral", call_index=1)
    assert usage.kind == "transport_retry"
    assert usage.status == "transport_error"
    assert usage.cost_usd == Decimal("0")
    assert usage.model_id == record.model_id
    assert usage.prompt_version == "v1"


def test_evaluate_case_goes_through_complete_structured() -> None:
    settings = Settings.from_env()
    fake = FakeAdapter([VALID_TRIAGE])
    adapter = CountingAdapter(fake)
    case = Case(id="T01", task="triage", document_text="duplicate charge")
    gold = GoldLabel(
        id="T01",
        task="triage",
        expected_queue="card_dispute",
        expected_escalation=False,
    )
    model = ModelConfig(logical_name="mistral", model_id="configured-a")

    output, scores, usage, calls, parsed = evaluate_case(
        adapter=adapter,
        settings=settings,
        run_id="run-day5",
        task="triage",
        case=case,
        gold=gold,
        model=model,
    )

    assert parsed is not None
    assert output.succeeded is True
    assert output.repairs == 0
    assert output.prompt_id == "triage"
    assert output.prompt_version == "v1"
    assert fake.requests[0].prompt_id == "triage"
    assert fake.requests[0].prompt_version == "v1"
    assert "duplicate charge" in fake.requests[0].user_content
    assert {score.metric for score in scores} >= {"queue_correct", "pii_leakage"}
    assert usage[0].kind == "primary"
    assert calls[0].provider == "ollama"
    assert output.model_id == "configured-a"
    assert output.case_latency_ms == usage[0].latency_ms
    assert output.case_input_tokens == usage[0].prompt_tokens
    assert output.case_output_tokens == usage[0].completion_tokens


def test_evaluate_case_counts_a_schema_repair() -> None:
    settings = Settings.from_env()
    fake = FakeAdapter(["not-json", VALID_TRIAGE])
    adapter = CountingAdapter(fake)
    case = Case(id="T01", task="triage", document_text="duplicate charge")
    gold = GoldLabel(
        id="T01",
        task="triage",
        expected_queue="card_dispute",
        expected_escalation=False,
    )
    output, _scores, usage, _calls, parsed = evaluate_case(
        adapter=adapter,
        settings=settings,
        run_id="run-day5",
        task="triage",
        case=case,
        gold=gold,
        model=ModelConfig(logical_name="qwen", model_id="configured-b"),
    )
    assert parsed is not None
    assert output.repairs == 1
    assert output.succeeded is True
    assert usage[0].kind == "primary"
    assert usage[1].kind == "repair"
    assert output.case_latency_ms == usage[0].latency_ms + usage[1].latency_ms
    assert output.case_input_tokens == usage[0].prompt_tokens + usage[1].prompt_tokens
    assert output.case_output_tokens == (
        usage[0].completion_tokens + usage[1].completion_tokens
    )
    assert output.case_cost_usd == Decimal("0")


def test_write_day5_evidence_only_after_72_evals(tmp_path: Path) -> None:
    run_path = tmp_path / "day5-run.jsonl"
    scores_path = tmp_path / "day5-scores.jsonl"
    run_path.write_text("keep\n", encoding="utf-8")
    outputs = [
        OutputRecord(
            run_id="day5-local-01",
            task="triage",
            case_id=f"T{index:02d}",
            model_name="mistral",
            model_id="configured-a",
            prompt_version="v1",
            prompt_id="triage",
            succeeded=True,
            repairs=0,
            output={},
        )
        for index in range(FULL_EVAL_COUNT - 1)
    ]
    write_day5_evidence(
        outputs=outputs,
        calls=[_call_record()],
        scores=[],
        run_path=run_path,
        scores_path=scores_path,
    )
    assert run_path.read_text(encoding="utf-8") == "keep\n"
    assert not scores_path.exists()

    outputs.append(
        OutputRecord(
            run_id="day5-local-01",
            task="triage",
            case_id="T72",
            model_name="mistral",
            model_id="configured-a",
            prompt_version="v1",
            prompt_id="triage",
            succeeded=True,
            repairs=0,
            output={},
        )
    )
    score = ScoreRecord(
        run_id="day5-local-01",
        task="triage",
        case_id="T01",
        model_name="mistral",
        prompt_version="v1",
        scorer_version="day5.v2",
        metric="queue_correct",
        numerator=1,
        denominator=1,
    )
    write_day5_evidence(
        outputs=outputs,
        calls=[_call_record()],
        scores=[score],
        run_path=run_path,
        scores_path=scores_path,
    )
    run_text = run_path.read_text(encoding="utf-8")
    score_text = scores_path.read_text(encoding="utf-8")
    assert "keep" not in run_text
    assert '"run_id":"run-day5"' in run_text
    assert '"run_id":"day5-local-01"' in score_text


def test_day5_docs_share_run_id_and_cover_72_evals() -> None:
    import json

    from promptlab.config import PROJECT_ROOT

    run_path = PROJECT_ROOT / "docs" / "day5-run.jsonl"
    scores_path = PROJECT_ROOT / "docs" / "day5-scores.jsonl"
    runs = [
        json.loads(line)
        for line in run_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert runs
    assert scores
    run_ids = {row["run_id"] for row in runs} | {row["run_id"] for row in scores}
    assert len(run_ids) == 1
    evals = {(row["task"], row["model_id"], row["case_id"]) for row in runs}
    assert len(evals) == FULL_EVAL_COUNT


def test_validate_only_does_not_call_ollama(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from promptlab import run as run_mod

    monkeypatch.setattr("sys.argv", ["promptlab", "--validate-only"])
    run_mod.main()
    captured = capsys.readouterr().out
    assert "triage=12" in captured
    assert "summarization=12" in captured
    assert "extraction=12" in captured


def test_run_source_has_no_model_identifier_literals_or_direct_ollama() -> None:
    from promptlab import run as run_mod

    source = Path(run_mod.__file__).read_text(encoding="utf-8")
    assert DEFAULT_MODEL_A not in source
    assert DEFAULT_MODEL_B not in source
    assert "/api/generate" not in source
    assert "httpx" not in source
    assert "complete_structured" in source
    assert "OllamaAdapter" in source
