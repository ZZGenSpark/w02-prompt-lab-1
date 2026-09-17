from decimal import Decimal
from pathlib import Path

from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.report import write_reports


def _usage(*, model_name: str = "mistral", prompt_version: str = "v1") -> UsageRecord:
    return UsageRecord(
        run_id="demo",
        task="triage",
        case_id="T01",
        model_name=model_name,
        model_id="configured-a",
        prompt_version=prompt_version,
        attempt=1,
        kind="primary",
        status="success",
        prompt_tokens=100,
        completion_tokens=25,
        latency_ms=125.0,
        cost_usd=Decimal("0"),
    )


def test_report_is_generated_from_records(tmp_path: Path) -> None:
    usage = [_usage()]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="configured-a",
            prompt_version="v1",
            succeeded=True,
            repairs=0,
            output={"queue": "card_dispute"},
            case_input_tokens=100,
            case_output_tokens=25,
            case_latency_ms=125.0,
            case_cost_usd=Decimal("0"),
        )
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            prompt_version="v1",
            scorer_version="day5.v2",
            metric="queue_correct",
            numerator=1,
            denominator=1,
        )
    ]
    report = tmp_path / "comparison.md"
    decision = tmp_path / "model-decision.md"
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=decision,
    )
    text = report.read_text(encoding="utf-8")
    assert "1/1" in text
    assert "triage.v1" in text
    assert "Input tokens/case" in text
    assert "125" in text
    assert "Median call latency" in text
    assert "Median case latency" in text
    assert "12 cases per task" in text
    assert "$0.00" in text
    assert "Untested combinations" in text
    assert "mistral" in decision.read_text(encoding="utf-8")
    assert "Human-boundary re-verification" not in text


def test_report_keeps_call_latency_and_sums_case_latency(tmp_path: Path) -> None:
    usage = [
        _usage(),
        UsageRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="configured-a",
            prompt_version="v1",
            attempt=1,
            kind="repair",
            status="success",
            prompt_tokens=40,
            completion_tokens=10,
            latency_ms=200.0,
            cost_usd=Decimal("0"),
        ),
    ]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="configured-a",
            prompt_version="v1",
            succeeded=True,
            repairs=1,
            output={"queue": "card_dispute"},
            case_input_tokens=140,
            case_output_tokens=35,
            case_latency_ms=325.0,
            case_cost_usd=Decimal("0"),
        )
    ]
    report = tmp_path / "comparison.md"
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=usage,
        outputs=outputs,
        scores=[],
        report_path=report,
        decision_path=tmp_path / "model-decision.md",
    )
    text = report.read_text(encoding="utf-8")
    assert "Median call latency" in text
    assert "Median case latency" in text
    assert "325 ms" in text
    assert "200 ms" in text
    assert "1/1" in text  # repairs


def test_report_labels_qwen_rows_as_prompt_transfer(tmp_path: Path) -> None:
    report = tmp_path / "comparison.md"
    decision = tmp_path / "model-decision.md"
    write_reports(
        run_id="demo",
        models=["mistral", "qwen"],
        usage=[_usage(), _usage(model_name="qwen")],
        outputs=[
            OutputRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="mistral",
                model_id="configured-a",
                prompt_version="v1",
                succeeded=True,
                repairs=0,
                output={"queue": "card_dispute"},
            ),
            OutputRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="qwen",
                model_id="configured-b",
                prompt_version="v1",
                succeeded=True,
                repairs=0,
                output={"queue": "card_dispute"},
            ),
        ],
        scores=[],
        report_path=report,
        decision_path=decision,
    )
    text = report.read_text(encoding="utf-8")
    assert "triage.v1 transfer" in text
    assert "Prompt-transfer rows" in text
    decision_text = decision.read_text(encoding="utf-8")
    assert "triage.v1 transfer" in decision_text
    assert "mean" not in text.lower() or "rather than mean" in text


def test_report_includes_human_boundary_when_triage_is_scored(tmp_path: Path) -> None:
    report = tmp_path / "comparison.md"
    decision = tmp_path / "model-decision.md"
    write_reports(
        run_id="demo",
        models=["mistral", "qwen"],
        usage=[_usage(), _usage(model_name="qwen")],
        outputs=[
            OutputRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="mistral",
                model_id="configured-a",
                prompt_version="v1",
                succeeded=True,
                repairs=0,
                output={"queue": "card_dispute"},
            ),
            OutputRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="qwen",
                model_id="configured-b",
                prompt_version="v1",
                succeeded=True,
                repairs=0,
                output={"queue": "card_dispute"},
            ),
        ],
        scores=[
            ScoreRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="mistral",
                prompt_version="v1",
                scorer_version="day5.v2",
                metric="human_boundary_compliance",
                numerator=1,
                denominator=1,
            ),
            ScoreRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="qwen",
                prompt_version="v1",
                scorer_version="day5.v2",
                metric="human_boundary_compliance",
                numerator=1,
                denominator=1,
            ),
            ScoreRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="mistral",
                prompt_version="v1",
                scorer_version="day5.v2",
                metric="pii_leakage",
                numerator=0,
                denominator=1,
                lower_is_better=True,
            ),
            ScoreRecord(
                run_id="demo",
                task="triage",
                case_id="T01",
                model_name="qwen",
                prompt_version="v1",
                scorer_version="day5.v2",
                metric="pii_leakage",
                numerator=0,
                denominator=1,
                lower_is_better=True,
            ),
        ],
        report_path=report,
        decision_path=decision,
    )
    text = report.read_text(encoding="utf-8")
    assert "Human-boundary re-verification" in text
    assert "1/1" in text


def test_write_reports_preserves_filled_model_decision(tmp_path: Path) -> None:
    decision = tmp_path / "model-decision.md"
    decision.write_text(
        "# Model Decision Record\n\n- selected model: mistral\n",
        encoding="utf-8",
    )
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=[_usage()],
        outputs=[],
        scores=[],
        report_path=tmp_path / "comparison.md",
        decision_path=decision,
    )
    assert "selected model: mistral" in decision.read_text(encoding="utf-8")


def test_day5_model_decision_names_task_model_prompt_and_reopen() -> None:
    from promptlab.config import PROJECT_ROOT

    text = (PROJECT_ROOT / "docs" / "model-decision.md").read_text(encoding="utf-8")
    assert "day5-local-03" in text
    for task in ("Triage", "Summarization", "Extraction"):
        assert f"### {task}" in text
    assert "selected model: qwen" in text
    assert "`triage.v1`" in text
    assert "`summarize.v1`" in text
    assert "`extract.v2`" in text
    assert "rejected alternative" in text
    assert "condition that would reopen the decision" in text
    assert "transfer" in text
    assert "human_boundary_compliance" in text
