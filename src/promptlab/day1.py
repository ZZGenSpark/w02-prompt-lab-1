"""Day 1 runner: three Mistral extraction calls with usage recording."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord, append_record, compute_cost

CASE_IDS = ("E12", "E07", "E11")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 256
CASES_PATH = PROJECT_ROOT / "cases" / "extraction.jsonl"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"


def load_cases(path: Path, case_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted = set(case_ids)
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            continue
        case_id = case["id"]
        if case_id in wanted:
            found[str(case_id)] = case
    missing = [case_id for case_id in case_ids if case_id not in found]
    if missing:
        raise KeyError(f"Missing extraction cases: {', '.join(missing)}")
    return [found[case_id] for case_id in case_ids]


def call_ollama(
    settings: Settings,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": max_output_tokens,
            },
        },
        timeout=180.0,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Ollama response was not a JSON object")
    return payload, latency_ms


def record_from_payload(
    *,
    run_id: str,
    model_id: str,
    case_id: str,
    payload: dict[str, Any],
    latency_ms: int,
) -> CallRecord:
    input_tokens = int(payload["prompt_eval_count"])
    output_tokens = int(payload["eval_count"])
    done_reason = payload.get("done_reason")
    response_text = payload.get("response")
    return CallRecord(
        record_id=str(uuid.uuid4()),
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task="extraction",
        case_id=case_id,
        prompt_id="baseline",
        prompt_version="v0",
        attempt=1,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=None,
        latency_ms=latency_ms,
        cost_usd=compute_cost(model_id, input_tokens, output_tokens),
        stop_reason=done_reason if isinstance(done_reason, str) else None,
        error_type=None,
        response_text=response_text if isinstance(response_text, str) else None,
    )


def main() -> None:
    settings = Settings.from_env()
    model_id = settings.models["mistral"].model_id
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    run_id = str(uuid.uuid4())
    cases = load_cases(CASES_PATH, CASE_IDS)

    print(f"run_id={run_id} model_id={model_id}")
    for case in cases:
        case_id = str(case["id"])
        prompt = prompt_template.replace("{document_text}", str(case["source"]))
        payload, latency_ms = call_ollama(settings, model_id, prompt, MAX_OUTPUT_TOKENS)
        record = record_from_payload(
            run_id=run_id,
            model_id=model_id,
            case_id=case_id,
            payload=payload,
            latency_ms=latency_ms,
        )
        append_record(record, run_id)
        print(
            f"{case_id} input_tokens={record.input_tokens} "
            f"output_tokens={record.output_tokens} latency_ms={record.latency_ms} "
            f"stop_reason={record.stop_reason}"
        )


if __name__ == "__main__":
    main()
