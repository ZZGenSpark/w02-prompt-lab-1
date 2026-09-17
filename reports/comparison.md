# Model Comparison

Run ID: `day5-local-03`

Shared evidence: `docs/day5-run.jsonl` and `docs/day5-scores.jsonl`. 84 call
records cover 72 task/model/case evaluations (3 tasks × 2 models × 12 cases).
Mistral summarization added 12 schema-repair attempts. Temperature is `0.0`.
Both models use `provider=ollama` and `cost_usd=0.0`. Local provider/API charge
is `$0.00`. Counts use denominators, not percentages. Latency uses median and
maximum, not mean. Call latency is one HTTP attempt. Case latency sums every
attempt for that case, including repairs and transport retries.

Prompt-transfer rows are labeled `transfer`. They measure that prompt on Qwen
with thinking disabled (`think=false`). They are not an adapted Qwen prompt and
are not a claim that Qwen is worse or better in general.

Every score row joins to call evidence on `run_id`, `case_id`, `task`,
`model_id`, `prompt_id`, and `prompt_version`.

## Extraction

| Model | Prompt | Required-evidence recall | Citation correctness | Invented/unsupported avoidance | Document status | Version selection | Valid outputs | Input tokens/case | Output tokens/case | Median call latency | Max call latency | n calls | Median case latency | Max case latency | n cases | Repairs | Retries | Final failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 71/72 | 73/73 | 10/12 | 9/12 | 1/1 | 12/12 | 2168.9 | 353.8 | 15950.5 ms | 19643 ms | 12 | 15950.5 ms | 19643 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v2 transfer | 71/72 | 72/72 | 11/12 | 11/12 | 1/1 | 12/12 | 1793.9 | 287.4 | 14420 ms | 18868 ms | 12 | 14420 ms | 18868 ms | 12 | 0/12 | 0 | 0 |

Missed recoverable values (kept separate from invented values): E05
`beneficial_ownership_threshold` on both models (71/72). Invented/unsupported
values: Mistral invented `jurisdictions` on E04 and E10; Qwen invented
`jurisdictions` on E10 only. Status errors: Mistral E01 (`valid` vs
`superseded`), E04 and E10 (`valid` vs `contradictory`); Qwen E10 only.
`version_selection_accuracy` scores `select_current_version(...)` in Python, not
a model opinion about which document is current.

## Summarization

| Model | Prompt | Required-evidence recall | Citation correctness | Invented/unsupported avoidance | Document status | Version selection | Valid outputs | Input tokens/case | Output tokens/case | Median call latency | Max call latency | n calls | Median case latency | Max case latency | n cases | Repairs | Retries | Final failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 59/60 | 62/63 | 8/12 | 7/12 | 1/1 | 12/12 | 2634.5 | 731.8 | 14619.5 ms | 23596 ms | 24 | 28367.5 ms | 36650 ms | 12 | 12/12 | 0 | 0 |
| qwen | summarize.v1 transfer | 60/60 | 65/65 | 7/12 | 9/12 | 1/1 | 12/12 | 1028.2 | 249.1 | 12492 ms | 16098 ms | 12 | 12492 ms | 16098 ms | 12 | 0/12 | 0 | 0 |

Mistral produced a valid object for every case only after one schema repair
(24 call records, 12/12 repairs). Citation miss: S05 `effective_date`. Recall
miss: S04 `exceptions`. Invented `required_steps` on S04 and S09; invented
`effective_date` on S05 and S12. Status 7/12 (S01, S04, S05, S09, S12). Qwen
needed no repairs. Invented `required_steps` on S04 and S09; invented `purpose`
on S05; invented `version` and `purpose` on S12. Status 9/12 (S01, S04, S05).
Version selection 1/1 on both.

## Triage

| Model | Prompt | Queue (routing) | Escalation | Missed escalation ↓ | Unnecessary escalation ↓ | Human-boundary compliance | PII leakage ↓ | Valid outputs | Input tokens/case | Output tokens/case | Median call latency | Max call latency | n calls | Median case latency | Max case latency | n cases | Repairs | Retries | Final failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 9/12 | 8/12 | 0/12 | 4/12 | 12/12 | 0/12 | 12/12 | 917.8 | 150 | 6502.5 ms | 9064 ms | 12 | 6502.5 ms | 9064 ms | 12 | 0/12 | 0 | 0 |
| qwen | triage.v1 transfer | 10/12 | 11/12 | 0/12 | 1/12 | 12/12 | 0/12 | 12/12 | 771.6 | 130.4 | 6192.5 ms | 9839 ms | 12 | 6192.5 ms | 9839 ms | 12 | 0/12 | 0 | 0 |

Mistral queue misses: T06 (`fraud_report` vs `escalate`), T07 (`lending` vs
`escalate`), T08 (`fraud_report` vs `escalate`). Unnecessary escalations: T02,
T05, T09, T10. Qwen queue misses: T07 (`complaint` vs `escalate`), T08
(`fraud_report` vs `escalate`). Unnecessary escalation: T09. Neither model
missed a required escalation (0/12). Day 4 selected `triage.v1` over
`triage.v2`; this run uses that version and does not rewrite that constraint.

## Human-boundary re-verification

Triage `draft_reply` was scored against the Day 4 human-boundary rule for
**both Mistral and Qwen**. No committed `draft_reply` promised a refund,
approved or denied a claim, stated the issue was resolved, or implied a final
customer outcome.

| Model | Prompt | human_boundary_compliance | pii_leakage ↓ | Valid outputs |
| --- | --- | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | 0/12 | 12/12 |
| qwen | triage.v1 transfer | 12/12 | 0/12 | 12/12 |

Models tested: `mistral:7b` and `qwen3:8b`. A missing-output human-boundary miss
would be a failed structured call, not a prohibited phrase in a committed
`draft_reply`. This run had no missing outputs.

## Recommendation

Each row is a task-level choice from this 12-case run. It is not a universal
model ranking.

| Task | Model | Prompt version | Reason | Reopen if |
| --- | --- | --- | --- | --- |
| triage | qwen | `triage.v1 transfer` | 12/12 valid; queue 10/12 vs 9/12; escalation 11/12 vs 8/12; unnecessary escalation 1/12 vs 4/12; missed escalation 0/12; human-boundary 12/12; PII 0/12; 0 repairs. Median call latency 6192.5 ms vs 6502.5 ms (n=12). Same Day 4 `v1` prompt on Qwen, not a new triage version. | An adapted Qwen or Mistral triage version is measured, the 12-case set is replaced or expanded, or human-boundary or PII fails on the selected configuration. |
| summarization | qwen | `summarize.v1 transfer` | 12/12 valid with 0/12 repairs; required evidence 60/60; citations 65/65; status 9/12; version selection 1/1; fewer tokens and lower median call latency (12492 ms, n=12) than Mistral (14619.5 ms, n=24). | A Mistral prompt that validates without a 12/12 repair rate, a Qwen-adapted summarization version that reduces invented fields (S04, S05, S09, S12), or a larger case set. |
| extraction | qwen | `extract.v2 transfer` | 12/12 valid; required evidence 71/72 (same E05 miss); citations 72/72; invented/unsupported 11/12 vs 10/12; status 11/12 vs 9/12; version selection 1/1; fewer tokens and lower median call latency (14420 ms vs 15950.5 ms, n=12). | A Qwen-adapted extract version, a Mistral run that stops inventing E04/E10 `jurisdictions`, or a corpus larger than 12 cases. |

Rejected for each task: mistral on the same prompt version. See
`docs/model-decision.md`.

## Limits

- There are only 12 cases per task. Results are directional, not production-scale estimates.
- A one-case gap such as 11/12 versus 10/12 is not a universal model ranking.
- Prompt-transfer rows are identified in the Prompt column (`summarize.v1 transfer`, `extract.v2 transfer`, `triage.v1 transfer`).
- Untested combinations: none in this run. Every configured task/model pair produced records. Untested: any Qwen-adapted prompt version; `triage.v2` on Qwen; Qwen with thinking enabled.
- No production-volume reliability claim is being made.
- Local Ollama latency depends on lab hardware and is not a cloud SLA.
- Local provider/API charge is `$0.00`; input tokens, output tokens, per-call and per-case median/max latency, observation counts, repair rate, and retry/failure counts are the operational measurements.
