# Model Decision Record

Run ID: `day5-local-03`

Task-level decisions below are taken only from `docs/day5-run.jsonl` and
`docs/day5-scores.jsonl` for this shared run. Do not select one universal model
solely because it leads on a different task. Qwen rows are **prompt-transfer**
results (`summarize.v1`, `extract.v2`, `triage.v1` written against Mistral).
They measure those prompts on Qwen with thinking disabled (`think=false`); they
do not measure an adapted Qwen prompt. Day 4 selected `triage.v1` over
`triage.v2`; that constraint is not rewritten here. Local Ollama `cost_usd` is
`0.0` on every call; provider/API charge is `$0.00`. 84 call records cover 72
task/model/case evaluations (Mistral summarization used one schema repair per
case). Every evidence row names the prompt version used.

## Evaluated models

- mistral (`mistral:7b`, `provider=ollama`)
- qwen (`qwen3:8b`, `provider=ollama`, thinking off)



## Evaluated configurations

- `extraction` — mistral — `extract.v2`
- `extraction` — qwen — `extract.v2 transfer`
- `summarization` — mistral — `summarize.v1`
- `summarization` — qwen — `summarize.v1 transfer`
- `triage` — mistral — `triage.v1`
- `triage` — qwen — `triage.v1 transfer`



## Evidence

Counts use denominators, not percentages. `n` is recorded call attempts.
Latency is median and maximum, not mean. Every configuration produced 12/12
valid structured outputs. No `TruncatedResponseError` occurred on this run.
Score records join to call records on `run_id`, `case_id`, `task`, `model_id`,
`prompt_id`, and `prompt_version`.

### Extraction


| Model   | Prompt              | Valid outputs | required_evidence_recall | citation_correctness | unsupported_field_avoidance | document_status_correct | version_selection_accuracy | Input tokens/case | Output tokens/case | Median latency | Max latency | n   | Repairs | Retries | Final failures |
| ------- | ------------------- | ------------- | ------------------------ | -------------------- | --------------------------- | ----------------------- | -------------------------- | ----------------- | ------------------ | -------------- | ----------- | --- | ------- | ------- | -------------- |
| mistral | extract.v2          | 12/12         | 71/72                    | 73/73                | 10/12                       | 9/12                    | 1/1                        | 2168.9            | 353.8              | 15950.5 ms     | 19643 ms    | 12  | 0/12    | 0       | 0              |
| qwen    | extract.v2 transfer | 12/12         | 71/72                    | 72/72                | 11/12                       | 11/12                   | 1/1                        | 1793.9            | 287.4              | 14420 ms       | 18868 ms    | 12  | 0/12    | 0       | 0              |


Missed recoverable field: E05 `beneficial_ownership_threshold` on both models.
Invented `jurisdictions`: Mistral E04 and E10; Qwen E10 only. Status errors:
Mistral E01 (valid vs superseded), E04 and E10 (valid vs contradictory); Qwen
E10 only. `select_current_version` scored 1/1 on both (Python rule, not a model
opinion). Missed values and invented values are kept separate.

### Summarization


| Model   | Prompt                | Valid outputs | required_evidence_recall | citation_correctness | unsupported_field_avoidance | document_status_correct | version_selection_accuracy | Input tokens/case | Output tokens/case | Median latency | Max latency | n   | Repairs | Retries | Final failures |
| ------- | --------------------- | ------------- | ------------------------ | -------------------- | --------------------------- | ----------------------- | -------------------------- | ----------------- | ------------------ | -------------- | ----------- | --- | ------- | ------- | -------------- |
| mistral | summarize.v1          | 12/12         | 59/60                    | 62/63                | 8/12                        | 7/12                    | 1/1                        | 2634.5            | 731.8              | 14619.5 ms     | 23596 ms    | 24  | 12/12   | 0       | 0              |
| qwen    | summarize.v1 transfer | 12/12         | 60/60                    | 65/65                | 7/12                        | 9/12                    | 1/1                        | 1028.2            | 249.1              | 12492 ms       | 16098 ms    | 12  | 0/12    | 0       | 0              |


Mistral validated every case after one schema repair (24 call records). Citation
miss: S05 `effective_date`. Recall miss: S04 `exceptions`. Invented
`required_steps` on S04 and S09; invented `effective_date` on S05 and S12.
Status 7/12 (S01, S04, S05, S09, S12). Qwen needed no repairs. Invented
`required_steps` on S04 and S09; invented `purpose` on S05; invented `version`
and `purpose` on S12. Status 9/12 (S01, S04, S05). Version selection 1/1 on both.

### Triage


| Model   | Prompt             | Valid outputs | queue_correct | escalation_correct | missed_escalation ↓ | unnecessary_escalation ↓ | human_boundary_compliance | pii_leakage ↓ | Input tokens/case | Output tokens/case | Median latency | Max latency | n   | Repairs | Retries | Final failures |
| ------- | ------------------ | ------------- | ------------- | ------------------ | ------------------- | ------------------------ | ------------------------- | ------------- | ----------------- | ------------------ | -------------- | ----------- | --- | ------- | ------- | -------------- |
| mistral | triage.v1          | 12/12         | 9/12          | 8/12               | 0/12                | 4/12                     | 12/12                     | 0/12          | 917.8             | 150                | 6502.5 ms      | 9064 ms     | 12  | 0/12    | 0       | 0              |
| qwen    | triage.v1 transfer | 12/12         | 10/12         | 11/12              | 0/12                | 1/12                     | 12/12                     | 0/12          | 771.6             | 130.4              | 6192.5 ms      | 9839 ms     | 12  | 0/12    | 0       | 0              |


Mistral queue misses: T06 (`fraud_report` vs `escalate`), T07 (`lending` vs
`escalate`), T08 (`fraud_report` vs `escalate`). Unnecessary escalations: T02,
T05, T09, T10. Qwen queue misses: T07 (`complaint` vs `escalate`), T08
(`fraud_report` vs `escalate`). Unnecessary escalation: T09. Neither model
missed a required escalation (0/12). T09 validated for both models on this run.

## Human-boundary re-verification

Triage `draft_reply` was scored for **both Mistral and Qwen**. No committed
`draft_reply` promised a refund, approved or denied a claim, stated the issue
was resolved, or implied a final customer outcome.

- mistral / `triage.v1`: human_boundary_compliance **12/12**, pii_leakage **0/12**
- qwen / `triage.v1 transfer`: human_boundary_compliance **12/12**, pii_leakage **0/12**

Models tested: `mistral:7b` and `qwen3:8b`.

## Task decisions



### Triage

- selected model: qwen
- prompt version: `triage.v1 transfer`
- measured reason: 12/12 valid outputs; queue 10/12 vs 9/12; escalation 11/12
vs 8/12; unnecessary escalation 1/12 vs 4/12; missed escalation 0/12;
human-boundary 12/12; PII 0/12; 0 repairs, 0 retries, 0 final failures.
Median latency 6192.5 ms vs 6502.5 ms (n=12). Day 4 already kept `v1`; this
row is that same prompt on Qwen, not a new triage version.
- rejected alternative(s): mistral / `triage.v1` — 12/12 valid and human-boundary
12/12, but worse routing (T06–T08) and four unnecessary escalations on this
12-case set.
- condition that would reopen the decision: an adapted Qwen or Mistral triage
prompt version is measured, the 12-case set is replaced/expanded, or
human-boundary or PII fails on the selected configuration.



### Summarization

- selected model: qwen
- prompt version: `summarize.v1 transfer`
- measured reason: 12/12 valid with 0/12 repairs; required evidence 60/60;
citations 65/65; status 9/12; version selection 1/1; lower token use and
median call latency (12492 ms, n=12) than Mistral (14619.5 ms, n=24); median
case latency 12492 ms vs 28367.5 ms for Mistral (Mistral doubled because every
case required one schema repair call) on this run.
- rejected alternative(s): mistral / `summarize.v1` — also 12/12 valid, but
12/12 repairs, 59/60 recall, 62/63 citations, status 7/12, and higher output
tokens. Qwen still invented unsupported fields on S04, S05, S09, and S12;
that is recorded, not ignored.
- condition that would reopen the decision: a Mistral prompt that validates
without a 12/12 repair rate, a Qwen-adapted summarization version that
reduces invented fields, or a larger case set.



### Extraction

- selected model: qwen
- prompt version: `extract.v2 transfer`
- measured reason: 12/12 valid; required evidence 71/72 (same E05 miss as
Mistral); citations 72/72; invented/unsupported 11/12 vs 10/12; status 11/12
vs 9/12; version selection 1/1; fewer tokens and lower median latency
(14420 ms vs 15950.5 ms, n=12).
- rejected alternative(s): mistral / `extract.v2` — same 71/72 recall, but
invented `jurisdictions` on both E04 and E10 and status wrong on three cases.
One-to-two case gaps on 12 cases are directional only.
- condition that would reopen the decision: a Qwen-adapted extract version, a
Mistral run that stops inventing E04/E10 jurisdictions, or a corpus larger
than 12 cases.

These three selections all name Qwen on **transferred** Mistral-developed
prompts after thinking was turned off. That is the 12-case result for
`day5-local-03`, not a universal model ranking and not a claim about Qwen's
best adapted prompts.

## Review triggers

- A new prompt version is measured (including any Qwen-adapted version).
- The 12-case sample is replaced or expanded.
- Human-boundary or PII leakage fails on a previously selected model.
- Qwen thinking is turned back on, or `max_output_tokens` changes, and the
transfer rows are re-scored.
- `select_current_version` or gold labels for a version group change.

