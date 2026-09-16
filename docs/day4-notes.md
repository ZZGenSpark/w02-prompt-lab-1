# Day 4 Notes

run_id `cd0ca6bd-59cf-4fdb-908f-69673b58746f` · `mistral:7b` · temperature `0.0` · 24 observations · provider/API cost `$0.00`

triage.v1
queue correct: 9/12
escalation correct: 8/12
missed escalations: 0
unnecessary escalations: 4
human-boundary passes: 12/12

triage.v2
queue correct: 8/12
escalation correct: 9/12
missed escalations: 0
unnecessary escalations: 3
human-boundary passes: 12/12

changed-queue count: 2 (T09 unsupported→lending, T12 fraud_report→card_dispute; both regressions vs gold)

token difference: not in `docs/day4-run.jsonl` (`OutputRecord` has no token fields)
latency comparison: not in `docs/day4-run.jsonl` (`OutputRecord` has no latency fields)

The extra `analysis` field did not earn its overhead. Queue accuracy dropped 1/12 and escalation rose 1/12; both diffs are within noise on a 12-case set.
