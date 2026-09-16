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

provider/API cost: $0.00

output tokens per case (from `runs/cd0ca6bd-59cf-4fdb-908f-69673b58746f.jsonl`):

```
case  v1   v2
T01   174  200
T02   125  215
T03   174  177
T04   126  184
T05   149  199
T06   161  175
T07   153  197
T08   133  215
T09   150  148
T10   177  222
T11   127  160
T12   151  184
sum   1800 2276
```

token difference: v2 used 476 more output tokens than v1 (1800 vs 2276)

latency comparison:
median latency: v1 6546 ms, v2 8116 ms (+1570 ms)
maximum latency: v1 8854 ms, v2 9186 ms (+332 ms)
observation count: 24 call records (12 per prompt version; cost_usd 0.0 on every record)

The extra `analysis` field did not earn its overhead. Queue accuracy dropped 1/12 and escalation rose 1/12; both diffs are within noise on a 12-case set. v2 paid 476 extra output tokens and ~1.6 s extra median latency for that.
