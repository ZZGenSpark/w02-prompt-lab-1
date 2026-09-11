Run `adf04a9f-f7a6-45db-b244-d00e9e5c3570`. Shared `max_output_tokens=548`. Local provider charge is 0.0 for both; no dollar comparison.

| model | success | input tokens | output tokens | median latency | max latency |
| --- | --- | --- | --- | --- | --- |
| mistral:7b | 12/12 | 2775 | 1318 | 4966 ms | 9588 ms |
| qwen3:8b | 12/12 | 2427 | 4748 | 19550 ms | 26488 ms |

Both models completed all 12 cases with no truncation. Qwen produced about 3.6× the output tokens and roughly 4× the median latency of Mistral, largely due to hidden reasoning before its visible answer, while Mistral's shorter, direct completions kept it consistently faster.
