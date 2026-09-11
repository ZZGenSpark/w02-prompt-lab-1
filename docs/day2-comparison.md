Run `5bba76b3-947f-4eea-8bf7-b22daee99677`. Shared `max_output_tokens=400`. Local provider charge is 0.0 for both; no dollar comparison.

| model | success | input tokens | output tokens | median latency | max latency |
| --- | --- | --- | --- | --- | --- |
| mistral:7b | 12/12 | 2775 | 1318 | 4002 ms | 8207 ms |
| qwen3:8b | 8/12 | 2427 | 4400 | 17050 ms | 18204 ms |

Qwen emitted about 3× the output tokens and 4× the median latency. S03, S06, S07, and S08 hit `TruncatedResponseError` (`stop_reason=length`, empty text); Mistral always stopped under the same ceiling.
