Run `ad7e5368-495a-4fa0-a72c-1e8f2a7b714c`. Shared `max_output_tokens=1024`. Local provider charge is 0.0 for both; no dollar comparison.

| model | success | input tokens | output tokens | median latency | max latency |
| --- | --- | --- | --- | --- | --- |
| mistral:7b | 12/12 | 2775 | 1313 | 3718 ms | 7682 ms |
| qwen3:8b | 12/12 | 2427 | 4748 | 16597 ms | 24273 ms |

Both models completed all 12 cases with no truncation. Qwen produced about 3.6× the output tokens and roughly 4.5× the median latency of Mistral, largely due to hidden reasoning before its visible answer, while Mistral's shorter, direct completions kept it consistently faster.
