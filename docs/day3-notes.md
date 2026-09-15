Run `9afa9d77-f975-4f27-9471-0c1e4fdd051f` on `mistral:7b` at temperature `0.0`. All 12 summarization cases used one semantic repair. All 12 extraction cases validated on the first call.

- summarization repair rate: 11/12 (92%)
- extraction repair rate: 0/12 (0%)
- example leakage count: 0
- citation-existence failure count: 0

The most common first-call failure on summarization was still a schema-shaped reply rather than a `SummarizationOutput` instance; the repair prompt now asks for a JSON instance only, and 11 of 12 cases validated after that one attempt. S12 still failed after repair because absent evidence fields omitted the required `value` key.
