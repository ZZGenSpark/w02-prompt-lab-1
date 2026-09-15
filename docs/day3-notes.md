Run `9a251780-4ea1-482b-8f6e-023c23c72dbe` on `mistral:7b` at temperature `0.0`. All 12 summarization cases validated after one semantic repair. All 12 extraction cases validated on the first call.

- summarization repair rate: 12/12 (100%)
- extraction repair rate: 0/12 (0%)
- example leakage count: 0
- citation-existence failure count: 1

The most common first-call failure on summarization was a schema-shaped reply rather than a `SummarizationOutput` instance; the repair prompt now asks for a JSON instance only and requires the `value` key on absent evidence fields, so S12 validated after that one attempt. One present field still failed the heading check: S05 `effective_date` cited `2. Date`, which is not a section heading in the source.
