Run `9d76ada0-df65-4e94-a65b-c335ac4d6564` on `mistral:7b` at temperature `0.0`. All 12 summarization cases and all 12 extraction cases used one semantic repair.

- summarization repair rate: 0/12 (0%)
- extraction repair rate: 8/12 (67%)
- example leakage count: 0
- citation-existence failure count: 0

The most common validation error was the model echoing the generated JSON Schema (`$defs`, `properties`, `$ref`) instead of a schema instance, so Pydantic rejected extra keys and missing evidence objects. The repair request included that validation error and asked the model to correct only those issues; extraction then produced a valid `PolicyExtraction` on 8 of 12 cases.
