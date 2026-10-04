# LangChain Documentation RAG

This project is a Python RAG assistant for answering questions about the official LangChain and LangGraph documentation, with source-page citations. Its pre-freeze evaluation candidate covers answerable and unanswerable questions across factual retrieval, definitions, procedures, conditions, comparisons, synthesis, version changes, and explicit negatives.

## Parse the documentation corpus

The parser reads the pinned checkout in `data/raw/docs` and writes one normalized page per JSONL record to `data/processed/docs.jsonl`. It keeps shared documentation text and Python examples, filters JavaScript-specific sections and examples, and preserves source paths and commit metadata for citations. Chunking and indexing are separate ingestion steps.

From the repository root, run:

```powershell
python .\src\langchain_rag\parser.py
```

The parser verifies that the documentation checkout matches `data/raw/docs.commit` and refuses to label modified source pages or snippets with the pinned commit. Pages with no extractable content are logged and skipped.

After parsing, use `data/processed/docs.jsonl` for ingestion and evaluation. Do not scan `data/raw` for normal RAG work; the raw checkout is only needed for parsing, provenance, or corpus updates.

## Evaluation dataset

`eval/questions.jsonl` is the candidate benchmark tied to the `corpus_commit` in each record. Each question records its answerability, primary intent, and whether answering it requires one evidence unit or multiple evidence hops. Expected sources are labeled `required` when they establish a required fact and `optional` when they only provide additional context.

Review and validate the candidate against `data/processed/docs.jsonl` before freezing it and recording its checksum. Do not modify a frozen benchmark to improve system scores.
