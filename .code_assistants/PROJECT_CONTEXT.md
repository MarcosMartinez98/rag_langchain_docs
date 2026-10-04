# Project context and agent instructions

## Purpose

This repository is intended to become a Python end-to-end retrieval-augmented generation (RAG) assistant for answering questions about the official LangChain and LangGraph documentation. Answers should be grounded in the selected documentation corpus and cite their source pages.

The project is also a learning and measurement exercise: establish a fixed evaluation set before optimizing retrieval or generation, measure the baseline, then compare deliberate changes against the same benchmark. Questions about obsolete and current APIs test whether answers follow the pinned documentation instead of unsupported prior knowledge. Questions whose answers are absent from the corpus should be rejected rather than guessed.

## Collaboration expectations

- The project owner writes the application code and makes architectural decisions. Act as a technical mentor and code reviewer; do not implement the ingestion pipeline, chains, evaluation system, or API unless explicitly asked.
- Work on one project phase at a time. Do not move to a later phase until the owner confirms the current phase is understood and working.
- Explain recommendations with measurable criteria where possible. Be explicit when a decision depends on the corpus or is only a hypothesis.
- Do not change the agreed stack without asking first.
- The owner may converse in Spanish, but repository artifacts must be in English: code, comments, documentation, prompts, evaluation data, and agent instructions.

## Planned stack

These are project intentions, not a guarantee that every component has been implemented:

- Python scripts organized into modules.
- LangChain 1.x and LCEL for deterministic chains; LangGraph for the agent phase.
- Persistent local Chroma vector store.
- Ollama `nomic-embed-text` embeddings.
- OpenAI chat models through `ChatOpenAI`.
- Pydantic structured output, `with_structured_output`, and LangSmith tracing.
- A modular layout beginning with configuration, vector store, ingestion, and chain modules, followed by evaluation, API, and agent modules as needed.

Do not assume planned modules, behavior, or dependencies exist. Inspect the relevant, narrow part of the working tree before relying on it.

## Documentation corpus

The upstream documentation checkout is `data/raw/docs`, cloned from `https://github.com/langchain-ai/docs`. Its pinned revision is recorded in `data/raw/docs.commit` (`e10670ebfb0ab8deb64a046c6ac558674e47e542`).

The agreed source scope is:

- `src/oss/langchain/**`, excluding `frontend/**` and `changelog-js.mdx`.
- `src/oss/langgraph/**`, excluding `frontend/**` (including its Markdown overview).
- `src/oss/python/releases/**`.
- `src/oss/python/migrate/**`.

The source pages are primarily MDX, not just Markdown. Pages may mix Python and JavaScript sections or import examples from `src/snippets`; parsing and citations must account for that instead of assuming every page is plain Markdown.

### Do not browse the raw corpus unnecessarily

`data/raw/docs` is the upstream source checkout, not the corpus that day-to-day RAG work should consume.

Before a parsed corpus exists, inspect only the specific source pages needed to implement or validate parsing. Do not recursively read or dump the raw documentation to understand the application.

Once the parsed corpus has been generated, use that parsed artifact as the working corpus for ingestion, retrieval, evaluation, chains, API, and agent work. **You do not need to read `data/raw` to start or continue those tasks.** Consult the raw checkout only when the task explicitly concerns parsing or extraction, source provenance, updating/reproducing the pinned corpus, or diagnosing a concrete mismatch between a parsed artifact and its source.

The parser lives at `src/langchain_rag/parser.py` and writes one normalized page per JSONL record to `data/processed/docs.jsonl` by default. Run it from the repository root with `python .\src\langchain_rag\parser.py`. It verifies the checkout against `data/raw/docs.commit`, retains the original source path and pinned commit in metadata, and logs pages with no extractable content. The parser output is page-level; later ingestion is responsible for chunking and indexing.

## Evaluation and phase order

The initial priority is Phase 0: create and review the evaluation questions before optimizing the system. The current candidate dataset is `eval/questions.jsonl`, with 40 English questions split evenly across:

1. Single-page factual questions.
2. Questions requiring multiple documents.
3. Questions about version changes.
4. Unanswerable questions that should be rejected.

Treat the question set as fixed once the owner approves and records its checksum. Do not edit questions, gold answers, or gold evidence to improve system scores. If a genuine dataset defect is found, report it to the owner rather than silently changing the benchmark.

The intended sequence is:

1. Freeze the evaluation dataset.
2. Build and measure the ingestion/retrieval/generation baseline.
3. Compare retrieval improvements one variable at a time using the same dataset.
4. Add product features such as conversation history, API, streaming, tracing, and tests.
5. Compare an agentic LangGraph RAG implementation against the classic baseline.

Do not skip ahead or treat a planned phase as already implemented.

## Working safely in this repository

- Check the relevant files and current working-tree status before editing.
- Preserve unrelated user changes; do not overwrite, revert, or reformat unrelated files.
- Use the parsed corpus for application tasks once available, and keep raw-corpus inspection narrowly scoped to corpus-specific work.
- Keep source citations traceable to stable repository paths and useful page/section metadata.
- Follow the repository's actual conventions and dependency manifests rather than assuming the planned stack has already been configured.
