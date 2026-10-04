# Repository instructions

Before working on this repository, read [`.code_assistants/PROJECT_CONTEXT.md`](.code_assistants/PROJECT_CONTEXT.md). It describes the project goals, collaboration expectations, corpus scope, phase order, and when the upstream `data/raw` checkout should—and should not—be read.

In particular, after the parsed corpus exists, use it for normal ingestion, retrieval, evaluation, chain, API, and agent work. Do not browse `data/raw` unless the task specifically requires source parsing, provenance, corpus updates, or diagnosis of a source-to-parsed-content mismatch.
