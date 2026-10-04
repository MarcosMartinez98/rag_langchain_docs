import json
from pathlib import Path

import pytest

from langchain_rag.parser import (
    discover_source_pages,
    parse_corpus,
    parse_document,
)


def test_parse_document_keeps_python_and_shared_content(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    page_path = docs_root / "src" / "oss" / "langchain" / "example.mdx"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        """---
title: Example page
url: "/oss/langchain/example"
---

Shared [documentation](/oss/langchain/guide) prose.

:::python
Python-only explanation.

```python
answer = 42
```
:::

:::js
JavaScript-only explanation.

```typescript
const answer = 42;
```
:::

<Tab title="Python SDK">
Python tab content.
</Tab>
<Tab title="JavaScript SDK">
JavaScript tab content.
</Tab>
""",
        encoding="utf-8",
    )

    page = parse_document(page_path, docs_root, "a" * 40)

    assert page is not None
    assert page.id == "src/oss/langchain/example.mdx"
    assert page.metadata == {
        "corpus_commit": "a" * 40,
        "source": "src/oss/langchain/example.mdx",
        "title": "Example page",
        "url": "/oss/langchain/example",
    }
    assert "# Example page" in page.page_content
    assert "Shared documentation prose." in page.page_content
    assert "/oss/langchain/guide" not in page.page_content
    assert "Python-only explanation." in page.page_content
    assert "Python tab content." in page.page_content
    assert "answer = 42" in page.page_content
    assert "JavaScript-only explanation." not in page.page_content
    assert "JavaScript tab content." not in page.page_content
    assert "const answer" not in page.page_content


def test_parse_document_expands_python_snippet_and_omits_javascript_snippet(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    page_path = docs_root / "src" / "oss" / "langchain" / "example.mdx"
    snippets = docs_root / "src" / "snippets" / "code-samples"
    page_path.parent.mkdir(parents=True)
    snippets.mkdir(parents=True)
    (snippets / "example-py.mdx").write_text(
        "```python\nprint('python snippet')\n```\n",
        encoding="utf-8",
    )
    page_path.write_text(
        """---
title: Snippet example
---
import ExamplePy from '/snippets/code-samples/example-py.mdx';
import ExampleJs from '/snippets/code-samples/example-js.mdx';

<ExamplePy />
<ExampleJs />
""",
        encoding="utf-8",
    )

    page = parse_document(page_path, docs_root, "b" * 40)

    assert page is not None
    assert "print('python snippet')" in page.page_content
    assert "<ExamplePy" not in page.page_content
    assert "<ExampleJs" not in page.page_content


def test_parse_document_skips_pages_without_extractable_content(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    docs_root = tmp_path / "docs"
    page_path = docs_root / "src" / "oss" / "langchain" / "empty.mdx"
    page_path.parent.mkdir(parents=True)
    page_path.write_text("---\ntitle: Empty page\n---\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        page = parse_document(page_path, docs_root, "c" * 40)

    assert page is None
    assert "Skipping page with no extractable content" in caplog.text


def test_discover_source_pages_applies_the_selected_scope_and_exclusions(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    included = [
        "src/oss/langchain/agents.mdx",
        "src/oss/langgraph/graph-api.mdx",
        "src/oss/python/releases/langchain-v1.mdx",
        "src/oss/python/migrate/langgraph-v1.mdx",
    ]
    excluded = [
        "src/oss/langchain/frontend/overview.mdx",
        "src/oss/langgraph/frontend/overview.md",
        "src/oss/langchain/changelog-js.mdx",
        "src/oss/javascript/langchain/agents.mdx",
    ]
    for relative_path in included + excluded:
        path = docs_root / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Example\n\nPage content.\n", encoding="utf-8")

    source_paths = discover_source_pages(docs_root)
    relative_paths = {
        path.relative_to(docs_root).as_posix() for path in source_paths
    }

    assert relative_paths == set(included)


def test_parse_corpus_writes_one_json_record_per_page(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    for scope in (
        "src/oss/langchain",
        "src/oss/langgraph",
        "src/oss/python/releases",
        "src/oss/python/migrate",
    ):
        (docs_root / Path(scope)).mkdir(parents=True, exist_ok=True)
    page_path = docs_root / "src" / "oss" / "langchain" / "example.mdx"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        "---\ntitle: Example\n---\nA complete page.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "processed" / "docs.jsonl"

    pages = parse_corpus(docs_root, output_path, "d" * 40)
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(pages) == len(records) == 1
    assert records[0]["id"] == "src/oss/langchain/example.mdx"
    assert records[0]["page_content"] == "# Example\n\nA complete page."
    assert records[0]["metadata"]["corpus_commit"] == "d" * 40
