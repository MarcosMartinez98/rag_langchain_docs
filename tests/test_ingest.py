from langchain_core.documents import Document

from langchain_rag.ingest import content_to_document, split_documents


def test_content_to_document_preserves_source_url() -> None:
    documents = content_to_document(
        [
            {
                "page_content": "# Page title\nPage body.",
                "metadata": {
                    "source": "docs/page.mdx",
                    "title": "Page title",
                    "corpus_commit": "a" * 40,
                    "url": "/docs/page",
                },
            }
        ]
    )

    assert documents[0].metadata["url"] == "/docs/page"


def test_split_documents_assigns_repeatable_chunk_ids() -> None:
    page = Document(
        page_content=" ".join(["Documented API behavior."] * 100),
        metadata={
            "source": "docs/page.mdx",
            "title": "Page title",
            "corpus_commit": "a" * 40,
            "url": "/docs/page",
        },
    )

    first_index = split_documents([page])
    second_index = split_documents([page])

    assert len(first_index) > 1
    assert [chunk.id for chunk in first_index] == [chunk.id for chunk in second_index]
    assert len({chunk.id for chunk in first_index}) == len(first_index)
    assert [chunk.metadata["chunk_index"] for chunk in first_index] == list(
        range(len(first_index))
    )
    assert all(chunk.metadata["url"] == "/docs/page" for chunk in first_index)


def test_split_documents_adds_section_context_to_each_chunk() -> None:
    page = Document(
        page_content=(
            "# Agents\n\nIntroductory overview.\n\n"
            "## Tool use\n\n"
            + "An agent calls tools to complete tasks. " * 80
        ),
        metadata={
            "source": "docs/agents.mdx",
            "title": "Agents",
            "corpus_commit": "a" * 40,
        },
    )

    chunks = split_documents([page])
    tool_chunks = [
        chunk for chunk in chunks if chunk.metadata["section"] == "Agents > Tool use"
    ]

    assert tool_chunks
    assert all(
        chunk.page_content.startswith("Section: Agents > Tool use\n\n")
        for chunk in tool_chunks
    )
    assert all(chunk.metadata["source"] == "docs/agents.mdx" for chunk in tool_chunks)
