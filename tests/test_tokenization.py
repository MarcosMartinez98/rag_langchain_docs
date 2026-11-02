from langchain_rag.tokenization import bm25_tokenize


def test_bm25_tokenize_normalizes_case_and_punctuation() -> None:
    assert bm25_tokenize("LangChain agent? LANGCHAIN.AGENTS") == [
        "langchain",
        "agent",
        "langchain",
        "agents",
    ]


def test_bm25_tokenize_preserves_underscores_in_identifiers() -> None:
    assert bm25_tokenize("create_agent search_query:") == [
        "create_agent",
        "search_query",
    ]
