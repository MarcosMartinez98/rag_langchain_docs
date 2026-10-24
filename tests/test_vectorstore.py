from langchain_rag import vectorstore


class FakeOllamaEmbeddings:
    def __init__(self, model: str) -> None:
        self.model = model
        self.document_inputs: list[str] = []
        self.query_input = ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs = texts
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_input = text
        return [1.0]


def test_nomic_task_prefixes_are_only_added_for_embedding_calls(
    monkeypatch,
) -> None:
    fake_embeddings = FakeOllamaEmbeddings(model="nomic-embed-text")
    monkeypatch.setattr(
        vectorstore,
        "OllamaEmbeddings",
        lambda model: fake_embeddings,
    )

    embeddings = vectorstore.NomicTaskPrefixedEmbeddings("nomic-embed-text")

    assert embeddings.embed_documents(["Section: Agents\n\nChunk text"]) == [[1.0]]
    assert fake_embeddings.document_inputs == [
        "search_document: Section: Agents\n\nChunk text"
    ]

    assert embeddings.embed_query("How does an agent work?") == [1.0]
    assert fake_embeddings.query_input == "search_query: How does an agent work?"


def test_get_vector_store_selects_collection_and_embedding_strategy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChroma:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(vectorstore, "Chroma", FakeChroma)
    monkeypatch.setattr(
        vectorstore,
        "OllamaEmbeddings",
        lambda model: FakeOllamaEmbeddings(model),
    )

    vectorstore.get_vector_store()
    assert captured["collection_name"] == vectorstore.COLLECTION_NAME
    assert isinstance(captured["embedding_function"], FakeOllamaEmbeddings)

    vectorstore.get_vector_store(use_nomic_prefix=True)
    assert captured["collection_name"] == vectorstore.COLLECTION_NAME_WITH_PREFIX
    assert isinstance(
        captured["embedding_function"],
        vectorstore.NomicTaskPrefixedEmbeddings,
    )
