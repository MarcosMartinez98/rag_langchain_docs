from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

if __package__:
    from .config import DATA_DIR, EMBEDDINGS_MODEL, FILE_NAME
    from .ingest import ingest
else:
    from config import DATA_DIR, EMBEDDINGS_MODEL, FILE_NAME
    from ingest import ingest


def get_vector_store(embeddings_model: str, data_dir: str | Path) -> Chroma:
    embeddings = OllamaEmbeddings(model=embeddings_model)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(Path(data_dir) / ".chromaDB"),
        collection_name="rag_langchain_documents"
    )


def _remove_stale_chunks(vectorstore: Chroma, expected_ids: set[str]) -> int:
    existing = vectorstore.get(include=["metadatas"])
    stale_ids = [
        document_id
        for document_id, metadata in zip(
            existing["ids"], existing["metadatas"], strict=True
        )
        if metadata is not None
        and "corpus_commit" in metadata
        and document_id not in expected_ids
    ]
    if stale_ids:
        vectorstore.delete(ids=stale_ids)
    return len(stale_ids)


if __name__ == "__main__":
    path = DATA_DIR / "processed" / FILE_NAME
    chunks = ingest(path)
    if not chunks:
        raise ValueError(f"No chunks were created from {path}; index was not changed.")

    vectorstore = get_vector_store(EMBEDDINGS_MODEL, DATA_DIR)
    batch_size = 100
    expected_ids = set()
    for chunk in chunks:
        if chunk.id is None:
            raise ValueError("Ingestion produced a chunk without a stable ID.")
        expected_ids.add(chunk.id)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(
            f"Processing chunks {i} → {min(i + batch_size, len(chunks))} "
            f"of {len(chunks)}"
        )
        vectorstore.add_documents(batch)

    removed_count = _remove_stale_chunks(vectorstore, expected_ids)
    print(f"Indexation completed. Stale chunks deleted: {removed_count}.")
