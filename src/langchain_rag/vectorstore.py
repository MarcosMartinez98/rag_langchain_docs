"""Index processed documentation chunks in the persistent Chroma vector store.

Run from the repository root:

    python .\\src\\langchain_rag\\vectorstore.py

Add ``--nomic-prefix`` to index the experimental collection with Nomic's
``search_document:`` and ``search_query:`` embedding prefixes:

    python .\\src\\langchain_rag\\vectorstore.py --nomic-prefix
"""

import argparse
from pathlib import Path

from config import (
    COLLECTION_NAME,
    COLLECTION_NAME_WITH_PREFIX,
    DATA_DIR,
    EMBEDDINGS_MODEL,
    FILE_NAME,
)
from ingest import ingest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class NomicTaskPrefixedEmbeddings(Embeddings):
    def __init__(self, model: str) -> None:
        self._embeddings = OllamaEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"search_document: {text}" for text in texts]
        return self._embeddings.embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(f"search_query: {text}")


def get_vector_store(*, use_nomic_prefix: bool = False) -> Chroma:
    if use_nomic_prefix:
        embeddings: Embeddings = NomicTaskPrefixedEmbeddings(EMBEDDINGS_MODEL)
        collection_name = COLLECTION_NAME_WITH_PREFIX
    else:
        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
        collection_name = COLLECTION_NAME

    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(Path(DATA_DIR) / ".chromaDB"),
        collection_name=collection_name,
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
    parser = argparse.ArgumentParser(
        description="Index the processed documentation corpus in Chroma."
    )
    parser.add_argument(
        "--nomic-prefix",
        action="store_true",
        help=(
            "Index into the experimental collection using Nomic's "
            "search_document/search_query prefixes."
        ),
    )
    arguments = parser.parse_args()

    path = DATA_DIR / "processed" / FILE_NAME
    chunks = ingest(path)
    if not chunks:
        raise ValueError(f"No chunks were created from {path}; index was not changed.")

    vectorstore = get_vector_store(use_nomic_prefix=arguments.nomic_prefix)
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
    collection_name = (
        COLLECTION_NAME_WITH_PREFIX if arguments.nomic_prefix else COLLECTION_NAME
    )
    print(
        f"Indexation completed in {collection_name}. "
        f"Stale chunks deleted: {removed_count}."
    )
