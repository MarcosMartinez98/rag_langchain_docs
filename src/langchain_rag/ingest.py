import json
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

if __package__:
    from .config import DATA_DIR, FILE_NAME
else:
    from config import DATA_DIR, FILE_NAME

def get_content(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        content = file.read()

    return [json.loads(line) for line in content.splitlines()]

def content_to_document(content: list[dict]) -> list[Document]:
    result = []
    for doc in content:
        doc_metadata = doc["metadata"]

        metadata = {
            "source": doc_metadata["source"],
            "title": doc_metadata["title"],
            "corpus_commit": doc_metadata["corpus_commit"],
        }
        if "url" in doc_metadata:
            metadata["url"] = doc_metadata["url"]

        page_content = doc["page_content"]

        result.append(
            Document(
                metadata=metadata,
                page_content=page_content
            )
        )

    return result

def split_documents(documents: list[Document]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]
    )
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200
    )
    chunks = []

    for document in documents:
        chunk_index = 0
        sections = header_splitter.split_text(document.page_content)
        for section in sections:
            section_path = " > ".join(
                section.metadata[key]
                for key in ("h1", "h2", "h3", "h4", "h5", "h6")
                if key in section.metadata
            )
            if not section_path:
                section_path = document.metadata["title"]

            section_chunks = character_splitter.split_documents([section])
            for chunk in section_chunks:
                chunk.metadata.update(document.metadata)
                chunk.metadata["section"] = section_path
                chunk.metadata["chunk_index"] = chunk_index
                chunk.page_content = f"Section: {section_path}\n\n{chunk.page_content}"

                identity = (
                    f"{chunk.metadata['corpus_commit']}:"
                    f"{chunk.metadata['source']}:{chunk_index}"
                )
                chunk.id = sha256(identity.encode("utf-8")).hexdigest()
                chunks.append(chunk)
                chunk_index += 1

    return chunks

def ingest(path: Path) -> list[Document]:
    content = get_content(path)
    docs = content_to_document(content)

    return split_documents(docs)

if __name__ == "__main__":
    path = DATA_DIR / "processed" / FILE_NAME
    chunks = ingest(path)
    print(f"Data ingested successfully. {len(chunks)} chunks created")
