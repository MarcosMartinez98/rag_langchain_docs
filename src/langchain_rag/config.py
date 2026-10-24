from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FILE_NAME = "docs.jsonl"

EMBEDDINGS_MODEL = "nomic-embed-text"
COLLECTION_NAME = "rag_langchain_documents"
COLLECTION_NAME_WITH_PREFIX = "rag_langchain_documents_nomic_prefix_test"
