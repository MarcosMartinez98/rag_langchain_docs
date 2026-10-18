import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from math import log2
from pathlib import Path
from statistics import mean
from collections import defaultdict

from config import ROOT_DIR, DATA_DIR, EMBEDDINGS_MODEL
from vectorstore import get_vector_store

QUESTIONS_PATH = ROOT_DIR / "eval" / "questions.jsonl"
GROUND_TRUTH_PATH = ROOT_DIR / "eval" / "ground_truth.json"
CORPUS_COMMIT_PATH = ROOT_DIR / "data" / "raw" / "docs.commit"
OUTPUT_PATH = ROOT_DIR / "eval" / "runs" / "dense-baseline-v1.json"

COLLECTION_NAME = "rag_langchain_documents"
RETRIEVAL_K = 10
METRIC_KS = (1, 3, 5, 10)

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        content = file.read()
    return [json.loads(line) for line in content.splitlines()]

def load_ground_truth() -> dict[str, set[str]]:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        content = json.load(file)
    return {
        entry["question_id"]: set(entry["relevant_chunks_id"])
        for entry in content
    }

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        raise ValueError("Recall is undefined when there are no relevant chunks.")

    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    denominator = min(k, len(retrieved_ids))
    if denominator == 0:
        return 0.0

    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / denominator

def mrr_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0

def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    top_k = retrieved_ids[:k]

    dcg = sum(
        int(chunk_id in relevant_ids) / log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
    )

    ideal_relevant_count = min(k, len(top_k), len(relevant_ids))
    idcg = sum(
        1 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg if idcg else 0.0

def get_git_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return revision, bool(status)

def main() -> None:
    if not METRIC_KS or any(k <= 0 for k in METRIC_KS):
        raise ValueError("METRIC_KS must contain positive integers.")
    if RETRIEVAL_K < max(METRIC_KS):
        raise ValueError("RETRIEVAL_K must be at least max(METRIC_KS).")

    questions = load_questions()
    relevant_by_question = load_ground_truth()

    question_ids = {question["id"] for question in questions}
    if question_ids != set(relevant_by_question):
        raise ValueError(
            "Question IDs in questions.jsonl and ground_truth.json do not match."
        )

    answerable_questions = [
        question
        for question in questions
        if question["answerability"] == "answerable"
    ]

    for question in answerable_questions:
        if not relevant_by_question[question["id"]]:
            raise ValueError(
                f"Answerable question {question["id"]} has no relevant chunk IDs."
            )

    with open(CORPUS_COMMIT_PATH, "r", encoding="utf-8") as file:
        content = file.read()
    corpus_commit= content.strip()

    question_commits = {
        question["corpus_commit"]
        for question in questions
        if "corpus_commit" in question
    }
    if question_commits and question_commits != {corpus_commit}:
        raise ValueError(
            "Question benchmark corpus_commit does not match data/raw/docs.commit."
        )

    vectorstore = get_vector_store()

    # Fail early if ground-truth chunk IDs do not belong to this Chroma index.
    index_data = vectorstore.get(include=["metadatas"])
    indexed_ids = set(index_data["ids"])
    relevant_ids_in_ground_truth = set().union(*relevant_by_question.values())

    missing_ids = relevant_ids_in_ground_truth - indexed_ids
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} ground-truth chunk IDs are missing from Chroma."
        )

    retrieved_by_question: dict[str, list[dict]] = {}
    metrics_by_question: dict[str, dict[str, float]] = {}

    metric_functions = {
        "recall": recall_at_k,
        "precision": precision_at_k,
        "mrr": mrr_at_k,
        "ndcg": ndcg_at_k
    }

    metric_values_by_name = defaultdict(list) # key=metric@k value=list[values per metric of all questions]

    for question in questions:
        question_id = question["id"]

        # Chroma already returns documents in ranking order.
        results = vectorstore.similarity_search_with_score(
            question["question"],
            k=RETRIEVAL_K,
        )

        retrieved_ids = []
        retrieved_items = []

        for doc, score in results:
            if doc.id is None:
                raise ValueError(f"Retrieved chunk for {question_id} has no ID.")
            
            retrieved_ids.append(doc.id)
            retrieved_items.append(
                {
                    "chunk_id": doc.id,
                    "score": float(score)
                }
            )

        relevant_ids = relevant_by_question[question_id]
        retrieved_by_question[question_id] = retrieved_items

        if question["answerability"] != "answerable":
            continue # Ignore the non-answerable for building metrics

        question_metrics = {}
        for metric_name, metric_function in metric_functions.items():
            for k in METRIC_KS:
                question_metrics[f"{metric_name}@{k}"] = round(metric_function(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ), 4)

        metrics_by_question[question_id] = question_metrics

        # Obtain mean metrics
        # First create a dictionary with each metric and a list for all values with each question
        for metric, value in metrics_by_question[question_id].items():
            metric_values_by_name[metric].append(value)

    mean_metrics = {
        metric: round(mean(list_values), 4)
        for metric, list_values in metric_values_by_name.items()
    }

    git_revision, git_worktree_dirty = get_git_state()

    report = {
        "run_name": "dense-baseline-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "retrieval_method": "dense similarity search",
            "embedding_model": EMBEDDINGS_MODEL,
            "collection_name": COLLECTION_NAME,
            "distance_metric": "l2",
            "retrieval_k": RETRIEVAL_K,
            "metric_ks": list(METRIC_KS),
            "chunking": {
                "strategy": (
                    "MarkdownHeaderTextSplitter followed by "
                    "RecursiveCharacterTextSplitter"
                ),
                "chunk_size_characters": 800,
                "chunk_overlap_characters": 200,
            },
        },
        "benchmark": {
            "questions_sha256": sha256_file(QUESTIONS_PATH),
            "ground_truth_sha256": sha256_file(GROUND_TRUTH_PATH),
            "corpus_commit": corpus_commit,
            "total_questions": len(questions),
            "answerable_questions": len(answerable_questions),
        },
        "implementation": {
            "git_revision": git_revision,
            "git_worktree_dirty": git_worktree_dirty,
            "python_version": platform.python_version(),
            "evaluation_script_sha256": sha256_file(Path(__file__)),
        },
        "retrieval_by_question": retrieved_by_question,
        "metrics_by_question": metrics_by_question,
        "mean_metrics_answerable_questions": mean_metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # "x" prevents silently replacing a previously recorded baseline.
    with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print(f"Saved baseline report to: {OUTPUT_PATH}")
    print("Mean metrics over answerable questions:")
    for metric_name, value in mean_metrics.items():
        print(f"  {metric_name}: {value:.3f}")


if __name__ == "__main__":
    main()

            






        