"""Evaluation-dataset indexer.

Reads ``data/evaluation/master_eval_dataset.json`` and indexes every
evidence passage into the Qdrant ``eval_documents`` collection with
metadata (source_type, is_correct, dataset_source).

Usage:
    python -m src.vectorstore.indexer
"""

from __future__ import annotations

import json
from pathlib import Path

from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

DATASET_PATH = Path("data/evaluation/master_eval_dataset.json")


def load_passages(path: Path = DATASET_PATH) -> tuple[list[str], list[dict]]:
    """Read the master eval dataset and flatten all evidence passages.

    For every record in the dataset each evidence passage with non-empty
    text becomes one indexable document.  Metadata tracks which dataset it
    came from, the passage's ``source`` label (e.g. ``"parametric_memory"``
    or a URL), and whether the passage represents the factually correct
    evidence (for ConflictQA, ``counter_memory`` is the correct one).

    Args:
        path: Path to the master JSON dataset file.

    Returns:
        Tuple of (texts, metadatas) aligned by index.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts: list[str] = []
    metadatas: list[dict] = []

    for rec in records:
        dataset_source = rec["source_dataset"]
        question = rec["question"]
        has_conflict = rec["has_known_conflict"]

        for passage in rec["evidence_passages"]:
            text = passage.get("text", "").strip()
            if not text:
                continue

            source_type = passage.get("source", "unknown")

            # For ConflictQA the counter_memory passage is the factually
            # correct one; parametric_memory is the LLM hallucination.
            if dataset_source == "conflict_qa":
                is_correct = source_type == "counter_memory"
            else:
                is_correct = True

            texts.append(text)
            metadatas.append({
                "dataset_source": dataset_source,
                "source_type": source_type,
                "is_correct": is_correct,
                "has_known_conflict": has_conflict,
                "question": question,
                "record_id": rec["id"],
            })

    return texts, metadatas


def run_indexer() -> None:
    """Load, embed, index, and print collection summary."""
    print("Loading passages from master eval dataset...")
    texts, metadatas = load_passages()
    print(f"  {len(texts)} non-empty passages found\n")

    print("Loading embedding model...")
    embedder = EmbeddingModel()
    print(f"  Model loaded — vector dim = {embedder.vector_size}\n")

    print("Embedding passages...")
    embeddings = embedder.embed_batch(texts)
    print(f"  {len(embeddings)} vectors produced\n")

    print("Indexing into Qdrant...")
    manager = QdrantManager()
    count = manager.add_documents(texts, embeddings, metadatas)
    print(f"  {count} points upserted\n")

    info = manager.collection_info()
    print("Collection info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    manager.close()


if __name__ == "__main__":
    run_indexer()
