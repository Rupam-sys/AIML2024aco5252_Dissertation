"""Search the local RAG collection with dense or hybrid retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from .ingest import COLLECTION_NAME
from .retrieval import SearchHit, retrieve_dense, retrieve_hybrid


def _format_location(hit: SearchHit) -> str:
    location = str(hit.metadata.get("source", "unknown source"))
    if "page" in hit.metadata:
        location += f" (PDF page {hit.metadata['page']})"
    return location


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or search text")
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="dense")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    if args.mode == "hybrid":
        hits = retrieve_hybrid(
            collection,
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
    else:
        hits = retrieve_dense(collection, args.query, top_k=args.top_k)

    if not hits:
        print("No matching chunks found.")
        return

    for rank, hit in enumerate(hits, start=1):
        print(f"\n[{rank}] { _format_location(hit) }")
        if args.mode == "dense":
            print(f"Cosine distance: {hit.cosine_distance:.4f}")
        else:
            print(
                "Ranks: "
                f"dense={hit.dense_rank if hit.dense_rank is not None else '-'}, "
                f"BM25={hit.lexical_rank if hit.lexical_rank is not None else '-'}"
            )
            print(f"RRF score: {hit.fusion_score:.6f}")
        print(hit.text)


if __name__ == "__main__":
    main()
