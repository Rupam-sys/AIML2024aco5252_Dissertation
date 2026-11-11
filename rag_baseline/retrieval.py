"""Dense and hybrid retrieval helpers for the local Chroma collection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_rank: int | None = None
    lexical_rank: int | None = None
    cosine_distance: float | None = None
    fusion_score: float | None = None


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def retrieve_dense(collection, query: str, top_k: int = 5) -> list[SearchHit]:
    """Retrieve the nearest chunks using Chroma's collection embedding function."""
    count = collection.count()
    if count == 0 or top_k <= 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        SearchHit(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
            dense_rank=rank,
            cosine_distance=float(distance),
        )
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        )
    ]


def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Calculate standard BM25 scores over the current document collection."""
    tokenized_documents = [_tokens(document or "") for document in documents]
    query_terms = set(_tokens(query))
    if not tokenized_documents or not query_terms:
        return [0.0] * len(tokenized_documents)

    document_frequencies: Counter[str] = Counter()
    term_frequencies = []
    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequencies.update(frequencies.keys())

    corpus_size = len(tokenized_documents)
    lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = sum(lengths) / corpus_size if corpus_size else 0.0
    scores = [0.0] * corpus_size

    for term in query_terms:
        document_frequency = document_frequencies.get(term, 0)
        if document_frequency == 0:
            continue
        inverse_document_frequency = math.log1p(
            (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for index, frequencies in enumerate(term_frequencies):
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            length_normalizer = (
                1.0 - b + b * lengths[index] / average_length
                if average_length
                else 1.0
            )
            scores[index] += inverse_document_frequency * (
                term_frequency * (k1 + 1.0)
                / (term_frequency + k1 * length_normalizer)
            )
    return scores


def retrieve_hybrid(
    collection,
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Fuse dense Chroma and lexical BM25 rankings with reciprocal-rank fusion."""
    count = collection.count()
    if count == 0 or top_k <= 0:
        return []
    candidate_k = min(max(candidate_k, top_k), count)

    dense_hits = retrieve_dense(collection, query, top_k=candidate_k)
    corpus = collection.get(include=["documents", "metadatas"])
    corpus_ids = corpus["ids"]
    corpus_documents = corpus["documents"] or []
    corpus_metadatas = corpus["metadatas"] or []
    scores = _bm25_scores(query, corpus_documents)

    lexical_order = sorted(
        (index for index, score in enumerate(scores) if score > 0.0),
        key=lambda index: (-scores[index], corpus_ids[index]),
    )[:candidate_k]
    lexical_hits = [
        SearchHit(
            chunk_id=corpus_ids[index],
            text=corpus_documents[index],
            metadata=corpus_metadatas[index],
            lexical_rank=rank,
        )
        for rank, index in enumerate(lexical_order, start=1)
    ]

    fused_scores: dict[str, float] = {}
    dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
    lexical_by_id = {hit.chunk_id: hit for hit in lexical_hits}
    for rank, hit in enumerate(dense_hits, start=1):
        fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, hit in enumerate(lexical_hits, start=1):
        fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    combined_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )[:top_k]
    hits = []
    for chunk_id in combined_ids:
        dense_hit = dense_by_id.get(chunk_id)
        lexical_hit = lexical_by_id.get(chunk_id)
        source = dense_hit or lexical_hit
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                text=source.text,
                metadata=source.metadata,
                dense_rank=dense_hit.dense_rank if dense_hit else None,
                lexical_rank=lexical_hit.lexical_rank if lexical_hit else None,
                cosine_distance=dense_hit.cosine_distance if dense_hit else None,
                fusion_score=fused_scores[chunk_id],
            )
        )
    return hits
