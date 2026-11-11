"""Load, chunk, embed, and persist documents in a local Chroma collection.

Supported inputs are UTF-8 plain text, Markdown, and text-based PDFs. Chroma's
default local embedding function creates the document vectors when chunks are
upserted into the collection.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb


LOGGER = logging.getLogger("rag_baseline.ingest")
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2 (Chroma default)"


@dataclass(frozen=True)
class SourcePage:
    source: str
    page: int | None
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int | None
    chunk_index: int
    text: str


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    paragraphs = [" ".join(line.split()) for line in text.replace("\r\n", "\n").split("\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    return "\n\n".join(paragraphs)


def load_file(path: Path, source_root: Path) -> list[SourcePage]:
    """Read one supported file and retain PDF page numbers where available."""
    source = path.relative_to(source_root).as_posix()
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(SourcePage(source, page_number, text))
        return pages

    text = normalize_text(path.read_text(encoding="utf-8-sig"))
    return [SourcePage(source, None, text)] if text else []


def split_into_chunks(
    pages: Iterable[SourcePage],
    chunk_words: int = 160,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Split pages into deterministic word windows with a small overlap."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1")

    chunks: list[Chunk] = []
    step = chunk_words - overlap_words
    for page in pages:
        words = page.text.split()
        start = 0
        chunk_index = 0
        while start < len(words):
            text = " ".join(words[start : start + chunk_words]).strip()
            if text:
                chunks.append(Chunk(page.source, page.page, chunk_index, text))
                chunk_index += 1
            start += step
    return chunks


def stable_chunk_id(chunk: Chunk) -> str:
    identity = f"{chunk.source}\0{chunk.page}\0{chunk.chunk_index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ingestion_fingerprint(path: Path, chunk_words: int, overlap_words: int) -> str:
    """Fingerprint file contents and settings that affect its stored vectors."""
    digest = hashlib.sha256(path.read_bytes())
    digest.update(f"\0{chunk_words}\0{overlap_words}\0{EMBEDDING_MODEL_ID}".encode("utf-8"))
    return digest.hexdigest()


def ingest_directory(
    source_dir: Path,
    persist_dir: Path,
    chunk_words: int = 160,
    overlap_words: int = 30,
) -> int:
    """Ingest supported files into Chroma and return the number of chunks."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")

    files = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        LOGGER.warning("No .txt, .md, or .pdf files found under %s", source_dir)
        return 0

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    max_batch_size = client.get_max_batch_size()
    total_chunks = 0
    skipped_files = 0

    for path in files:
        try:
            source = path.relative_to(source_dir).as_posix()
            fingerprint = ingestion_fingerprint(path, chunk_words, overlap_words)
            saved = collection.get(
                where={"source": source},
                include=["metadatas"],
                limit=1,
            )
            saved_metadatas = saved["metadatas"] or []
            if saved_metadatas and saved_metadatas[0].get("ingestion_fingerprint") == fingerprint:
                LOGGER.info("Skipping unchanged file %s", source)
                skipped_files += 1
                continue

            pages = load_file(path, source_dir)
        except Exception as error:
            LOGGER.warning("Skipping %s: %s", path, error)
            continue

        chunks = split_into_chunks(pages, chunk_words, overlap_words)
        if not chunks:
            LOGGER.warning("No extractable text in %s; leaving stored chunks unchanged", path)
            continue

        ids = [stable_chunk_id(chunk) for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "content_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                "ingestion_fingerprint": fingerprint,
                "embedding_model": EMBEDDING_MODEL_ID,
            }
            if chunk.page is not None:
                metadata["page"] = chunk.page
            metadatas.append(metadata)
        existing_ids = collection.get(where={"source": chunks[0].source}, include=[])["ids"]

        # Chroma embeds each document through the collection's embedding
        # function. Upsert first; remove obsolete chunks only after it succeeds.
        for start in range(0, len(chunks), max_batch_size):
            end = start + max_batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=[chunk.text for chunk in chunks[start:end]],
                metadatas=metadatas[start:end],
            )

        current_ids = set(ids)
        stale_ids = [record_id for record_id in existing_ids if record_id not in current_ids]
        for start in range(0, len(stale_ids), max_batch_size):
            collection.delete(ids=stale_ids[start : start + max_batch_size])

        total_chunks += len(chunks)
        LOGGER.info("Ingested %s: %d chunks", chunks[0].source, len(chunks))

    LOGGER.info(
        "Done. Added/updated %d chunks; skipped %d unchanged files in %s",
        total_chunks,
        skipped_files,
        persist_dir,
    )
    return total_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/documents"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    ingest_directory(
        args.source_dir,
        args.persist_dir,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )


if __name__ == "__main__":
    main()
