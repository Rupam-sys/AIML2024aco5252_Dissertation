# Dissertation RAG baseline

This project starts with a small, local ingestion pipeline using Chroma as the
vector database. It reads text, Markdown, and text-based PDF files, splits them
into overlapping word chunks, and stores their text and metadata in a persistent
Chroma collection. Chroma's default embedding function embeds the text locally
when it is added, and Chroma embeds search queries when they are queried.

## 1. Create an environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first run downloads Chroma's default embedding model (`all-MiniLM-L6-v2`)
and its required runtime. Embeddings are generated locally after download.

## 2. Add source documents

Put `.txt`, `.md`, or text-based `.pdf` files in `data/documents/`. PDFs made
from scanned images need OCR, which this first version does not perform.

## 3. Run ingestion

```powershell
python -m rag_baseline.ingest --verbose
```

By default, the pipeline reads `data/documents/` and writes the Chroma database
to `data/chroma/`. You can override paths and chunk settings:

```powershell
python -m rag_baseline.ingest `
  --source-dir data/documents `
  --persist-dir data/chroma `
  --chunk-words 160 `
  --overlap-words 30
```

## Search the collection

```powershell
python -m rag_baseline.search "What evidence supports the answer?" --top-k 5
```

Chroma embeds the query text and returns the nearest stored chunks with their
source/page metadata and cosine distances. The ingester records a fingerprint
for each source file, including its content and chunking/model settings. On
later runs, unchanged files are skipped. New files are added, and changed files
are refreshed with obsolete chunks removed. Keep the same embedding function
and distance setting for a collection; create a new collection if those
settings change.

### Hybrid retrieval

```powershell
python -m rag_baseline.search "What evidence supports the answer?" `
  --mode hybrid `
  --top-k 5 `
  --candidate-k 20
```

Hybrid mode runs Chroma dense search and a transparent local BM25 pass over the
stored chunks, then fuses both rankings with reciprocal-rank fusion (RRF). The
CLI prints each result's dense rank, BM25 rank, and fusion score so we can
inspect how the two retrieval signals contributed. This first version scans
the local collection to calculate BM25 for each query, which is suitable for
the small dissertation prototype; for a larger corpus, the lexical index
should be built once and reused.

### Generate a grounded answer with local Llama

Install [Ollama](https://ollama.com/download/mac) on macOS, then download the
small Llama 3.2 3B model:

```bash
ollama pull llama3.2:3b
```

Keep Ollama running, then ask a question against the ingested Chroma collection:

```powershell
python -m rag_baseline.ask "What does the document say about eligibility?" --mode hybrid
```

This step retrieves evidence with hybrid search and sends the question plus
those chunks to Ollama's local API. The answer is instructed to use only the
retrieved evidence and cite it as `[S1]`, `[S2]`, and so on. If there are no
retrieved chunks, generation is skipped. No hosted model API is used, so there
is no per-request inference charge; generation runs on the computer where
Ollama is installed.

The default model and API address are `llama3.2:3b` and
`http://localhost:11434`. They can be changed with `--model` and
`--ollama-url`, or through the `OLLAMA_MODEL` and `OLLAMA_BASE_URL` environment
variables. This is the first generation baseline; adversarial critique,
verification, and bandit action selection will be added as separate steps.

## What we have learned so far

- **Loading** extracts text and keeps source/page metadata for citations.
- **Normalization** cleans inconsistent whitespace without merging paragraphs.
- **Chunking** creates bounded passages and overlap preserves context across
  chunk boundaries.
- **Embedding and search** are handled by Chroma's collection embedding
  function and vector index.
- **Hybrid search** combines dense semantic ranking with BM25 lexical ranking
  using RRF.
- **Persistence** makes ingestion reusable without reprocessing the files on
  every question.
- **Grounded generation** sends retrieved chunks to a local chat model and asks
  it to cite those chunks in its answer.

The chunk size, overlap, embedding model, and document collection are initial
baseline settings. We will record and evaluate changes to them rather than
treating them as final dissertation decisions.
