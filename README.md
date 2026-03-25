# Azure ETL & Retrieval Pipeline

## Overview

This project implements a retrieval-augmented search pipeline for a knowledge base containing product manuals, troubleshooting guides, and policy documents.

The system processes mixed document formats (PDF, Markdown, TXT), transforms them into structured chunks, generates embeddings, indexes them, and enables hybrid search combining keyword and vector similarity.

---

## Architecture

The pipeline consists of the following stages:

1. **Extraction**
   - Reads documents from a storage source (simulated locally via `docs/` folder)
   - Supports PDF, Markdown, and TXT formats
   - Extracts text and metadata (source, category, file type)

2. **Chunking**
   - Splits documents into smaller segments using a sentence-aware sliding window
   - Produces retrieval-friendly chunks (target: 150 tokens each)

3. **Embedding**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` locally
   - Converts chunks into 384-dimensional L2-normalised dense vectors
   - Maps to Azure OpenAI `text-embedding-ada-002` in production

4. **Indexing**
   - Uses FAISS `IndexFlatIP` for exact inner-product (cosine) search locally
   - Maps to Azure AI Search HNSW vector index in production
   - Saves `faiss_index.bin` and `chunks_store.pkl` to disk

5. **Hybrid Search**
   - Combines vector similarity (semantic search) and keyword match boosting
   - Returns ranked results with scores, metadata, and extractive captions
   - Maps to Azure AI Search BM25 + RRF fusion + Semantic Ranker in production

---

## Project Structure

```
BMO-AI Eng/
├── src/
│   ├── extract.py          # Document extraction (Azure Blob + pdfplumber)
│   ├── chunk.py            # Sentence-aware sliding window chunker
│   ├── embed.py            # Azure OpenAI embedding client
│   ├── index.py            # Azure AI Search index schema + uploader
│   ├── ingest.py           # End-to-end ingestion pipeline orchestrator
│   └── search.py           # Hybrid / semantic search client
├── notebooks/
│   └── pipeline_walkthrough.ipynb   # Full walkthrough with outputs
├── docs/
│   ├── manuals/            # PDF product manuals
│   ├── troubleshooting/    # Markdown troubleshooting guides
│   └── policies/           # TXT policy documents
├── run_local.py            # Local validation pipeline (no Azure needed)
├── create_samples.py       # Generates sample docs for testing
├── faiss_index.bin         # Saved FAISS vector index (from local run)
├── chunks_store.pkl        # Saved chunk metadata (from local run)
├── requirements.txt
└── .gitignore
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

```
sentence-transformers
faiss-cpu
pdfplumber
fpdf2
numpy
```

### 2. Generate sample documents

```bash
python create_samples.py
```

This creates the `docs/` folder with 4 sample documents (2 PDFs, 1 Markdown, 1 TXT).

### 3. (Optional) Azure Configuration

If using Azure services, create a `.env` file in the project root:

```env
AZURE_BLOB_CONNECTION_STRING=your_connection_string
AZURE_SEARCH_ENDPOINT=your_endpoint
AZURE_SEARCH_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_KEY=your_openai_key
```

> Note: `.env` is excluded from version control via `.gitignore`.

---

## Running the Pipeline

### Local Validation

Run the full local pipeline from the **project root**:

```bash
python run_local.py
```

This performs all 5 stages: extraction → chunking → embedding → indexing → hybrid search, and saves `faiss_index.bin` and `chunks_store.pkl` to disk.

---

## Notebook Demonstration

The notebook provides a step-by-step walkthrough of the pipeline with live code and outputs:

```
notebooks/pipeline_walkthrough.ipynb
```

### Important: Running the Notebook

Because the notebook lives inside `notebooks/`, you must add the following as the **first cell** before running anything else. This ensures Python can find `run_local.py` in the project root and that the `docs/` folder is accessible:

```python
import sys, os
sys.path.insert(0, os.path.abspath("../src"))
os.chdir(os.path.abspath(".."))
```

Then run all remaining cells top-to-bottom.

### Install Jupyter if needed

```bash
pip install jupyter notebook
```

### What the notebook covers

- **Stage 1** — Extraction: loads all 4 documents with metadata preview
- **Stage 2** — Chunking: shows chunk counts per document and sample chunk text
- **Stage 3** — Embedding: prints matrix shape `(6, 384)` and L2 norms
- **Stage 4** — Indexing: builds FAISS index and saves artifacts to disk
- **Stage 5** — Hybrid Search: runs 3 demo queries with ranked results, scores, and captions
- **Summary table** mapping each local component to its Azure production equivalent

---

## Example Queries

The system was tested with the following queries:

| Query | Top Result | Notes |
|-------|-----------|-------|
| "How do I reset the device to factory settings?" | `deviceA` (manuals) | High keyword + vector score |
| "error code 101 connection timeout fix" | `error101` (troubleshooting) | Category filter applied |
| "password and encryption security policy" | `security.txt` (policies) | Large score gap confirms precision |

Results include ranked relevance scores, vector similarity scores, keyword match counts, metadata (category, file type), and caption-style previews.

---

## Local vs Azure Architecture Mapping

| Local (validation) | Azure (production) |
|--------------------|--------------------|
| `sentence-transformers all-MiniLM-L6-v2` | Azure OpenAI `text-embedding-ada-002` |
| FAISS `IndexFlatIP` | Azure AI Search HNSW vector index |
| BM25-style keyword boost | Azure AI Search BM25 + RRF fusion |
| First-2-sentences caption | Azure Semantic Ranker extractive captions |
| Local `docs/` folder | Azure Blob Storage container |

---

## Assumptions

- Documents are small-scale (~10 files)
- Local execution simulates Azure pipeline behaviour
- FAISS is used for efficient vector indexing in local mode
- SentenceTransformers model is used for local embeddings

---

## Known Limitations

- Azure services are not actively invoked in local validation (pipeline is Azure-ready but executed locally for testing)
- The notebook must be run from the correct working directory — see setup note above
- Hybrid scoring is simplified (vector score + keyword weight); production uses full RRF fusion
- PDF extraction is basic and may not handle complex layouts or scanned documents without the Azure Document Intelligence OCR fallback

---

## Conclusion

This project demonstrates a complete end-to-end retrieval pipeline, including document ingestion, transformation, indexing, and hybrid search. The design mirrors real-world Azure-based RAG architectures while remaining fully testable in a local environment without any cloud credentials.
