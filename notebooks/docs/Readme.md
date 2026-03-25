# Azure ETL & Retrieval Pipeline

## Overview

This project implements a retrieval-augmented search pipeline for a knowledge base containing product manuals, troubleshooting guides, and policy documents.

The system processes mixed document formats (PDF, Markdown, TXT), transforms them into structured chunks, generates embeddings, indexes them, and enables hybrid search combining keyword and vector similarity.

---

## Architecture

The pipeline consists of the following stages:

1. **Extraction**
   - Reads documents from a storage source (simulated locally)
   - Supports PDF, Markdown, and TXT formats
   - Extracts text and metadata (source, category, file type)

2. **Chunking**
   - Splits documents into smaller segments
   - Produces retrieval-friendly chunks

3. **Embedding**
   - Uses `sentence-transformers/all-MiniLM-L6-v2`
   - Converts chunks into dense vector representations

4. **Indexing**
   - Uses FAISS for vector similarity search
   - Stores embeddings efficiently for fast retrieval

5. **Hybrid Search**
   - Combines:
     - Vector similarity (semantic search)
     - Keyword matching
   - Returns ranked results with metadata and captions

---

## Project Structure

```

/src
extract.py
chunk.py
embed.py
index.py
ingest.py
search.py

/notebooks
pipeline_walkthrough.ipynb

/docs
README.md

````

Additional helper files:
- `run_local.py` → used for local validation of the full pipeline
- `create_samples.py` → generates sample dataset
- `faiss_index.bin` → saved vector index
- `chunks_store.pkl` → saved chunk metadata

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

---

### 2. (Optional) Azure Configuration

If using Azure services, create a `.env` file:

```env
AZURE_BLOB_CONNECTION_STRING=your_connection_string
AZURE_SEARCH_ENDPOINT=your_endpoint
AZURE_SEARCH_KEY=your_key
```

> Note: `.env` is excluded from version control for security reasons.

---

## Running the Pipeline

### Local Validation

The pipeline was validated locally using:

```bash
python run_local.py
```

This performs:

* extraction
* chunking
* embedding
* indexing
* hybrid search queries

---

### Install Jupyter notebook

```bash
pip install jupyter notebook
```

## Notebook Demonstration

The notebook:

```
notebooks/pipeline_walkthrough.ipynb
```

demonstrates:

* pipeline structure
* data flow between stages
* sample outputs from a successful local run

---

## Example Queries

The system was tested with queries such as:

* "How do I reset the device to factory settings?"
* "error code 101 connection timeout fix"
* "password and encryption security policy"

Results include:

* ranked relevance scores
* vector similarity scores
* keyword match counts
* metadata (category, file type)
* caption-style previews

---

## Assumptions

* Documents are small-scale (~10 files)
* Local execution simulates Azure pipeline behavior
* FAISS is used for efficient vector indexing
* SentenceTransformers model is used for embeddings

---

## Known Limitations

* Azure services are not actively invoked in local validation
* Jupyter notebook execution may be environment-dependent
* Hybrid scoring is simplified (vector + keyword weighting)
* PDF extraction is basic and may not handle complex layouts
- Azure services are not actively invoked in local validation (pipeline design is Azure-ready but executed locally for testing)

---

## Results Preview

Example output from hybrid search:

- Query: "How do I reset the device to factory settings?"
  - Top result: Device B Technical Manual
  - High keyword match + strong vector similarity

- Query: "password and encryption security policy"
  - Top result: security.txt
  - Relevant policy section retrieved with correct category

This demonstrates the effectiveness of combining semantic and keyword-based retrieval.

## Conclusion

This project demonstrates a complete end-to-end retrieval pipeline, including document ingestion, transformation, indexing, and hybrid search.

The design mirrors real-world Azure-based architectures while remaining fully testable in a local environment.

