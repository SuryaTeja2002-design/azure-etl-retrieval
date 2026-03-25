# run_local.py
# Full pipeline: extract → chunk → embed → FAISS index → search
# No Azure credentials needed!

import os
import sys
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pdfplumber
import re

# ── CONFIG ────────────────────────────────────────────────────────────────────
DOCS_FOLDER = "docs"
INDEX_FILE  = "faiss_index.bin"
CHUNKS_FILE = "chunks_store.pkl"
MODEL_NAME  = "all-MiniLM-L6-v2"
TARGET_TOKENS = 150   # smaller = more chunks for demo purposes

# ── STAGE 1: EXTRACT ──────────────────────────────────────────────────────────
def extract_documents(folder):
    documents = []
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() == ".pdf":
            with pdfplumber.open(path) as pdf:
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            file_type = "pdf"
        elif path.suffix.lower() == ".md":
            raw = path.read_text(encoding="utf-8", errors="replace")
            # Strip YAML front-matter
            raw = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL)
            text = raw.strip()
            file_type = "md"
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            file_type = "txt"
        else:
            continue

        category = path.parts[1] if len(path.parts) > 2 else "general"
        documents.append({
            "doc_id":   path.stem,
            "blob_name": str(path),
            "file_type": file_type,
            "category":  category,
            "text":      text.strip(),
        })
        print(f"  [EXTRACT] {path}  ({len(text)} chars)")
    return documents

# ── STAGE 2: CHUNK ────────────────────────────────────────────────────────────
def chunk_documents(documents):
    all_chunks = []
    for doc in documents:
        sentences = re.split(r'(?<=[.!?])\s+', doc["text"])
        buffer, buf_words, idx = [], 0, 0
        for sent in sentences:
            w = len(sent.split())
            if buf_words + w > TARGET_TOKENS and buffer:
                chunk_text = " ".join(buffer)
                all_chunks.append({
                    "chunk_id":  f"{doc['doc_id']}_{idx:04d}",
                    "doc_id":    doc["doc_id"],
                    "text":      chunk_text,
                    "category":  doc["category"],
                    "file_type": doc["file_type"],
                    "blob_name": doc["blob_name"],
                })
                idx += 1
                buffer, buf_words = [], 0
            buffer.append(sent)
            buf_words += w
        if buffer:
            all_chunks.append({
                "chunk_id":  f"{doc['doc_id']}_{idx:04d}",
                "doc_id":    doc["doc_id"],
                "text":      " ".join(buffer),
                "category":  doc["category"],
                "file_type": doc["file_type"],
                "blob_name": doc["blob_name"],
            })
    print(f"  [CHUNK] {len(all_chunks)} total chunks from {len(documents)} documents")
    return all_chunks

# ── STAGE 3: EMBED ────────────────────────────────────────────────────────────
def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]
    print(f"  [EMBED] Embedding {len(texts)} chunks with '{MODEL_NAME}'...")
    vectors = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=True, batch_size=32)
    return vectors.astype("float32")

# ── STAGE 4: INDEX ────────────────────────────────────────────────────────────
def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine on normalised vecs
    index.add(vectors)
    print(f"  [INDEX] FAISS index built: {index.ntotal} vectors (dim={dim})")
    return index

# ── STAGE 5: SEARCH ───────────────────────────────────────────────────────────
def search(query, index, chunks, model, top_n=5, filter_category=None):
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # Vector search
    scores, indices = index.search(q_vec, top_n * 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if filter_category and chunk["category"] != filter_category:
            continue

        # Keyword boost (simple BM25-style: count query term matches)
        query_words = set(query.lower().split())
        chunk_words = chunk["text"].lower().split()
        keyword_hits = sum(1 for w in chunk_words if w in query_words)
        hybrid_score = float(score) + (keyword_hits * 0.05)  # RRF-style boost

        # Semantic caption: first 2 sentences of the chunk
        sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
        caption = " ".join(sentences[:2])

        results.append({
            "rank":       0,
            "chunk_id":   chunk["chunk_id"],
            "score":      round(hybrid_score, 4),
            "vector_score": round(float(score), 4),
            "keyword_hits": keyword_hits,
            "content":    chunk["text"],
            "caption":    caption,
            "doc_id":     chunk["doc_id"],
            "category":   chunk["category"],
            "file_type":  chunk["file_type"],
            "blob_name":  chunk["blob_name"],
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG PIPELINE — LOCAL MODE (no Azure credentials)")
    print("=" * 60)

    # Stage 1
    print("\n[STAGE 1] Extracting documents...")
    docs = extract_documents(DOCS_FOLDER)
    print(f"  Extracted {len(docs)} documents")

    # Stage 2
    print("\n[STAGE 2] Chunking...")
    chunks = chunk_documents(docs)

    # Stage 3
    print("\n[STAGE 3] Loading embedding model (downloads once ~90MB)...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = embed_chunks(chunks, model)

    # Stage 4
    print("\n[STAGE 4] Building FAISS index...")
    index = build_index(vectors)

    # Save index + chunks to disk
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  Saved index to '{INDEX_FILE}' and chunks to '{CHUNKS_FILE}'")

    # Stage 5 — run 3 demo queries
    print("\n[STAGE 5] Running demo searches...")
    demo_queries = [
        ("How do I reset the device to factory settings?", None),
        ("error code 101 connection timeout fix",          "troubleshooting"),
        ("password and encryption security policy",        None),
    ]

    for query, cat in demo_queries:
        filter_label = f" [filter: category='{cat}']" if cat else ""
        print(f"\n{'─'*60}")
        print(f"  Query: \"{query}\"{filter_label}")
        print(f"{'─'*60}")
        results = search(query, index, chunks, model, top_n=3, filter_category=cat)
        for r in results:
            print(f"  [{r['rank']}] {r['doc_id']}  score={r['score']}  "
                  f"(vector={r['vector_score']}, keyword_hits={r['keyword_hits']})")
            print(f"      Category : {r['category']}")
            print(f"      Caption  : {r['caption'][:120]}...")
            print()

    print("=" * 60)
    print("  Pipeline complete! All 5 stages ran successfully.")
    print("=" * 60)