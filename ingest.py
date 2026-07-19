"""Ingest pipeline: policy documents -> chunks -> embeddings -> Chroma vector DB.

Run at setup and again whenever a policy file changes:
    python ingest.py

Safe to re-run: upsert replaces chunks with the same id instead of duplicating.
"""

import os

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# chunk_size=500 chars ~ one paragraph. Too big blurs multiple topics into one
# embedding; too small strips context. overlap=100 repeats the tail of each
# chunk at the start of the next, so a rule cut mid-sentence survives whole
# on at least one chunk.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Two lists kept in lock-step: all_chunks[i] is the text, sources[i] is the
# file it came from — that pairing is what makes source citations possible.
all_chunks = []
sources = []

# Any .txt dropped into policies/ is picked up automatically — no code change.
for filename in os.listdir("policies"):
    with open(os.path.join("policies", filename), encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.split_text(text)
    all_chunks.extend(chunks)
    sources.extend([filename] * len(chunks))

print(f"Total chunks created: {len(all_chunks)}")

# Embed every chunk (384-dim vectors, local Hugging Face model — no API cost)
# and store text + vector + source metadata together in Chroma.
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("policies")

embeddings = model.encode(all_chunks).tolist()

collection.upsert(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=[{"source": s} for s in sources],
    ids=[f"chunk-{i}" for i in range(len(all_chunks))],
)

print("stored in chroma:", collection.count())
