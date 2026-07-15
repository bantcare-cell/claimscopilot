# ClaimsCopilot

An AI-powered assistant for healthcare claims — built with Python, FastAPI, and a RAG (Retrieval-Augmented Generation) pipeline.

> **Status: in active development** (built as a hands-on deep dive into production GenAI patterns)

## What it does

- **Claims API** — REST endpoints (JSON) to submit, validate, and query healthcare claims
- **Data pipeline** — Pandas-based ingestion: raw claims CSV → cleaning/transformation → SQLite
- **AI assistant** — answers claims & policy questions using an LLM with RAG over policy documents (LangChain + Chroma vector DB + Hugging Face embeddings), with source citations and guardrails
- **ML model** — scikit-learn classifier predicting claim denial risk

## Tech stack

Python · FastAPI · Pydantic · Pandas · SQLite · scikit-learn · LangChain · ChromaDB · sentence-transformers (Hugging Face) · Groq LLM API

## Why synthetic data?

Real healthcare claims are PHI-protected and cannot leave secure systems. This project generates realistic synthetic claims data (including deliberately dirty records — duplicates, missing values, inconsistent casing) so the cleaning pipeline solves real-world problems.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Interactive API docs: http://127.0.0.1:8000/docs
