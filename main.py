"""ClaimsCopilot API — FastAPI service for healthcare claims + AI policy assistant.

Endpoints:
    GET  /health                  liveness check
    POST /claims, GET /claims     submit / list claims (in-memory demo store)
    GET  /analytics/summary       claim counts & totals from SQLite (Pandas + SQL)
    GET  /analytics/top-providers providers ranked by total billed amount
    POST /ask                     AI assistant: RAG retrieval + Groq LLM, logged to DB
"""

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import joblib

# ---------------------------------------------------------------------------
# Startup: heavy resources are created ONCE here, never inside a request
# handler — loading the embedding model takes seconds, and paying that cost
# on every request would make every question slow.
# ---------------------------------------------------------------------------

# Denial-risk model trained by train_model.py.
denial_model, model_columns = joblib.load("denial_model.joblib")

# Embedding model + vector DB for RAG retrieval (populated by ingest.py).
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
policy_collection = chroma_client.get_or_create_collection("policies")

# LLM client — API key lives in .env (gitignored), never hard-coded.
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Ensure the chat log table exists before the first /ask request arrives.
init_conn = sqlite3.connect("claims.db")
init_conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        category TEXT,
        needs_human INTEGER,
        created_at TEXT
    )
""")
init_conn.commit()
init_conn.close()


app = FastAPI(title="ClaimsCopilot")

# Demo store for claims submitted via the API. In-memory on purpose:
# it resets on restart, which is why persistent data lives in SQLite.
claims_db = []


class Claim(BaseModel):
    """Schema for an incoming claim — FastAPI rejects invalid payloads with a 422."""
    member_id: str
    provider_name: str
    claim_amount: float
    diagnosis_code: str
    status: str = "submitted"


@app.get("/health")
def healthcheck():
    return {"status": "ok", "service": "claimscopilot"}


@app.get("/analytics/summary")
def analytics_summary():
    """Overall claim volume, total billed amount, and status breakdown."""
    conn = sqlite3.connect("claims.db")
    total_claims = pd.read_sql("SELECT COUNT(*) FROM claims", conn).iloc[0, 0]
    total_amount = pd.read_sql("SELECT SUM(claim_amount) FROM claims", conn).iloc[0, 0]
    by_status = pd.read_sql("SELECT status, COUNT(*) AS count FROM claims GROUP BY status", conn)
    conn.close()
    return {
        # numpy types don't serialize to JSON — cast to plain int/float first.
        "total_claims": int(total_claims),
        "total_amount": round(float(total_amount), 2),
        "by_status": by_status.set_index("status")["count"].to_dict(),
    }


@app.post("/claims")
def create_claim(claim: Claim):
    claim_data = claim.model_dump()
    claim_data["claim_id"] = len(claims_db) + 1
    claims_db.append(claim_data)
    return {"message": "claim received", "claim": claim_data}


@app.get("/claims")
def list_claims():
    return {"count": len(claims_db), "claims": claims_db}


@app.get("/analytics/top-providers")
def analytics_top_providers():
    """Providers ranked by total billed amount (SUM = cost view, not volume)."""
    conn = sqlite3.connect("claims.db")
    top_providers = pd.read_sql("""
        SELECT provider_name, SUM(claim_amount) AS total_amount
        FROM claims
        GROUP BY provider_name
        ORDER BY total_amount DESC
    """, conn)
    conn.close()
    return top_providers.to_dict(orient="records")


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    """Expected shape of the LLM's JSON reply — validation catches malformed output."""
    answer: str
    category: str
    needs_human: bool


@app.post("/ask")
def ask(question: Question):
    # --- RAG retrieval: find the policy chunks closest in meaning to the question ---
    question_spot = embed_model.encode(question.text).tolist()
    results = policy_collection.query(query_embeddings=[question_spot], n_results=3)

    # Assemble retrieved chunks into one context block, each stamped with its
    # source file so the assistant can cite where an answer came from.
    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"[source: {meta['source']}]\n{doc}\n\n"

    system_prompt = f"""You are ClaimsCopilot, an assistant for a health insurance claims team.
        Answer using ONLY the policy excerpts below. Do not use any other knowledge.
        If the answer is not in the excerpts, say you don't know and that a human agent will follow up.
        Never give medical advice. Answer in plain, simple language.
        Always reply with a JSON object with exactly these three fields:
        "answer" (string), "category" (one of: "billing", "eligibility", "claim_status", "policy", "other"),
        "needs_human" (true or false - true if the excerpts do not contain the answer, or the question is complex or sensitive).
        If the question is not about claims or insurance, politely decline, category "other", needs_human false.

        POLICY EXCERPTS:
    {context}"""


    # --- LLM call: low temperature for factual answers, JSON output enforced ---
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.text},
        ],
    )
    raw_text = response.choices[0].message.content

    # Parse then validate: json.loads catches broken JSON, the Answer model
    # catches valid JSON with wrong/missing fields. Either failure degrades
    # gracefully to a human handoff instead of a 500 error.
    try:
        data = json.loads(raw_text)
        validated = Answer(**data)
        sources_list = []
        for meta in results["metadatas"][0]:
            if meta["source"] not in sources_list:
                sources_list.append(meta["source"])
    except (json.JSONDecodeError, ValidationError):
        return {
            "question": question.text,
            "answer": "I'm sorry, I couldn't process your question. Please try again or contact support.",
            "category": "other",
            "needs_human": True,
        }

    # Log every Q&A for auditability. Parameterized placeholders (?) prevent
    # SQL injection — user text is never concatenated into the query string.
    log_conn = sqlite3.connect("claims.db")
    log_conn.execute(
        "INSERT INTO chat_history (question, answer, category, needs_human, created_at) VALUES (?, ?, ?, ?, ?)",
        (question.text, validated.answer, validated.category, int(validated.needs_human), datetime.utcnow().isoformat()),
    )
    log_conn.commit()
    log_conn.close()

    return {
        "question": question.text,
        "answer": validated.answer,
        "category": validated.category,
        "needs_human": validated.needs_human,
        "sources": sources_list,
    }

class ClaimFeatures(BaseModel):
    provider_name: str
    diagnosis_code: str
    claim_amount: float
    claim_month: int


@app.post("/predict")
def predict_denial(claim: ClaimFeatures):
    # Encode the incoming claim the same way training data was encoded,
    # then force the exact training column layout: missing dummy columns
    # get 0, unknown ones are dropped.
    input_df = pd.get_dummies(pd.DataFrame([claim.model_dump()]))
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    risk = float(denial_model.predict_proba(input_df)[0, 1])
    return {
        "denial_risk": round(risk, 3),
        "flag_for_review": risk >= 0.35,
    }