from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
from uvicorn import run
import json
from datetime import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

claims_db = []


class Claim(BaseModel):
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
     conn = sqlite3.connect("claims.db")
     total_claims = pd.read_sql("SELECT COUNT(*) FROM claims", conn).iloc[0, 0]
     total_amount = pd.read_sql("SELECT SUM(claim_amount) FROM claims", conn).iloc[0, 0]
     by_status = pd.read_sql("SELECT status, COUNT(*) AS count FROM claims GROUP BY status", conn)
     conn.close()
     return {
    "total_claims": int(total_claims),
    "total_amount": round(float(total_amount), 2),
    "by_status": by_status.set_index("status")["count"].to_dict()
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
    answer: str
    category: str
    needs_human: bool

@app.post("/ask")
def ask(question: Question):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are ClaimsCopilot, an assistant for a health insurance claims team. Answer only questions about health insurance claims and policies, in plain, simple language. Never give medical advice. Always reply with a JSON object containing exactly these three fields: \"answer\" (a clear plain-language answer, as a string), \"category\" (one of: \"billing\", \"eligibility\", \"claim_status\", \"policy\", \"other\"), and \"needs_human\" (true or false - true if the question is complex or sensitive and a human agent should follow up). If the question is not about claims or insurance, set answer to a polite decline, category to \"other\", and needs_human to false."},
            {"role": "user", "content": question.text},
        ],
    )   
    raw_text = response.choices[0].message.content
    try:
        data = json.loads(raw_text)
        validated = Answer(**data)

    except (json.JSONDecodeError, ValidationError):
        return {
            "question": question.text,
            "answer": "I'm sorry, I couldn't process your question. Please try again or contact",
            "category": "other",
            "needs_human": True,
        }
    
    log_conn = sqlite3.connect("claims.db")
    log_conn.execute(
        "INSERT INTO chat_history (question, answer, category, needs_human, created_at) VALUES (?, ?, ?, ?, ?)",
        (question.text, validated.answer, validated.category, int(validated.needs_human), datetime.utcnow().isoformat()),
    )
    log_conn.commit()
    log_conn.close()


    return{
        "question": question.text,
        "answer": validated.answer,
        "category": validated.category,
        "needs_human": validated.needs_human,
    }
