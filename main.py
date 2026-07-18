from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import pandas as pd
from uvicorn import run
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
