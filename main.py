from fastapi import FastAPI
from pydantic import BaseModel

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


@app.post("/claims")
def create_claim(claim: Claim):
    claim_data = claim.model_dump()
    claim_data["claim_id"] = len(claims_db) + 1
    claims_db.append(claim_data)
    return {"message": "claim received", "claim": claim_data}


@app.get("/claims")
def list_claims():
    return {"count": len(claims_db), "claims": claims_db}