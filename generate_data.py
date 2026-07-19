"""Generate synthetic healthcare claims data (claims_raw.csv).

Real claims are PHI and can't be used, so this script fabricates realistic
data — including deliberately dirty records (duplicates, negative amounts,
missing providers, inconsistent status casing) for the cleaning pipeline
to fix. Seeded so every run produces the same dataset.
"""

import csv
import random

random.seed(42)

PROVIDERS = ["Apollo Hospital", "Yashoda Hospital", "Care Clinic", "Rainbow Hospital", "KIMS"]
DIAGNOSES = ["E11.9", "I10", "J45.909", "M54.5", "N39.0"]
# Mixed casing on purpose — the pipeline normalizes it.
STATUSES = ["approved", "denied", "pending", "APPROVED", "Denied "]

def pick_status(amount, provider):
    """Denial is not random in real life: high-cost claims (prior-auth risk)
    and certain providers get denied more often. Base noise keeps it realistic."""
    p_denied = 0.08
    if amount > 20000:
        p_denied += 0.35
    if provider == "Care Clinic":
        p_denied += 0.25
    r = random.random()
    if r < p_denied:
        return random.choice(["denied", "Denied "])
    if r < p_denied + 0.15:
        return "pending"
    return random.choice(["approved", "APPROVED"])



rows = []


for i in range(1, 501):
    row_provider = random.choice(PROVIDERS)
    row_amount = round(random.uniform(500, 50000), 2)
    row = {
        "claim_id": i,
        "member_id": f"M{random.randint(1000, 1099)}",
        "diagnosis_code": random.choice(DIAGNOSES),
        "claim_month": random.randint(1, 12),
        "status": pick_status(row_amount, row_provider),
        "provider_name": row_provider,
        "claim_amount": row_amount,

    }

    if random.random() < 0.05:
        row["provider_name"] = ""                          # 5%: missing provider
    if random.random() < 0.03:
        row["claim_amount"] = -abs(row["claim_amount"])    # 3%: negative amount (data-entry error)

    rows.append(row)

rows.extend(random.sample(rows, 10))                       # plant 10 duplicate rows

with open("claims_raw.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} rows to claims_raw.csv")
