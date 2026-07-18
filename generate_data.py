import csv
import random

random.seed(42)

PROVIDERS = ["Apollo Hospital", "Yashoda Hospital", "Care Clinic", "Rainbow Hospital", "KIMS"]
DIAGNOSES = ["E11.9", "I10", "J45.909", "M54.5", "N39.0"]
STATUSES = ["approved", "denied", "pending", "APPROVED", "Denied "]

rows = []
for i in range(1, 501):
    row = {
        "claim_id": i,
        "member_id": f"M{random.randint(1000, 1099)}",
        "provider_name": random.choice(PROVIDERS),
        "diagnosis_code": random.choice(DIAGNOSES),
        "claim_amount": round(random.uniform(500, 50000), 2),
        "claim_month": random.randint(1, 12),
        "status": random.choice(STATUSES),
    }

    if random.random() < 0.05:
        row["provider_name"] = ""          # 5%: missing provider
    if random.random() < 0.03:
        row["claim_amount"] = -abs(row["claim_amount"])   # 3%: negative amount (data-entry error)

    rows.append(row)

rows.extend(random.sample(rows, 10))       # plant 10 duplicate rows

with open("claims_raw.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} rows to claims_raw.csv")