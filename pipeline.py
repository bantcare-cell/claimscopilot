import sqlite3
import pandas as pd

RAW_FILE = "claims_raw.csv"
DB_FILE = "claims.db"

# 1. INGEST
df = pd.read_csv(RAW_FILE)
print(f"loaded {len(df)} raw rows")

# 2. CLEAN
# 2a. drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"removed {before - len(df)} duplicates")

# 2b. normalize status: strip spaces, lowercase  ('APPROVED', 'Denied ' -> 'approved', 'denied')
df["status"] = df["status"].str.strip().str.lower()

# 2c. fix negative amounts (data-entry errors -> flip sign)
negatives = (df["claim_amount"] < 0).sum()
df["claim_amount"] = df["claim_amount"].abs()
print(f"fixed {negatives} negative amounts")

`# 2d. missing provider names -> explicit 'UNKNOWN' (empty strings hide problems; labels expose them)
df["provider_name"] = df["provider_name"].fillna("UNKNOWN")
missing = (df["provider_name"] == "UNKNOWN").sum()
print(f"labeled {missing} missing providers")

# 3. STORE
conn = sqlite3.connect(DB_FILE)
df.to_sql("claims", conn, if_exists="replace", index=False)
conn.close()
print(f"wrote {len(df)} clean rows to {DB_FILE} (table: claims)")