import requests
import pandas as pd

BASE = "https://api.openf1.org/v1"

sessions = pd.read_csv("backend/data/sessions.csv")

valid = []
invalid = []

for _, row in sessions.iterrows():
    sk = row["session_key"]
    name = row["session_name"]

    r = requests.get(
        f"{BASE}/laps",
        params={"session_key": sk},
        timeout=10
    )

    if r.ok and len(r.json()) > 0:
        print(f"✓ {sk} ({name}): {len(r.json())} laps")
        valid.append({"session_key": sk, "session_name": name})
    else:
        print(f"✗ {sk} ({name}): no data")
        invalid.append(sk)

pd.DataFrame(valid).to_csv(
    "backend/data/valid_sessions.csv", index=False
)

print("\nSUMMARY")
print("Valid sessions:", len(valid))
print("Invalid sessions:", len(invalid))
