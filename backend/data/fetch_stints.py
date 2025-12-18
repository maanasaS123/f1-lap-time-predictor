import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
BASE = "https://api.openf1.org/v1"

def fetch_stints_for_session(session_key):
    r = requests.get(
        f"{BASE}/stints",
        params={"session_key": session_key},
        timeout=30,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json())

if __name__ == "__main__":
    sessions = pd.read_csv(DATA_DIR / "selected_sessions.csv")
    session_keys = sessions["session_key"].tolist()

    all_stints = []
    for sk in session_keys:
        df = fetch_stints_for_session(sk)
        print(f"session {sk}: {len(df)} stints")
        if not df.empty:
            all_stints.append(df)

    stints_df = pd.concat(all_stints, ignore_index=True)
    out = DATA_DIR / "stints.csv"
    stints_df.to_csv(out, index=False)

    print("Saved:", out, "rows:", len(stints_df))
