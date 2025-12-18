import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
BASE = "https://api.openf1.org/v1"

def fetch_laps_for_session(session_key):
    r = requests.get(
        f"{BASE}/laps",
        params={"session_key": session_key},
        timeout=30,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json())

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sessions = pd.read_csv(DATA_DIR / "selected_sessions.csv")
    session_keys = sessions["session_key"].tolist()
    print("Fetching", len(session_keys), "race sessions")

    all_laps = []
    for session_key in session_keys:
        df = fetch_laps_for_session(session_key)
        print(f"session {session_key}: {len(df)} rows")
        if not df.empty:
            all_laps.append(df)

    laps_df = pd.concat(all_laps, ignore_index=True)
    out = DATA_DIR / "laps.csv"
    laps_df.to_csv(out, index=False)

    print("Saved:", out, "rows:", len(laps_df), "sessions:", laps_df["session_key"].nunique())
