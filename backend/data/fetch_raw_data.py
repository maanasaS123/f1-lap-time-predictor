import pandas as pd
import requests
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[2]   # project root
DATA_DIR = ROOT / "backend" / "data"
 
def fetch_laps(session_key: int, limit: int = 100000) -> pd.DataFrame:
    url = "https://api.openf1.org/v1/laps"
    r = requests.get(url, params={"session_key": session_key, "limit": limit})
    r.raise_for_status()
    return pd.DataFrame(r.json())
 
if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
 
    # pick ONE session_key from sessions.csv (example: 9161)
    df = fetch_laps(session_key=9161)
 
    out = DATA_DIR / "laps.csv"
    df.to_csv(out, index=False)
    print("Saved:", out, "rows:", len(df))