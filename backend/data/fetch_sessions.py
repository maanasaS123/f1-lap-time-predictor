import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"

def fetch_sessions(year=2023):
    url = "https://api.openf1.org/v1/sessions"
    r = requests.get(url, params={"year": year})
    r.raise_for_status()
    return pd.DataFrame(r.json())

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_sessions(2023)
    out = DATA_DIR / "sessions.csv"
    df.to_csv(out, index=False)

    print(df[["session_key", "session_name", "circuit_short_name", "year"]].head(10))
