import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
BASE = "https://api.openf1.org/v1"

def fetch_lap(session_key, lap_number):
    r = requests.get(
        f"{BASE}/laps",
        params={"session_key": session_key, "lap_number": lap_number},
        timeout=20,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json())

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    session_key = 7953  # ✅ confirmed full race

    all_laps = []
    for lap in range(1, 80):  # safe upper bound
        df_lap = fetch_lap(session_key, lap)
        if not df_lap.empty:
            print(f"lap {lap}: {len(df_lap)} rows")
            all_laps.append(df_lap)

    laps_df = pd.concat(all_laps, ignore_index=True)
    out = DATA_DIR / "laps.csv"
    laps_df.to_csv(out, index=False)

    print("Saved:", out, "rows:", len(laps_df))
