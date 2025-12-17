import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
 
# takes the raw data, then cleans the data
RAW_PATH = "backend/data/laps.csv"
PROCESSED_PATH = "backend/data/"

def load_raw_data():
    return pd.read_csv(RAW_PATH)
 
def clean_data(df):
    # OpenF1 laps columns
    df = df.dropna(subset=["driver_number", "session_key", "lap_number", "lap_duration"])
    df = df.drop_duplicates()
    df = df[df["lap_duration"] > 0]
    return df
 
def encode_features(df):
    # driver_number is numeric, but encoding is okay
    le_driver = LabelEncoder()
    df["driver_number_enc"] = le_driver.fit_transform(df["driver_number"])
 
    le_session = LabelEncoder()
    df["session_key_enc"] = le_session.fit_transform(df["session_key"])
 
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    joblib.dump(le_driver, os.path.join(PROCESSED_PATH, "le_driver.pkl"))
    joblib.dump(le_session, os.path.join(PROCESSED_PATH, "le_session.pkl"))
 
    return df
 
def normalize_features(df):
    scaler = MinMaxScaler()
    df[["lap_number"]] = scaler.fit_transform(df[["lap_number"]])
    joblib.dump(scaler, os.path.join(PROCESSED_PATH, "scaler.pkl"))
    return df
 
def split_and_save(df):
    X = df[["driver_number_enc", "session_key_enc", "lap_number"]]
    y = df["lap_duration"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"), index=False)
 
if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df = encode_features(df)
    df = normalize_features(df)
    split_and_save(df)
    print("Data preprocessing complete! Processed files saved in:", PROCESSED_PATH)