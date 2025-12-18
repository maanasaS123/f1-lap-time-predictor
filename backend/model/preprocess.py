# takes the raw data, then cleans the data

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

#tells the script where the raw csv is and where to save processed files
RAW_PATH = "backend/data/laps.csv"
PROCESSED_PATH = "backend/data/"

def load_raw_data():
    df = pd.read_csv(RAW_PATH)

    stints = pd.read_csv("backend/data/stints.csv")

    # merge stints onto laps using session_key + driver_number
    df = df.merge(
        stints,
        on=["session_key", "driver_number"],
        how="left"
    )

    # keep only the stint row that matches the lap_number
    df = df[
        (df["lap_number"] >= df["lap_start"]) &
        (df["lap_number"] <= df["lap_end"])
    ].copy()

    # tyre age for each lap
    df["tyre_age"] = df["tyre_age_at_start"] + (df["lap_number"] - df["lap_start"])

    # encode compound
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    df["compound_code"] = df["compound"].map(compound_map)

    return df
    
def clean_data(df):
    df = df.dropna(subset=[
    'driver_number', 'session_key', 'lap_number', 'lap_duration',
    'stint_number', 'tyre_age', 'compound_code', 'st_speed'
])


    # Optional: remove duplicates
    df = df.drop_duplicates()
    # Optional: remove invalid lap times
    df = df[df['lap_duration'] > 0]

    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] == False]
    # remove absurdly slow laps (pit in / safety car)
    df = df[df["lap_duration"] < df["lap_duration"].quantile(0.96)]


    return df

def encode_features(df):
    le_driver = LabelEncoder()
    df['driver_number_enc'] = le_driver.fit_transform(df['driver_number'])

    le_session = LabelEncoder()
    df['session_key_enc'] = le_session.fit_transform(df['session_key'])

    # Save encoders for future use
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    joblib.dump(le_driver, os.path.join(PROCESSED_PATH, 'le_driver.pkl'))
    joblib.dump(le_session, os.path.join(PROCESSED_PATH, 'le_session.pkl'))

    return df

def normalize_features(df):


    return df


def split_and_save(df):
    X = df[[
        'driver_number_enc',
        'session_key_enc',
        'lap_number',
        'st_speed',
        'stint_number',
        'tyre_age',
        'compound_code'
    ]]


    y = df['lap_duration']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_PATH, 'y_test.csv'), index=False)

if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df = encode_features(df)
    df = normalize_features(df)
    split_and_save(df)
    print("Data preprocessing complete! Processed files saved in:", PROCESSED_PATH)

