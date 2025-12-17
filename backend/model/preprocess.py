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
    return df
    
def clean_data(df):
    df = df.dropna(subset=[
        'driver_number', 'session_key', 'lap_number', 'lap_duration',
        'duration_sector_1', 'duration_sector_2', 'duration_sector_3'
    ])

    # Optional: remove duplicates
    df = df.drop_duplicates()
    # Optional: remove invalid lap times
    df = df[df['lap_duration'] > 0]

    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] == False]

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
    cols_to_scale = ['lap_number', 'duration_sector_1', 'duration_sector_2', 'duration_sector_3']

    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    joblib.dump(scaler, os.path.join(PROCESSED_PATH, 'scaler.pkl'))
    return df


def split_and_save(df):
    X = df[[
        'driver_number_enc',
        'session_key_enc',
        'lap_number',
        'duration_sector_1',
        'duration_sector_2',
        'duration_sector_3'
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

