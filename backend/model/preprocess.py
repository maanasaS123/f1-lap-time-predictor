import pandas as pd #for working with csv files and dataframes
import os #for handling file paths and directories
from sklearn.preprocessing import LabelEncoder, MinMaxScaler #scales numerical features to a 0-1 range
from sklearn.model_selection import train_test_split #splits data into training and testing sets
import joblib #saves python objects (like encoders and scalers) to disk so you can reuse them later
 
# takes the raw data, then cleans the data
RAW_PATH = "backend/data/laps.csv"
PROCESSED_PATH = "backend/data/"

#returns the csv at RAW_PATH into a pandas dataframe and returns the raw data for further processing
def load_raw_data():
    return pd.read_csv(RAW_PATH)
 
def clean_data(df):
    # OpenF1 laps columns
    df = df.dropna(subset=["driver_number", "session_key", "lap_number", "lap_duration"]) #removes rows where critical columns are missing
    df = df.drop_duplicates() #removes duplicate rows to avoid bias
    df = df[df["lap_duration"] > 0] #removes laps with zero or negative duration, which are invalid
    return df

#converts categorical columns into integers
def encode_features(df):
    # driver_number is numeric, but encoding is okay
    le_driver = LabelEncoder()
    df["driver_number_enc"] = le_driver.fit_transform(df["driver_number"]) #numeric encoding of drivers
 
    le_session = LabelEncoder()
    df["session_key_enc"] = le_session.fit_transform(df["session_key"]) #numeric encoding of race sessions
 
    os.makedirs(PROCESSED_PATH, exist_ok=True) #ensure the processed file exists
    joblib.dump(le_driver, os.path.join(PROCESSED_PATH, "le_driver.pkl")) #saves the encoders as .pkl files so the mapping can be used later when making predictions
    joblib.dump(le_session, os.path.join(PROCESSED_PATH, "le_session.pkl"))
 
    return df
 
def normalize_features(df):
    scaler = MinMaxScaler() #scales lap number to 0-1
    df[["lap_number"]] = scaler.fit_transform(df[["lap_number"]]) #computes min/max and scales the column
    joblib.dump(scaler, os.path.join(PROCESSED_PATH, "scaler.pkl"))
    return df
 
def split_and_save(df):
    X = df[["driver_number_enc", "session_key_enc", "lap_number"]] #X --> input features for the model
    y = df["lap_duration"] #y --> target variable (lap duration)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 #splits data into 80% training and 20% testing, the random state ensures reproducibility
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