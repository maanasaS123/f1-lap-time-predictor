import os
import joblib
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error

# Paths
DATA_PATH = "backend/data/"
MODEL_PATH = "backend/model/"

FEATURE_COLUMNS = [
    "driver_number_enc",
    "session_key_enc",
    "lap_number",
    "st_speed",
    "stint_number",
    "tyre_age",
    "compound_code",
]

def load_artifacts():
    model = joblib.load(os.path.join(MODEL_PATH, "lap_time_model.pkl"))
    le_driver = joblib.load(os.path.join(DATA_PATH, "le_driver.pkl"))
    le_session = joblib.load(os.path.join(DATA_PATH, "le_session.pkl"))
    return model, le_driver, le_session

def predict_lap_time(
    driver_number, session_key, lap_number,
    st_speed, stint_number, tyre_age, compound_code
):
    model, le_driver, le_session = load_artifacts()

    # Encode categorical inputs exactly like preprocess
    if driver_number not in le_driver.classes_:
        raise ValueError(f"Driver {driver_number} not seen in training")
    if session_key not in le_session.classes_:
        raise ValueError(f"Session {session_key} not seen in training")

    driver_enc = int(le_driver.transform([driver_number])[0])
    session_enc = int(le_session.transform([session_key])[0])

    X = pd.DataFrame([{
        "driver_number_enc": driver_enc,
        "session_key_enc": session_enc,
        "lap_number": int(lap_number),
        "st_speed": float(st_speed),
        "stint_number": int(stint_number),
        "tyre_age": int(tyre_age),
        "compound_code": int(compound_code),
    }], columns=FEATURE_COLUMNS)

    pred = float(model.predict(X)[0])
    return pred

def predict_from_csv(input_csv, output_csv, y_true_csv=None):
    df = pd.read_csv(input_csv)
    predictions = []

    for _, row in df.iterrows():
        pred = predict_lap_time(
            row["driver_number"],
            row["session_key"],
            row["lap_number"],
            row["st_speed"],
            row["stint_number"],
            row["tyre_age"],
            row["compound_code"]
        )
        predictions.append(pred)

    df["predicted_lap_time"] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # Optional: compute metrics if true values provided
    if y_true_csv:
        y_true = pd.read_csv(y_true_csv)["lap_time"]
        mae = mean_absolute_error(y_true, predictions)
        print(f"Mean Absolute Error (MAE): {mae:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict lap times")
    parser.add_argument("--input", type=str, help="Path to input CSV")
    parser.add_argument("--output", type=str, help="Path to save predictions CSV")
    parser.add_argument("--y_true", type=str, default=None, help="Optional true lap times CSV for metrics")
    args = parser.parse_args()

    if args.input and args.output:
        predict_from_csv(args.input, args.output, args.y_true)
    else:
        # Hardcoded example (for testing)
        driver_number = 44
        session_key = 7953
        lap_number = 5
        st_speed = 310.0
        stint_number = 1
        tyre_age = 4
        compound_code = 0

        predicted_time = predict_lap_time(
            driver_number, session_key, lap_number,
            st_speed, stint_number, tyre_age, compound_code
        )
        print(f"Predicted lap time: {predicted_time:.3f} seconds")
