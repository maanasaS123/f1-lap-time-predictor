import pandas as pd
import joblib
import os

# Paths
DATA_PATH = "backend/data/"
MODEL_PATH = "backend/model/"

def load_artifacts():
    """Load model, encoders, and scaler"""
    model = joblib.load(os.path.join(MODEL_PATH, "lap_time_model.pkl"))
    le_driver = joblib.load(os.path.join(DATA_PATH, "le_driver.pkl"))
    le_session = joblib.load(os.path.join(DATA_PATH, "le_session.pkl"))
    scaler = joblib.load(os.path.join(DATA_PATH, "scaler.pkl"))
    return model, le_driver, le_session, scaler


def predict_lap_time(driver_number, session_key, lap_number):
    """
    Predict lap time using ONLY pre-lap information
    """
    model, le_driver, le_session, scaler = load_artifacts()

    # Encode categorical inputs
    if driver_number not in le_driver.classes_:
        raise ValueError(
            f"Driver {driver_number} was not seen during training. "
            f"Available drivers: {list(le_driver.classes_)}"
        )
    driver_enc = le_driver.transform([driver_number])[0]

    if session_key not in le_session.classes_:
        raise ValueError(
            f"Session {session_key} was not seen during training."
        )
    session_enc = le_session.transform([session_key])[0]

    # Scale lap number
    lap_df = pd.DataFrame({'lap_number': [lap_number]})
    lap_scaled = scaler.transform(lap_df)

    # Create input DataFrame
    X = pd.DataFrame(
        [[driver_enc, session_enc, lap_scaled]],
        columns=["driver_number_enc", "session_key_enc", "lap_number"]
    )

    # Predict
    prediction = model.predict(X)[0]
    return prediction


if __name__ == "__main__":
    # Example usage (change values as needed)
    driver_number = 44     # Verstappen
    session_key = 7953     # Example session
    lap_number = 5

    predicted_time = predict_lap_time(driver_number, session_key, lap_number)
    print(f"Predicted lap time: {predicted_time:.3f} seconds")
