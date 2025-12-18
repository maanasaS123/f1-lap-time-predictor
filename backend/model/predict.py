import os
import joblib
import pandas as pd

# Paths (repo-root relative when run from repo root)
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
    driver_number: int,
    session_key: int,
    lap_number: int,
    st_speed: float,
    stint_number: int,
    tyre_age: int,
    compound_code: int,
) -> float:
    """
    Predict lap_duration using the SAME 7 features used in training.
    compound_code mapping must match preprocess: SOFT=0, MEDIUM=1, HARD=2
    """
    model, le_driver, le_session = load_artifacts()

    # Encode categorical inputs exactly like preprocess
    if driver_number not in le_driver.classes_:
        raise ValueError(f"Driver {driver_number} not seen in training. Seen: {list(le_driver.classes_)}")
    if session_key not in le_session.classes_:
        raise ValueError(f"Session {session_key} not seen in training. Seen: {list(le_session.classes_)}")

    driver_enc = int(le_driver.transform([driver_number])[0])
    session_enc = int(le_session.transform([session_key])[0])

    # Build EXACT feature row (names + order)
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


if __name__ == "__main__":
    # Example (YOU must provide the 7 inputs)
    driver_number = 44
    session_key = 7953
    lap_number = 5

    st_speed = 310.0        # example
    stint_number = 1
    tyre_age = 4            # e.g., tyre_age_at_start + (lap_number - lap_start)
    compound_code = 0       # SOFT=0, MEDIUM=1, HARD=2

    predicted_time = predict_lap_time(
        driver_number, session_key, lap_number,
        st_speed, stint_number, tyre_age, compound_code
    )
    print(f"Predicted lap time: {predicted_time:.3f} seconds")
