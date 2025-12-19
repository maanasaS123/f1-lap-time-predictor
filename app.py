import streamlit as st
import pandas as pd

from backend.model.predict import predict_lap_time

st.set_page_config(page_title="F1 Lap Time Predictor", layout="centered")

# ---------- Title ----------
st.title("🏎️ F1 Lap Time Predictor")
st.write("Predict Formula 1 lap times using a trained machine learning model.")

st.divider()

# ---------- Single Prediction ----------
st.header("Single Lap Prediction")

col1, col2 = st.columns(2)

with col1:
    driver_number = st.number_input("Driver Number", min_value=0, value=44)
    session_key = st.number_input("Session Key", min_value=0, value=7953)
    lap_number = st.number_input("Lap Number", min_value=1, value=5)
    stint_number = st.number_input("Stint Number", min_value=1, value=1)

with col2:
    st_speed = st.number_input("Speed (km/h)", min_value=0.0, value=310.0)
    tyre_age = st.number_input("Tyre Age (laps)", min_value=0, value=4)
    compound_code = st.selectbox(
        "Tyre Compound",
        options=[0, 1, 2],
        format_func=lambda x: ["SOFT", "MEDIUM", "HARD"][x]
    )

if st.button("Predict Lap Time"):
    try:
        prediction = predict_lap_time(
            driver_number,
            session_key,
            lap_number,
            st_speed,
            stint_number,
            tyre_age,
            compound_code
        )
        st.success(f"⏱️ Predicted Lap Time: **{prediction:.3f} seconds**")
    except Exception as e:
        st.error(str(e))

st.divider()
st.header("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload a CSV with the required columns",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded data:")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
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
        st.success("Predictions completed!")
        st.dataframe(df)

        st.download_button(
            label="Download Predictions CSV",
            data=df.to_csv(index=False),
            file_name="predicted_lap_times.csv",
            mime="text/csv"
        )