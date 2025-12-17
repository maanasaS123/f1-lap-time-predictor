import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "backend" / "data" 
MODELS = ROOT / "backend" / "model"
MODELS.mkdir(exist_ok=True)

# Load data
X_train = pd.read_csv(PROCESSED / "X_train.csv")
X_test  = pd.read_csv(PROCESSED / "X_test.csv")
y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()
y_test  = pd.read_csv(PROCESSED / "y_test.csv").squeeze()

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("MAE:", mae)
print("RMSE:", rmse)

# Save model
joblib.dump(model, MODELS / "lap_time_model.pkl")
print("Model saved!")
