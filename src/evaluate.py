import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load test data
X_test = pd.read_csv("data/x_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Load model
model = joblib.load("models/model.pkl")

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# 🔥 THIS PART IS IMPORTANT
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)