import joblib

model = joblib.load("models/ckd_xgb_model.pkl")

print("XGBoost Model Loaded Successfully")