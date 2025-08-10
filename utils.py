import joblib
import pandas as pd

def load_model():
    return joblib.load("artifacts/model.pkl")

def load_encoders():
    return joblib.load("artifacts/encoders.pkl")

def preprocess_input(input_data: dict):
    encoders = load_encoders()
    df = pd.DataFrame([input_data])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df
