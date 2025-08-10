import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv("dataset.csv")

# Drop irrelevant columns
drop_cols = ["id", "date", "umpire1", "umpire2", "umpire3"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Fill missing values
df = df.fillna(method='ffill')

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save processed data
os.makedirs("artifacts", exist_ok=True)
df.to_csv("artifacts/processed_data.csv", index=False)

# Save encoders
joblib.dump(label_encoders, "artifacts/encoders.pkl")

print("✅ Data preprocessing complete. Processed data & encoders saved.")
