import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

df = pd.read_csv("artifacts/ipl_colab.csv")

X = df.drop("winner", axis=1)
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")

print(f"✅ Model trained with accuracy: {accuracy:.2f}")

