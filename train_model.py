"""
Train the Diabetes Risk Prediction model and save it as diabetes_prediction_model.pkl

Usage:
    python train_model.py

- If a file named 'diabetes.csv' (the Pima Indians Diabetes dataset, with the
  columns Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
  DiabetesPedigreeFunction, Age, Outcome) is placed in this folder, the model
  is trained on that real dataset.
- Otherwise a synthetic Pima-style dataset is generated so the project always
  runs out of the box.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def make_synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(6.0, 5.5, n) + 18, 21, 81).round()
    preg = np.clip(rng.poisson(np.clip((age - 18) / 8, 0.3, 6))
                   + rng.integers(0, 2, n), 0, 17)
    bmi = np.clip(rng.normal(32, 6.5, n), 16, 65).round(1)
    glucose = np.clip(rng.normal(118, 30, n) + (bmi - 32) * 0.6, 56, 199).round()
    bp = np.clip(rng.normal(70, 12, n) + (age - 33) * 0.2, 40, 122).round()
    skin = np.clip(rng.normal(21, 10, n) + (bmi - 32) * 0.7, 0, 60).round()
    insulin = np.clip(rng.lognormal(4.5, 0.8, n) + (glucose - 118) * 0.8, 0, 600).round()
    dpf = np.clip(rng.lognormal(-0.9, 0.55, n), 0.08, 2.4).round(3)

    z = 1.7 * (-9.3 + 0.036 * glucose + 0.075 * bmi + 0.028 * age
               + 0.11 * preg + 1.0 * dpf + 0.004 * bp - 0.0005 * insulin) - 1.2
    prob = 1 / (1 + np.exp(-z))
    outcome = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome,
    })


def main():
    if os.path.exists("diabetes.csv"):
        df = pd.read_csv("diabetes.csv")
        print("Training on real dataset: diabetes.csv", df.shape)
        # In the Pima dataset a 0 means 'missing' for these columns
        for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    else:
        df = make_synthetic()
        print("diabetes.csv not found -> using synthetic Pima-style data", df.shape)

    X, y = df[FEATURES], df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42)
    model.fit(X_train.values, y_train.values)

    pred = model.predict(X_test.values)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred))

    joblib.dump(model, "diabetes_prediction_model.pkl", compress=3)
    print("Saved diabetes_prediction_model.pkl")


if __name__ == "__main__":
    main()
