# src/train_model.py

import pandas as pd
import yaml
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

from feature_pipeline import ClaimFeatureEngineer


# -----------------------------
# Load config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
numeric_features = config["features"]["numeric_features"]
categorical_features = config["features"]["categorical_features"]


# -----------------------------
# Load raw data (NO precomputed features)
# -----------------------------
DATA_PATH = PROJECT_ROOT / config["paths"]["processed_data"] / "claims_denial_base.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["is_denied"])
y = df["is_denied"]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# -----------------------------
# Preprocessing pipelines
# -----------------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# -----------------------------
# Final training pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("feature_engineering", ClaimFeatureEngineer(str(CONFIG_PATH))),
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# -----------------------------
# Train model
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Save trained model as PKL
# -----------------------------
MODEL_DIR = PROJECT_ROOT / config["paths"]["model_dir"]
MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "claim_denial_model.pkl"
joblib.dump(pipeline, model_path)

print(f"Model saved at: {model_path}")

# -----------------------------
# MLflow tracking
# -----------------------------
mlflow.set_experiment("claim_denial_logistic_model")

with mlflow.start_run(run_name="LogisticRegression_Balanced"):
    pipeline.fit(X_train, y_train)

    metrics = pipeline.score(X_test, y_test)

    mlflow.log_params({
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "max_iter": 1000
    })

    mlflow.log_metric("accuracy", metrics)
    mlflow.sklearn.log_model(pipeline, name="model")
