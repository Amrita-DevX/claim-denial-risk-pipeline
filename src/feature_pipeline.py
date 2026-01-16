# src/feature_pipeline.py

import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


class ClaimFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that performs all deterministic
    feature engineering steps for claim denial prediction.

    This transformer is used identically during:
    - notebook experimentation
    - model training
    - batch inference
    - future production scoring
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.high_risk_insurance = self.config["features"]["high_risk_insurance"]
        self.final_features = self.config["features"]["final_features"]

    def fit(self, X, y=None):
        # No learned parameters
        return self

    def transform(self, X):
        df = X.copy()

        # Handle length of stay
        df["length_of_stay"] = df["length_of_stay"].fillna(0)
        df["long_stay_flag"] = (df["length_of_stay"] > 5).astype(int)

        # Age bucketing
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 18, 35, 50, 65, 120],
            labels=["child", "young_adult", "adult", "senior", "elder"]
        )

        # Readmission risk
        df["readmitted_flag"] = df["readmitted_flag"].map({"Yes": 1, "No": 0}).fillna(0)

        # Insurance risk
        df["high_risk_insurance_flag"] = df["insurance_type"].isin(
            self.high_risk_insurance
        ).astype(int)

        # Diagnosis presence
        df["has_diagnosis"] = df["diagnosis_code"].notna().astype(int)

        # Provider experience
        df["years_experience"] = df["years_experience"].fillna(
            df["years_experience"].median()
        )
        df["low_experience_provider"] = (df["years_experience"] < 5).astype(int)

        # Return only model-ready features
        return df[self.final_features]
