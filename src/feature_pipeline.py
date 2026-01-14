# src/feature_pipeline.py

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class ClaimFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer that performs all business
    feature engineering steps used during notebook experimentation.

    This ensures identical feature transformations during:
    - training
    - batch inference
    - future production scoring
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration file to keep logic configurable and reusable
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract high-risk insurance list from config
        self.high_risk_insurance = self.config["features"]["high_risk_insurance"]

    def fit(self, X, y=None):
        """
        Fit method required by sklearn API.
        No fitting is required because all transformations
        are rule-based and not learned from data.
        """
        return self

    def transform(self, X):
        """
        Applies all feature engineering steps exactly as defined
        in the feature engineering notebook.
        """

        # Create a copy to avoid mutating original dataframe
        df = X.copy()

        # Replace missing length_of_stay with 0 for outpatient cases
        df["length_of_stay"] = df["length_of_stay"].fillna(0)

        # Create long stay flag (business rule: stay > 5 days)
        df["long_stay_flag"] = (df["length_of_stay"] > 5).astype(int)

        # Bucketize age into business-relevant categories
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 18, 35, 50, 65, 120],
            labels=["child", "young_adult", "adult", "senior", "elder"]
        )

        # Convert readmission indicator to binary
        df["readmitted_flag"] = df["readmitted_flag"].map({"Yes": 1, "No": 0})
        df["readmitted_flag"] = df["readmitted_flag"].fillna(0)

        # Flag high-risk insurance types
        df["high_risk_insurance_flag"] = df["insurance_type"].isin(
            self.high_risk_insurance
        ).astype(int)

        # Indicator for presence of diagnosis code
        df["has_diagnosis"] = df["diagnosis_code"].notna().astype(int)

        # Fill missing provider experience with median
        df["years_experience"] = df["years_experience"].fillna(
            df["years_experience"].median()
        )

        # Flag low experience providers (< 5 years)
        df["low_experience_provider"] = (df["years_experience"] < 5).astype(int)

        # Select final feature set used for modeling
        selected_features = self.config["features"]["final_features"]

        return df[selected_features]
