from pathlib import Path
import pandas as pd
from typing import Dict

# Import shared utility to load configuration
from utils import load_config


# --------------------------------------------------
# Load configuration values
# --------------------------------------------------

# Load config.yaml once so paths and column names are consistent
config = load_config()

# Directory where raw CSV files are stored (never modified)
RAW_DATA_DIR = Path(config["paths"]["raw_data"])

# Directory where cleaned / merged data will be saved
PROCESSED_DATA_DIR = Path(config["paths"]["processed_data"])

# Column that uniquely identifies a claim
CLAIM_ID_COL = config["data"]["claim_id_column"]

# Target variable used for model training
TARGET_COL = config["data"]["target_column"]


# --------------------------------------------------
# Step 1: Load raw healthcare tables
# --------------------------------------------------

def load_raw_tables() -> Dict[str, pd.DataFrame]:
    # This function loads all required raw datasets
    # Keeping raw tables separate makes the pipeline modular and auditable

    tables = {
        "claims": pd.read_csv(RAW_DATA_DIR / "claims_and_billing.csv"),
        "denials": pd.read_csv(RAW_DATA_DIR / "denials.csv"),
        "patients": pd.read_csv(RAW_DATA_DIR / "patients.csv"),
        "encounters": pd.read_csv(RAW_DATA_DIR / "encounters.csv"),
        "providers": pd.read_csv(RAW_DATA_DIR / "providers.csv"),
    }

    return tables


# --------------------------------------------------
# Step 2: Create base dataset with denial label
# --------------------------------------------------

def build_claim_denial_dataset(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Extract claims and denials tables
    claims = tables["claims"]
    denials = tables["denials"]

    # Create binary target variable:
    # 1 = claim denied, 0 = claim not denied
    denials[TARGET_COL] = 1

    # Left join ensures all claims are retained
    # Missing denial records mean the claim was paid
    df = claims.merge(
        denials[[CLAIM_ID_COL, TARGET_COL]],
        on=CLAIM_ID_COL,
        how="left"
    )

    # Replace missing values with 0 (not denied)
    # Convert to integer for ML compatibility
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    return df


# --------------------------------------------------
# Step 3: Enrich claims with context data
# --------------------------------------------------

def enrich_claims(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Add patient-level information
    # This helps models learn demographic and insurance-related patterns
    df = df.merge(
        tables["patients"],
        on="patient_id",
        how="left"
    )

    # Add encounter-level information
    # Includes admission type, length of stay, department, etc.
    df = df.merge(
        tables["encounters"],
        on="encounter_id",
        how="left"
    )

    # Add provider-level information
    # Helps capture provider behavior and billing patterns
    df = df.merge(
        tables["providers"],
        on="provider_id",
        how="left"
    )

    return df


# --------------------------------------------------
# Step 4: Save processed dataset
# --------------------------------------------------

def save_processed_data(df: pd.DataFrame, filename: str):
    # Create processed directory if it does not exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save merged dataset for feature engineering and modeling
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)


# --------------------------------------------------
# Main execution block
# --------------------------------------------------

if __name__ == "__main__":
    # Load raw data tables
    tables = load_raw_tables()

    # Build claim-level dataset with denial label
    base_df = build_claim_denial_dataset(tables)

    # Enrich dataset with patient, encounter, and provider data
    final_df = enrich_claims(base_df, tables)

    # Save final dataset for downstream pipelines
    save_processed_data(final_df, "claims_denial_base.csv")

    print("Data loading and preparation completed.")
