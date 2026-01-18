# src/batch_score.py

import pandas as pd
import joblib
from pathlib import Path
import yaml
from datetime import datetime


# -----------------------------
# Resolve project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# -----------------------------
# Paths from config
# -----------------------------
MODEL_PATH = PROJECT_ROOT / config["paths"]["model_dir"] / "claim_denial_model.pkl"
INCOMING_DIR = PROJECT_ROOT / config["paths"]["new_data_dir"]
OUTPUT_DIR = PROJECT_ROOT / config["paths"]["output_dir"]

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Load trained pipeline
# -----------------------------
pipeline = joblib.load(MODEL_PATH)


# -----------------------------
# Find incoming claim files
# -----------------------------
incoming_files = list(INCOMING_DIR.glob("*.csv"))

if not incoming_files:
    raise FileNotFoundError(
        f"No new claim files found in {INCOMING_DIR}"
    )

# Simulate daily processing (process first available file)
input_file = incoming_files[0]
print(f"Processing file: {input_file.name}")

df = pd.read_csv(input_file)


# -----------------------------
# Run batch scoring
# -----------------------------
df["denial_probability"] = pipeline.predict_proba(df)[:, 1]
df["denial_prediction"] = (df["denial_probability"] >= 0.5).astype(int)


# -----------------------------
# Save scored output
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"scored_claims_{timestamp}.csv"

df.to_csv(output_file, index=False)

print("Batch scoring completed successfully.")
print(f"Output saved to: {output_file}")
