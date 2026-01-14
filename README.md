# Healthcare Claim Denial Prediction (Enterprise ML Pipeline)

## Overview
This project implements a production-style machine learning pipeline to predict healthcare insurance claim denials using structured claims and billing data.

The solution follows an enterprise batch-scoring architecture with:
- Feature engineering and ML pipelines
- Scheduled batch inference
- Model versioning
- Optional API-based scoring

## Architecture
Raw Claims Data → ML Pipeline → Batch Scoring → Scheduled Execution → Scored Outputs → API / BI

## Tech Stack
- Python, SQL
- scikit-learn (pipelines)
- MLflow (experiment tracking)
- FastAPI (on-demand scoring)
- Cron (batch scheduling simulation)
- Power BI (downstream reporting)

## Project Structure
(standard enterprise ML repo layout)
