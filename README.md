# 🏥 Claim Denial Risk Prediction Pipeline

**Live API**  
👉 https://claim-denial-risk-pipeline.onrender.com

---

## 📌 Business Problem

Healthcare insurance claims are frequently denied due to administrative, clinical, or policy-related reasons. Claim denials lead to delayed reimbursements, increased operational costs, and poor provider experience.

The objective of this project is to **predict the likelihood of a claim being denied** before adjudication so that high-risk claims can be prioritized for review and corrective action.

---

## 🎯 Solution Overview

This project implements a **production-style machine learning pipeline** that:

- Engineers healthcare-specific features from raw claim data  
- Trains a class-imbalance-aware Logistic Regression model  
- Supports **batch scoring** and **real-time API inference**  
- Uses a single sklearn pipeline to avoid training–serving skew  
- Is containerized using Docker and deployed on the cloud  

---

## 🏗 Architecture Diagram

               ┌────────────────────────┐
               │   Incoming Claim Data  │
               │ (API request / CSV)    │
               └────────────┬───────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │ Feature Engineering Pipeline │
             │ (Custom sklearn Transformer)│
             │                             │
             │ • Length of stay flags      │
             │ • Age buckets               │
             │ • Insurance risk flags      │
             │ • Provider experience flags │
             └────────────┬────────────────┘
                          │
                          ▼
           ┌─────────────────────────────────┐
           │ Preprocessing Pipeline           │
           │                                 │
           │ • Numeric imputation + scaling  │
           │ • Categorical encoding          │
           └────────────┬────────────────────┘
                        │
                        ▼
          ┌──────────────────────────────────┐
          │ ML Model                          │
          │ Logistic Regression               │
          │ (class_weight = balanced)         │
          └────────────┬─────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ Output                                   │
    │ • Denial prediction (0 / 1)              │
    │ • Denial risk probability                │
    └──────────────────────────────────────────┘

---

## 📊 Model Performance Summary

Due to strong class imbalance (few denied claims), recall and risk ranking were prioritized over raw accuracy.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-----|----------|-----------|--------|--------|
| Random Forest | High | Low | Low | ~0.50 |
| **Logistic Regression (selected)** | Moderate | Low | **Higher** | ~0.49 |

### Why Logistic Regression?
- Better recall on denied claims  
- More stable probability outputs  
- Easier interpretability for risk scoring use cases  

---

## 🧪 Data Leakage Prevention

- Only pre-adjudication features are used  
- No post-outcome signals included  
- Feature logic is embedded inside the pipeline  
- Same transformations are applied during:
  - Training
  - Batch scoring
  - Real-time inference  

---

## ⚖️ Class Imbalance Handling

- Used `class_weight="balanced"`  
- Evaluated recall and ROC-AUC  
- Threshold tuning performed to optimize business relevance  

---

## 🚀 Deployment

### 🔹 Real-Time API (FastAPI)

- Endpoint: `/predict`
- Accepts raw claim attributes as JSON
- Returns denial prediction and probability

Swagger UI:

---

### 🔹 Batch Scoring

- New claims placed in `data/incoming/`
- Scheduled execution via Windows Task Scheduler
- Outputs scored claims with denial probability

---

## 🐳 Docker & Cloud Deployment

- Application packaged using Docker
- Trained model artifact included in container
- Deployed as a cloud service on Render

**Production Note:**  
In real enterprise environments, model artifacts are typically loaded from object storage or a model registry rather than committed to source control.

---

## 📁 Project Structure


```
claim-denial-risk-pipeline/
│
├── api/                          # FastAPI application
│   ├── app.py                    # Main FastAPI app with prediction endpoints
│   ├── schemas.py                # Pydantic models for request/response
│   └── utils.py                  # Helper functions for API
│
├── src/                          # Training, feature pipeline, batch scoring
│   ├── train_model.py            # Model training script
│   ├── feature_pipeline.py       # Feature engineering pipeline
│   ├── batch_score.py            # Batch inference for multiple claims
│   ├── data_loader.py            # Data loading utilities
│   ├── utils.py                  # Helper functions
│   └── __init__.py               # Package initialization
│
├── models/                       # Trained ML model
│   ├── claim_denial_model.pkl    # Serialized trained model
│   └── scaler.pkl                # Feature scaler (if applicable)
│
├── data/
│   ├── raw/                      # Original raw data
│   │   └── claims_data.csv
│   ├── processed/                # Cleaned and processed data
│   │   └── processed_claims.csv
│   ├── incoming/                 # New claims for batch inference
│   │   └── new_claims.csv
│   └── output/                   # Scored outputs
│       └── scored_claims.csv
│
├── config/                       # YAML configuration
│   └── config.yaml               # Feature names, model parameters
│
├── notebooks/                    # EDA and experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── README.md                     # Project documentation
└── mlflow.db                     # MLflow experiment tracking (auto-generated)
```


---

## 🧠 Skills Demonstrated

- End-to-end ML pipeline design  
- Healthcare feature engineering  
- Handling imbalanced datasets  
- MLflow experiment tracking  
- Batch and real-time inference  
- FastAPI  
- Docker  
- Cloud deployment  

---

## 🔗 Live Demo

👉 https://claim-denial-risk-pipeline.onrender.com/docs

---

## ✅ Project Status

✔ Model trained  
✔ Batch scoring implemented  
✔ Real-time API deployed  
✔ Dockerized and cloud hosted  
