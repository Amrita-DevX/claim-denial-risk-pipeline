# Claim Denial Risk Pipeline - Documentation

## Directory & File Descriptions

### `/api` - FastAPI Application

REST API for serving model predictions.

- **app.py**: Main FastAPI application with endpoints for real-time predictions

### `/config` - Configuration

Project configuration and parameters.

- **config.yaml**: Feature names, model settings, and pipeline configuration

### `/data/incoming` - Batch Data

New claims data for batch inference/scoring.

- **new_claims_sample.csv**: Sample dataset of claims to be scored

### `/models` - Trained Models

Serialized ML models.

- **claim_denial_model.pkl**: Trained scikit-learn model (main artifact)

### `/notebooks` - Jupyter Notebooks

Exploratory analysis and experimentation.

- **01_data_exploration.ipynb**: Initial data exploration and visualization
- **02_feature_engineering.ipynb**: Feature engineering experimentation
- **03_model_training.ipynb**: Model training and evaluation

### `/src` - Source Code

Training, preprocessing, and inference logic.

Contains modules like:
- Feature engineering pipeline
- Data loading utilities
- Helper functions

### Root Files

| File | Purpose |
|------|---------|
| **.gitignore** | Specifies files to exclude from Git |
| **Dockerfile** | Docker container configuration |
| **README.md** | Project documentation |
| **requirements.txt** | Python package dependencies |
| **run_batch_score.bat** | Windows batch script for scoring claims |
| **test_mlflow.py** | MLflow experiment tracking tests |

---

## Project Workflow

```
Data (data/incoming/)
    ↓
Load & Preprocess
    ↓
Feature Engineering (src/)
    ↓
Model Inference (models/claim_denial_model.pkl)
    ↓
Predictions Output
    ↓
Batch Score (run_batch_score.bat)
    ↓
API Serving (api/app.py)
```

---

## Quick Start Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI Server

```bash
uvicorn api.app:app --reload
```

### Test MLflow

```bash
python test_mlflow.py
```

### Batch Score Claims (Windows)

```bash
run_batch_score.bat
```

### Build Docker Image

```bash
docker build -t claim-denial-api
```

### Run Docker Container

```bash
docker run -p 8000:8000 claim-denial-api
```

---

## API Documentation

Once the API is running, visit:

```
http://localhost:8000/docs
```

You will see an interactive Swagger UI with all available endpoints and the ability to test them directly.

---

## Key Technologies

- **FastAPI**: REST API framework for building modern Python web APIs
- **Scikit-learn**: Machine learning models and algorithms
- **Pandas**: Data manipulation and analysis
- **Pydantic**: Data validation and settings management
- **Docker**: Containerization for deployment
- **MLflow**: Experiment tracking and model management
- **Jupyter**: Exploratory data analysis and experimentation

---

## Deployment

### Local Testing

Start the API locally:

```bash
uvicorn api.app:app --reload
```

Test prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "billed_amount": 5000,
    "age": 45,
    "insurance_type": "Medicare",
    "visit_type": "Inpatient",
    "department_x": "Cardiology",
    "length_of_stay": 3,
    "years_experience": 10
  }'
```

### Docker Deployment

Build the Docker image:

```bash
docker build -t claim-denial-api:1.0 .
```

Run the Docker container:

```bash
docker run -p 8000:8000 claim-denial-api:1.0
```

### Render Deployment

Push code to GitHub:

```bash
git add .
git commit -m "Deploy to Render"
git push
```

Then connect your GitHub repository to Render and deploy automatically.

**Steps:**
1. Go to https://render.com/
2. Connect your GitHub account
3. Select your repository
4. Render will automatically detect the Dockerfile
5. Configure environment variables if needed
6. Deploy!

**Live API**  
👉 https://claim-denial-risk-pipeline.onrender.com


---

## Files to Track in Git

### ✅ Include in Git

Track these files in version control:

- **Source code**: `api/`, `src/`, `notebooks/`
- **Configuration**: `config.yaml`, `Dockerfile`, `requirements.txt`
- **Documentation**: `README.md`
- **Scripts**: `run_batch_score.bat`, `test_mlflow.py`
- **Version control**: `.gitignore`

### ❌ Exclude from Git (in .gitignore)

Do NOT track these files:

- **Models**: `models/*.pkl`
- **Data**: `data/incoming/`, `data/output/`
- **Cache**: `__pycache__/`, `.ipynb_checkpoints/`
- **Environment**: `venv/`, `.env`
- **MLflow outputs**: `mlruns/`, `mlflow.db`
- **OS files**: `.DS_Store`, `Thumbs.db`

---

## Example .gitignore Content

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
venv/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/*
data/processed/*
data/incoming/*
data/output/*
outputs/*

# Models
models/*.pkl

# OS
.DS_Store
Thumbs.db

# MLflow
mlflow.db
mlruns/
.mlflow/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.tmp
*.temp
.trash/
```

---

## Development Workflow

### 1. Data Exploration
Start with Jupyter notebooks in `/notebooks/`:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Feature Engineering
Develop features in:
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### 3. Model Training
Train and evaluate models in:
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 4. Create Production Code
Convert notebook logic to `/src/` modules:
- `src/feature_pipeline.py` - Feature engineering
- `src/train_model.py` - Model training
- `src/data_loader.py` - Data loading

### 5. Build API
Create `/api/app.py` for serving predictions:
```bash
uvicorn api.app:app --reload
```

### 6. Test & Deploy
- Test locally with Docker
- Push to GitHub
- Deploy to Render

---

## Monitoring & Logging

### MLflow Tracking

Track experiments:
```bash
python test_mlflow.py
```

View MLflow UI:
```bash
mlflow ui
```

Then visit: `http://localhost:5000`

### API Logs

View Docker logs:
```bash
docker logs <container-id>
```

View Render logs:
- Go to Render dashboard
- Select your service
- View logs in real-time

---

## Troubleshooting

### Docker Issues

**Docker not found:**
```bash
# Make sure Docker Desktop is running
# Verify installation
docker --version
```

**Port already in use:**
```bash
# Use different port
docker run -p 9000:8000 claim-denial-api:1.0
```

### API Issues

**Module not found:**
```bash
# Ensure requirements.txt is updated
pip install -r requirements.txt
```

**Model file not found:**
```bash
# Check models/ directory exists and has .pkl file
ls models/
```

### Deployment Issues

**Render deployment fails:**
- Check Dockerfile syntax
- Verify all files are in Git
- Check build logs in Render dashboard

---

## Project Statistics

- **Models**: 1 (claim_denial_model.pkl)
- **API Endpoints**: Multiple prediction endpoints
- **Notebooks**: 3 (exploration, feature engineering, training)
- **Configuration Files**: 1 (config.yaml)
- **Batch Data**: 1 sample file (new_claims_sample.csv)

---

## Contact & Support

For issues or questions:
1. Check logs: `docker logs <container-id>`
2. Review README.md
3. Check Render dashboard for deployment logs
4. Review API documentation at `/docs` endpoint

---

## License

This project is for educational purposes.