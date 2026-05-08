# Heart Disease MLOps Project

**BITS Pilani MTech — MLOps Assignment (AMLCSZG523)**
**Student:** Eshwar Pradeep K | **Email:** eshwarpradeep.k95@gmail.com

---

## Overview

End-to-end MLOps pipeline that predicts heart disease risk using the UCI Heart Disease dataset. The project covers data preprocessing, EDA, model training with experiment tracking, REST API serving, containerization, CI/CD, and Kubernetes deployment.

---

## Project Structure

```
heart_disease_mlops/
├── src/
│   ├── data_preprocessing.py    # Data loading, cleaning, pipeline
│   ├── eda.py                   # EDA plots and analysis
│   ├── train.py                 # Model training + MLflow tracking
│   └── app.py                   # FastAPI prediction server
├── tests/
│   ├── test_preprocessing.py    # Unit tests: data pipeline
│   ├── test_model.py            # Unit tests: ML models
│   └── test_api.py              # Unit tests: API endpoints
├── data/
│   ├── raw/heart_disease.csv    # Raw UCI dataset
│   └── processed/               # Cleaned dataset (auto-generated)
├── models/                      # Trained model artifacts (auto-generated)
├── reports/
│   ├── eda_plots/               # EDA visualizations (auto-generated)
│   └── model_plots/             # Model evaluation plots (auto-generated)
├── .github/workflows/ci_cd.yml  # GitHub Actions CI/CD pipeline
├── manifests/
│   ├── deployment.yaml          # Kubernetes deployment + service + HPA
│   └── monitoring.yaml          # Prometheus + Grafana stack
├── Dockerfile                   # Multi-stage Docker build
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Preprocess the data
```bash
python src/data_preprocessing.py
```

### 5. Run EDA (generates plots in reports/eda_plots/)
```bash
python src/eda.py
```

### 6. Train models (logs to MLflow)
```bash
python src/train.py
```

### 7. View MLflow experiment dashboard
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### 8. Run unit tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 9. Start the API server locally
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs for interactive API docs
```

### 10. Test the API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0,
    "chol": 250.0, "fbs": 0.0, "restecg": 0.0, "thalach": 150.0,
    "exang": 0.0, "oldpeak": 1.5, "slope": 2.0, "ca": 0.0, "thal": 3.0
  }'
```

---

## Docker

### Build the image
```bash
docker build -t heart-disease-api:latest .
```

### Run locally
```bash
docker run -d -p 8000:8000 --name heart_api heart-disease-api:latest
```

### Test the container
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":1.5,"slope":2,"ca":0,"thal":3}'
```

---

## Kubernetes Deployment (Minikube)

```bash
# Start Minikube
minikube start

# Load local Docker image into Minikube
minikube image load heart-disease-api:latest

# Deploy the application
kubectl apply -f manifests/deployment.yaml

# Deploy monitoring
kubectl apply -f manifests/monitoring.yaml

# Get service URL
minikube service heart-disease-api-service --url

# Check pods
kubectl get pods
kubectl get services
```

---

## CI/CD Pipeline (GitHub Actions)

The pipeline in `.github/workflows/ci_cd.yml` runs automatically on every push:

| Job | Steps |
|-----|-------|
| **Lint** | flake8 (PEP8), black (formatting), isort (imports) |
| **Test** | pytest with coverage report |
| **Train** | Data preprocessing → EDA → Model training |
| **Docker** | Build image → Integration test → Push to Docker Hub |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + info |
| GET | `/health` | Detailed model status |
| GET | `/docs` | Interactive Swagger UI |
| POST | `/predict` | Heart disease prediction |

### Sample Request Body (`/predict`)
```json
{
  "age": 54.0,
  "sex": 1.0,
  "cp": 2.0,
  "trestbps": 130.0,
  "chol": 250.0,
  "fbs": 0.0,
  "restecg": 0.0,
  "thalach": 150.0,
  "exang": 0.0,
  "oldpeak": 1.5,
  "slope": 2.0,
  "ca": 0.0,
  "thal": 3.0
}
```

### Sample Response
```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "confidence": 0.8423,
  "probability_no_disease": 0.8423,
  "probability_disease": 0.1577,
  "model_version": "1.0.0"
}
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Dataset
- **Features:** 13 clinical features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Target:** Binary (0 = No Heart Disease, 1 = Heart Disease)
- **Samples:** 303 patients (164 no disease, 139 disease)

---

## Models

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~85% | ~0.91 |
| Random Forest | ~87% | ~0.93 |

---

## GitHub Setup (First Time — Step by Step)

1. Go to [github.com](https://github.com) and create a free account
2. Create a new repository called `heart-disease-mlops`
3. Install Git from [git-scm.com](https://git-scm.com)
4. Run these commands in your project folder:
```bash
git init
git add .
git commit -m "Initial MLOps assignment submission"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-mlops.git
git push -u origin main
```

---

*MLOps Assignment — BITS Pilani MTech | AMLCSZG523 | 2025*
