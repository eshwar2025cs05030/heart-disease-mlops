"""
FastAPI Model Serving Application
Heart Disease UCI Dataset - MLOps Assignment
BITS Pilani MTech (AMLCSZG523)

Endpoints:
  GET  /         → health check
  GET  /health   → detailed health
  POST /predict  → heart disease prediction
"""

import logging
import os
import pickle
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_requests.log", mode="a"),
    ]
)
logger = logging.getLogger("heart_disease_api")

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best_model.pkl"))
PIPELINE_PATH = os.environ.get(
    "PIPELINE_PATH", os.path.join(BASE_DIR, "models", "preprocessing_pipeline.pkl")
)

# Load model and pipeline at startup
model = None
pipeline = None


def load_artifacts():
    global model, pipeline
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PIPELINE_PATH, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Model loaded from: {MODEL_PATH}")
        logger.info(f"Pipeline loaded from: {PIPELINE_PATH}")
    except FileNotFoundError as e:
        logger.warning(f"Artifacts not found: {e}. Run train.py first.")


# App
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps Assignment — BITS Pilani MTech (AMLCSZG523)\n\n"
        "Predicts heart disease risk from patient health features."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response schemas
class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years", example=54.0)
    sex: float = Field(..., ge=0, le=1, description="Sex (1=Male, 0=Female)", example=1.0)
    cp: float = Field(..., ge=1, le=4, description="Chest pain type (1-4)", example=2.0)
    trestbps: float = Field(
        ..., ge=50, le=300, description="Resting blood pressure (mmHg)", example=130.0
    )
    chol: float = Field(
        ..., ge=100, le=600, description="Serum cholesterol (mg/dl)", example=250.0
    )
    fbs: float = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=True)", example=0.0
    )
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG (0-2)", example=0.0)
    thalach: float = Field(
        ..., ge=60, le=250, description="Max heart rate achieved", example=150.0
    )
    exang: float = Field(
        ..., ge=0, le=1, description="Exercise induced angina (1=Yes)", example=0.0
    )
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0, description="ST depression induced by exercise", example=1.5
    )
    slope: float = Field(
        ..., ge=1, le=3, description="Slope of peak exercise ST (1-3)", example=2.0
    )
    ca: float = Field(
        ..., ge=0, le=3, description="Number of major vessels (0-3)", example=0.0
    )
    thal: float = Field(
        ...,
        ge=3,
        le=7,
        description="Thalassemia (3=normal, 6=fixed defect, 7=reversable)",
        example=3.0,
    )

    class Config:
        schema_extra = {
            "example": {
                "age": 54.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0,
                "chol": 250.0, "fbs": 0.0, "restecg": 0.0, "thalach": 150.0,
                "exang": 0.0, "oldpeak": 1.5, "slope": 2.0, "ca": 0.0, "thal": 3.0
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probability_no_disease: float
    probability_disease: float
    model_version: str = "1.0.0"


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 4)
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} [{duration}s] "
        f"client={request.client.host if request.client else 'unknown'}"
    )
    return response


@app.on_event("startup")
async def startup_event():
    load_artifacts()
    logger.info("Heart Disease Prediction API started successfully.")


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
        "docs": "/docs",
        "predict": "/predict"
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: PatientFeatures):
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run training first."
        )

    try:
        feature_order = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        import pandas as pd
        input_df = pd.DataFrame([features.dict()])[feature_order]
        X_processed = pipeline.transform(input_df)
        prediction = int(model.predict(X_processed)[0])
        probabilities = model.predict_proba(X_processed)[0]
        confidence = float(round(max(probabilities), 4))

        logger.info(
            f"Prediction: {prediction} | "
            f"Confidence: {confidence:.4f} | "
            f"Age: {features.age} | Sex: {features.sex}"
        )

        return PredictionResponse(
            prediction=prediction,
            prediction_label="Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            confidence=confidence,
            probability_no_disease=float(round(probabilities[0], 4)),
            probability_disease=float(round(probabilities[1], 4)),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
