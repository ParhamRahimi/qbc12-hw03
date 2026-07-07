import os
from contextlib import asynccontextmanager
from typing import List

import mlflow
from fastapi import FastAPI

from airbnb_serving.schema import ListingFeatures, PredictionResponse
from airbnb_serving.predictor import predict_single, predict_batch


MODEL_RUN_ID: str = ""
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global model, MODEL_RUN_ID

    MODEL_RUN_ID = os.environ["MODEL_RUN_ID"]
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    username = os.environ["MLFLOW_TRACKING_USERNAME"]
    password = os.environ["MLFLOW_TRACKING_PASSWORD"]

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")
    print(f"Model loaded from MLflow run {MODEL_RUN_ID}")

    yield


app = FastAPI(
    title="Airbnb Serving API",
    description="Serves the HW02 model for predicting high-demand listings.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": MODEL_RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ListingFeatures):
    return predict_single(features, model, MODEL_RUN_ID)


@app.post("/predict/batch", response_model=List[PredictionResponse])
def batch_predict(features_list: List[ListingFeatures]):
    return predict_batch(features_list, model, MODEL_RUN_ID)