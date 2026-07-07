import os
from contextlib import asynccontextmanager
from typing import List

import mlflow
from fastapi import FastAPI, HTTPException

from airbnb_serving.schema import ListingFeatures, PredictionResponse
from airbnb_serving.predictor import predict_single, predict_batch


MODEL_RUN_ID: str = ""
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, with local fallback."""
    global model, MODEL_RUN_ID

    MODEL_RUN_ID = os.environ.get("MODEL_RUN_ID", "")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://185.50.38.163:33014")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    loaded = False

    # Try remote MLflow first
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        mlflow.set_tracking_uri(tracking_uri)
        try:
            model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")
            loaded = True
            print(f"Model loaded from remote MLflow run {MODEL_RUN_ID}")
        except Exception as e:
            print(f"Remote MLflow failed: {e}")

    # Fall back to local SQLite
    if not loaded:
        import os as _os
        local_db = "artifacts/mlflow.db"
        if _os.path.exists(local_db):
            mlflow.set_tracking_uri(f"sqlite:///{local_db}")
            try:
                if MODEL_RUN_ID:
                    model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")
                else:
                    runs = mlflow.search_runs(
                        order_by=["metrics.f1 DESC"],
                        max_results=1,
                    )
                    if len(runs) > 0:
                        MODEL_RUN_ID = runs.iloc[0]["run_id"]
                        model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")
                loaded = True
                print(f"Model loaded from local DB, run {MODEL_RUN_ID}")
            except Exception as e:
                print(f"Local MLflow fallback failed: {e}")

    if not loaded:
        print("WARNING: Model not loaded. Endpoints will return errors.")

    yield


app = FastAPI(
    title="Airbnb Serving API",
    description="Serves the HW02 model for predicting high-demand listings.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if model is None:
        return {"status": "model_not_loaded", "model_run_id": MODEL_RUN_ID}
    return {"status": "ok", "model_run_id": MODEL_RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ListingFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(features, model, MODEL_RUN_ID)


@app.post("/predict/batch", response_model=List[PredictionResponse])
def batch_predict(features_list: List[ListingFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_batch(features_list, model, MODEL_RUN_ID)
