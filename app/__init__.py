import os
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from .utils import get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    if not model_name or not model_version:
        raise ValueError("Les variables MLFLOW_MODEL_NAME et MLFLOW_MODEL_VERSION doivent être définies")
    # Chargement du modèle depuis MLflow
    model = get_model(model_name, model_version)
    yield

# Définition d'un modèle Pydantic pour l'entrée (si besoin d'un POST)
class InputData(BaseModel):
    chol: float
    crp: float
    phos: float

app = FastAPI(
    lifespan=lifespan,
    title="Hospital Readmission Predictor",
    description="API pour prédire la réadmission hospitalière via un modèle XGBoost enregistré avec MLflow",
    version="1.0.0"
)

@app.get("/", tags=["Welcome"])
def read_root():
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "message": "Hospital Readmission Predictor",
        "model_name": model_name,
        "model_version": model_version
    }

@app.get("/predict", tags=["Predict"])
def predict(chol: float, crp: float, phos: float) -> Dict:
    try:
        input_df = pd.DataFrame([{"chol": chol, "crp": crp, "phos": phos}])
        preds = model.predict(input_df)
        return {"prediction": preds[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")
