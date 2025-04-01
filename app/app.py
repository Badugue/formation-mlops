from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd

app = FastAPI(
    title="API de Prédiction - Réadmission Hospitalière",
    description="API pour servir le modèle XGBoost enregistré via MLflow pour prédire la réadmission hospitalière.",
    version="1.0.0"
)

# Chargez le modèle enregistré depuis MLflow (ajustez le nom et la version si nécessaire)
model_uri = "models:/hospital-readmission-xgb/latest"
try:
    model = mlflow.xgboost.load_model(model_uri)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle MLflow: {e}")

class PredictionRequest(BaseModel):
    # 'data' doit être une liste de dictionnaires, chaque dictionnaire représentant une observation
    data: list

@app.post("/predict", summary="Effectue une prédiction sur les données d'entrée")
def predict(request: PredictionRequest):
    try:
        # Convertir la liste de dictionnaires en DataFrame
        input_df = pd.DataFrame(request.data)
        preds = model.predict(input_df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {e}")

# Pour lancer l'application, utilisez la commande suivante depuis le terminal :
# uvicorn src.app:app --host 0.0.0.0 --port 8000
