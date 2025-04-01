import mlflow.xgboost

def get_model(model_name: str, model_version: str):
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.xgboost.load_model(model_uri)
