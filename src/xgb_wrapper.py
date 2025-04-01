import mlflow.xgboost
import pandas as pd

class XGBWrapper:
    def __init__(self, model_uri):
        self.model = mlflow.xgboost.load_model(model_uri)

    def predict(self, input_data):
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame(input_data)
        preds = self.model.predict(input_data)
        return preds.tolist()

# Exemple d'utilisation
if __name__ == "__main__":
    model_uri = "models:/hospital-readmission-xgb/1"  # Ajustez le nom et la version
    wrapper = XGBWrapper(model_uri)
    example_data = [
        {"chol": 180, "crp": 5.0, "phos": 3.5},
        {"chol": 200, "crp": 10.0, "phos": 4.0}
    ]
    print("Predictions:", wrapper.predict(example_data))
