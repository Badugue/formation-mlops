import mlflow
import pandas as pd

model_name = "exam"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

# Chargez le fichier TSV
df = pd.read_csv("DSA-2025_clean_data.tsv", sep="\t")
target = "readmission"
input_data = df.drop(columns=[target]).head(2)

print("Shape de l'entrée:", input_data.shape)
print("Colonnes de l'entrée:", input_data.columns.tolist())

results = model.predict(input_data)
print("Prédictions :", results)
