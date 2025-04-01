import sys
import mlflow
import mlflow.xgboost
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, ParameterGrid
import xgboost as xgb

def load_data():
    # Chargez vos données depuis le fichier TSV (ajustez le chemin si nécessaire)
    df = pd.read_csv("DSA-2025_clean_data.tsv", sep="\t")
    return df

def train_model(remote_server_uri, experiment_name, run_name):
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    
    df = load_data()
    # On suppose que la colonne cible s'appelle "readmission"
    target = "readmission"
    X = df.drop(columns=[target])
    y = df[target]
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Définition de la grille de recherche sur deux hyperparamètres
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    grid = list(ParameterGrid(param_grid))
    results = []
    
    for params in grid:
        sub_run_name = f"{run_name}_lr{params['learning_rate']}_depth{params['max_depth']}"
        with mlflow.start_run(run_name=sub_run_name) as run:
            mlflow.log_params(params)
            
            # Entraînement du modèle XGBoost
            model = xgb.XGBClassifier(
                **params,
                n_estimators=100,  # valeur fixe ici, ajustable si besoin
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            # Prédictions et calcul des métriques
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except Exception:
                auc = 0.0
            f1 = f1_score(y_test, preds, average="weighted")
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1_score", f1)
            
            # Log du modèle via mlflow.xgboost (sans mlflow.autolog)
            mlflow.xgboost.log_model(model, artifact_path="xgb_model")
            
            results.append({
                "params": params,
                "accuracy": acc,
                "auc": auc,
                "f1_score": f1,
                "run_id": run.info.run_id
            })
    
    # Création d'une visualisation de la grille de recherche (ici, Accuracy en fonction du learning_rate pour chaque max_depth)
    fig, ax = plt.subplots()
    for depth in param_grid['max_depth']:
        subset = [r for r in results if r["params"]["max_depth"] == depth]
        lr_values = [r["params"]["learning_rate"] for r in subset]
        acc_values = [r["accuracy"] for r in subset]
        ax.plot(lr_values, acc_values, marker='o', label=f"max_depth={depth}")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Grid Search Accuracy")
    ax.legend()
    plot_path = "grid_search_accuracy.png"
    fig.savefig(plot_path)
    mlflow.log_artifact(plot_path, artifact_path="visualizations")

if __name__ == "__main__":
    # Les arguments attendus : remote_server_uri, experiment_name, run_name
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    train_model(remote_server_uri, experiment_name, run_name)
