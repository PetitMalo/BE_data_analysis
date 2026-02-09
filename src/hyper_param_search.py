import json
import os
from datetime import datetime

import numpy as np
import optuna
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from src.utils import logger

DEVICE = "cuda"


def find_xgb_hyperparams_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    verbose: bool = True,
    output_dir: str = "./models",
) -> dict:
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # --- CONFIGURATION GPU ---
            "tree_method": "hist",  # Méthode rapide basée sur les histogrammes
            "device": "cuda",
        }

        logger.info(f"Testing parameters: {param}")
        clf = xgb.XGBClassifier(**param, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        logger.info(f"Validation score: {score}")
        return score

    db_name = "optuna_study.db"
    storage_name = f"sqlite:///{db_name}"

    # On supprime l'ancienne DB si elle existe pour repartir à neuf
    if os.path.exists(db_name):
        os.remove(db_name)

    study = optuna.create_study(
        study_name="xgb_optimization_gpu",
        direction="maximize",
        storage=storage_name,
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    best_params = study.best_params
    best_params.update({"tree_method": "hist", "device": DEVICE})

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(
            output_dir, f"xgb_best_hyperparams_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        ),
        "w",
    ) as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best hyperparameters: {best_params}")
    return best_params


def find_mlp_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    verbose: bool = True,
    output_dir: str = "./models",
) -> dict:
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        param = {}
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_{i}", 1, 100))
        param["hidden_layer_sizes"] = tuple(layers)

        logger.info(f"Testing parameters: {param}")
        mlp_clf = MLPClassifier(**param, random_state=42)
        mlp_clf = mlp_clf.fit(X_train, y_train)
        training_score = mlp_clf.score(X_train, y_train)
        score = mlp_clf.score(X_val, y_val)
        logger.info(f"Validation score: {score} - Training score {training_score}")
        return score

    db_name = "optuna_study.db"
    storage_name = f"sqlite:///{db_name}"

    # On supprime l'ancienne DB si elle existe pour repartir à neuf
    if os.path.exists(db_name):
        os.remove(db_name)

    study = optuna.create_study(
        study_name="mlp_optimization_gpu",
        direction="maximize",
        storage=storage_name,
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    best_params = study.best_params

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"mlp_best_hyperparams_{timestamp}.json"),
        "w",
    ) as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best hyperparameters: {best_params}")
    return best_params


def main():
    pass


if __name__ == "__main__":
    main()
