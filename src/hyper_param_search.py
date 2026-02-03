import os

import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import logger


def find_xgb_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict:
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

        logger.info(f"Testing parameters: {param}")
        clf = GradientBoostingClassifier(**param, random_state=42, verbose=verbose)
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
        study_name="xgb_optimization",
        direction="maximize",
        storage=storage_name,
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)

    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params


def find_xgb_hyperparams_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    verbose: bool = True,
) -> dict:
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # --- CONFIGURATION GPU ---
            "tree_method": "hist",  # Méthode rapide basée sur les histogrammes
            "device": "cuda",
        }

        logger.info(f"Testing parameters: {param}")
        clf = xgb.XGBClassifier(**param, random_state=42, verbose=verbose)
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

    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params


def main():
    pass


if __name__ == "__main__":
    main()
