import json
import os
from datetime import datetime

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils import logger

DEVICE = "cuda"
SEED = 42


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
        clf = xgb.XGBClassifier(**param, random_state=SEED)
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
        mlp_clf = MLPClassifier(**param, random_state=SEED)
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
    best_params = {}
    layers = []
    for i in range(study.best_params.get("n_layers")):
        layers.append(study.best_params.get(f"n_units_{i}"))
    best_params["hidden_layer_sizes"] = tuple(layers)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"mlp_best_hyperparams_{timestamp}.json"),
        "w",
    ) as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best hyperparameters: {best_params}")
    return best_params


def find_dtc_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    verbose: bool = True,
    output_dir: str = "./models",
    metric: str = "f1",
) -> dict:
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 256)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=SEED,
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        if metric == "f1":
            scorer = make_scorer(f1_score, average="weighted")
        else:
            scorer = "accuracy"

        scores = cross_val_score(clf, X, y, cv=skf, scoring=scorer, n_jobs=-1)
        final_score = scores.mean()

        if verbose:
            logger.info(
                f"Trial {trial.number} - CV {metric.upper()}: {final_score:.4f} "
                f"(+/- {scores.std():.4f})"
            )

        return final_score

    db_name = "optuna_study.db"
    storage_name = f"sqlite:///{db_name}"

    # On supprime l'ancienne DB si elle existe pour repartir à neuf
    if os.path.exists(db_name):
        os.remove(db_name)

    study = optuna.create_study(
        study_name="dtc_optimization_gpu",
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
        os.path.join(output_dir, f"dtc_best_hyperparams_{timestamp}.json"),
        "w",
    ) as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best hyperparameters: {best_params}")
    return best_params


def find_svm_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    verbose: bool = True,
    output_dir: str = "./models",
    metric: str = "f1",
) -> dict:
    def objective(trial):
        svc_C = trial.suggest_float("C", 0.1, 1000, log=True)
        svc_gamma = trial.suggest_float("gamma", 0.0001, 1, log=True)
        svc_kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])

        svc = SVC(C=svc_C, gamma=svc_gamma, kernel=svc_kernel, random_state=SEED)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        if metric == "f1":
            scorer = make_scorer(f1_score, average="weighted")
        else:
            scorer = "accuracy"
        scores = cross_val_score(svc, X, y, cv=skf, scoring=scorer, n_jobs=-1)
        final_score = scores.mean()

        logger.info(f"Trial {trial.number} - CV {metric.upper()} Mean: {final_score:.4f}")
        return final_score

    db_name = "optuna_study.db"
    storage_name = f"sqlite:///{db_name}"

    if os.path.exists(db_name):
        os.remove(db_name)

    study = optuna.create_study(
        study_name="svm_optimization_gpu",
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
        os.path.join(output_dir, f"svm_best_hyperparams_{timestamp}.json"),
        "w",
    ) as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best hyperparameters: {best_params}")
    return best_params


def main():
    pass


if __name__ == "__main__":
    main()
