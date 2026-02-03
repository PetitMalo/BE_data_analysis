import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import logger


def find_xgb_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
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

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params


def main():
    pass


if __name__ == "__main__":
    main()
