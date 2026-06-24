"""Model training, tuning and evaluation (RF & SVM).

Paper protocol:
  * 80/20 train/test split
  * 3-fold cross-validation for hyper-parameter tuning on training data
  * RF tuned over n_estimators 100..1000
  * SVM (RBF) tuned over C and gamma
  * Cohen's Kappa on the test set is the headline metric
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class FitResult:
    name: str
    best_params: dict
    accuracy: float
    kappa: float
    estimator: object


def tune_random_forest(X, y, n_estimators_grid, cv, random_state, scoring="accuracy"):
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_grid={"n_estimators": n_estimators_grid},
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def tune_svm(X, y, C_grid, gamma_grid, kernel, cv, scoring="accuracy"):
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel=kernel))])
    grid = GridSearchCV(
        pipe,
        param_grid={"svc__C": C_grid, "svc__gamma": gamma_grid},
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def evaluate(estimator, X_test, y_test) -> tuple[float, float]:
    pred = estimator.predict(X_test)
    return accuracy_score(y_test, pred), cohen_kappa_score(y_test, pred)


def fit_and_evaluate_both(
    X_train, y_train, X_test, y_test, model_cfg
) -> list[FitResult]:
    cv = model_cfg.get("cv_folds", 3)
    rs = model_cfg.get("random_state", 42)
    results: list[FitResult] = []

    rf_est, rf_params = tune_random_forest(
        X_train, y_train,
        model_cfg["random_forest"]["n_estimators_grid"], cv, rs,
    )
    acc, kappa = evaluate(rf_est, X_test, y_test)
    results.append(FitResult("RF", rf_params, acc, kappa, rf_est))

    svm_cfg = model_cfg["svm"]
    svm_est, svm_params = tune_svm(
        X_train, y_train,
        svm_cfg["C_grid"], svm_cfg["gamma_grid"], svm_cfg.get("kernel", "rbf"), cv,
    )
    acc, kappa = evaluate(svm_est, X_test, y_test)
    results.append(FitResult("SVM", svm_params, acc, kappa, svm_est))

    return results
