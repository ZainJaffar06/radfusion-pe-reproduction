"""EHR estimators: elastic-net regularized logistic regression (primary) and a support-vector
classifier (alternative). Both use class weighting to address label imbalance.

Table 1 discrimination uses the elastic-net logistic model; the Table 2 operating point uses the
support-vector variant. They are deliberately two different estimators.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from . import config as C


@dataclass
class FittedModel:
    name: str
    estimator: Pipeline

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


def fit_elastic_net(X_train, y_train) -> FittedModel:
    """Elastic-net logistic regression with standardization, tuned by ROC-AUC cross-validation."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=5000, random_state=C.SEED)),
    ])
    grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0.1, 0.5, 0.9],
    }
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return FittedModel("EHR (elastic-net logistic)", search.best_estimator_)


def fit_svc(X_train, y_train) -> FittedModel:
    """Support-vector classifier (probability-calibrated) as an alternative EHR estimator."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced", random_state=C.SEED)),
    ])
    grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto"],
    }
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return FittedModel("EHR (SVC)", search.best_estimator_)
