"""XGBoost-based document/query router. Classifies text into a category
(e.g. hr, finance, engineering, legal, general) so retrieval can be biased.
Loads a trained model from disk; if none exists it returns 'general' with low
confidence so the pipeline still works before training.

The fitted transformers are loaded with joblib, and the XGBoost booster is
loaded from its native .ubj file (not unpickled), which avoids the booster
deserialization crash seen with pickle on some platforms.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

import joblib

from config.settings import settings


class RouterClassifier:
    def __init__(self, transformer=None, classifier=None, labels: List[str] | None = None):
        self.transformer = transformer    # sklearn Pipeline: tfidf -> svd -> norm
        self.classifier = classifier      # XGBClassifier (booster loaded natively)
        self.labels = labels or ["general"]

    @classmethod
    def load(cls, path: str) -> "RouterClassifier":
        if not os.path.exists(path):
            return cls()

        bundle = joblib.load(path)
        booster_path = bundle.get("booster_path")
        if not booster_path or not os.path.exists(booster_path):
            return cls()

        # Rebuild the classifier from the native booster file (no unpickling
        # of the booster object).
        from xgboost import XGBClassifier

        clf = XGBClassifier()
        clf.load_model(booster_path)
        return cls(
            transformer=bundle["transformer"],
            classifier=clf,
            labels=bundle["labels"],
        )

    def predict(self, text: str) -> Tuple[str, float]:
        if self.transformer is None or self.classifier is None:
            return "general", 0.0
        features = self.transformer.transform([text])
        proba = self.classifier.predict_proba(features)[0]
        idx = int(proba.argmax())
        return self.labels[idx], float(proba[idx])


@lru_cache(maxsize=1)
def get_router() -> RouterClassifier:
    return RouterClassifier.load(settings.router_model_path)