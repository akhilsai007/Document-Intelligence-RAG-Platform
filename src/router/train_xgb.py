"""Train the XGBoost router and log to MLflow.

Features: TF-IDF -> TruncatedSVD (LSA) -> Normalizer, then XGBoost on the
dense components. Reducing the sparse TF-IDF matrix to dense latent features
plays to XGBoost's strength and is far more stable than feeding raw sparse
vectors to boosted trees. SVD dimensionality adapts to dataset size.

Persistence: the fitted transformers are saved with joblib, but the XGBoost
booster is saved in its native binary-JSON (.ubj) format via save_model().
This avoids pickling the booster, which can crash on reload (notably on macOS
/ Apple Silicon) and is the format XGBoost officially recommends.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from xgboost import XGBClassifier

from config.settings import settings


def _build_transformer(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ("norm", Normalizer()),
        ]
    )


def _build_classifier(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=2,
    )


def train(examples: List[Tuple[str, str]], model_path: str | None = None) -> dict:
    model_path = model_path or settings.router_model_path
    texts = [t for t, _ in examples]
    raw_labels = [c for _, c in examples]

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    labels = list(le.classes_)
    n_classes = len(labels)

    # A stratified split needs at least one test sample per class. On small
    # corpora that isn't possible, so fall back to training on everything.
    test_count = int(round(0.25 * len(texts)))
    can_split = (
        len(texts) > n_classes
        and test_count >= n_classes
        and int(np.bincount(y).min()) >= 2
    )
    if can_split:
        x_tr, x_te, y_tr, y_te = train_test_split(
            texts, y, test_size=test_count, random_state=42, stratify=y
        )
    else:
        x_tr, y_tr = texts, y
        x_te, y_te = [], []

    # Safe SVD dimensionality: strictly less than vocab size and sample count.
    vocab_probe = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    n_features = vocab_probe.fit_transform(x_tr).shape[1]
    n_components = max(2, min(40, n_features - 1, len(x_tr) - 1))

    transformer = _build_transformer(n_components)
    x_tr_dense = transformer.fit_transform(x_tr)

    clf = _build_classifier(n_classes)
    clf.fit(x_tr_dense, y_tr)

    if len(x_te):
        acc = float(clf.score(transformer.transform(x_te), y_te))
    else:
        acc = float("nan")

    # ---- persist: transformers via joblib, booster via native format ----
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    booster_path = os.path.splitext(model_path)[0] + ".ubj"
    clf.save_model(booster_path)
    joblib.dump(
        {
            "transformer": transformer,
            "labels": labels,
            "booster_path": booster_path,
            "n_classes": n_classes,
        },
        model_path,
    )

    # best-effort MLflow logging
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("router")
        with mlflow.start_run():
            mlflow.log_param("svd_components", n_components)
            mlflow.log_param("n_classes", n_classes)
            mlflow.log_param("n_train", len(x_tr))
            mlflow.log_metric("test_accuracy", acc)
    except Exception:
        pass

    return {"accuracy": acc, "labels": labels, "model_path": model_path,
            "svd_components": n_components}