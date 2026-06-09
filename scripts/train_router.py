"""Train the XGBoost router.

Prefers a curated set of short, query-like labeled phrases
(data/router_training.json) since that matches the inference distribution.
Also folds in the labeled documents themselves. Run scripts/seed_data.py first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import load_documents
from src.router.train_xgb import train


def build_examples(docs_dir: str, labels_path: str, router_path: str):
    examples = []

    # short query-like phrases (primary signal)
    if os.path.exists(router_path):
        with open(router_path) as f:
            for row in json.load(f):
                examples.append((row["text"], row["category"]))

    # labeled documents (additional signal)
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
        for doc in load_documents(docs_dir):
            cat = labels.get(doc["doc_id"])
            if cat:
                examples.append((doc["text"], cat))

    return examples


def main(docs_dir: str, labels_path: str, router_path: str):
    examples = build_examples(docs_dir, labels_path, router_path)
    if not examples:
        raise SystemExit("No training examples. Run scripts/seed_data.py first.")
    result = train(examples)
    acc = result["accuracy"]
    acc_str = f"{acc:.3f}" if acc == acc else "n/a (dataset too small to split)"
    print(f"Router trained on {len(examples)} examples. "
          f"Test accuracy={acc_str}, svd_components={result['svd_components']}, "
          f"labels={result['labels']} -> {result['model_path']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/sample_docs")
    p.add_argument("--labels", default="data/labels.json")
    p.add_argument("--router", default="data/router_training.json")
    args = p.parse_args()
    main(args.docs, args.labels, args.router)
