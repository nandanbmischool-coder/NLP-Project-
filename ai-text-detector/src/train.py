import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import StyleFeatures, build_pipeline, clean_text

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "dataset.csv"
MODEL = BASE / "models" / "detector.joblib"
METRICS = BASE / "models" / "metrics.json"
PREDICTIONS = BASE / "models" / "evaluation_predictions.json"
ROC = BASE / "models" / "roc_curve.json"
MAX_PER_CLASS = 100_000


def classification_metrics(labels, predictions, probabilities) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def baseline_models() -> dict:
    classifier = LogisticRegression(
        C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
    )
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "stylometry_only": Pipeline([
            ("features", StyleFeatures()),
            ("scale", StandardScaler()),
            ("clf", classifier),
        ]),
        "word_only_tfidf": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True
            )),
            ("clf", classifier),
        ]),
        "character_only_tfidf": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                max_features=100_000, sublinear_tf=True
            )),
            ("clf", classifier),
        ]),
    }


def main() -> None:
    df = pd.read_csv(DATA).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "group" not in df.columns:
        df["group"] = np.arange(len(df))
    df["group"] = df["group"].astype(str)
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label must be 0 (human) or 1 (AI)")
    available = df["label"].value_counts()
    sample_size = int(min(MAX_PER_CLASS, available.min()))
    if len(df) > sample_size * 2:
        df = pd.concat([
            df[df["label"] == label].sample(sample_size, random_state=42)
            for label in (0, 1)
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Using a reproducible balanced training subset: {sample_size:,} per class")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(df, df["label"], groups=df["group"]))
    train, test = df.iloc[train_indices], df.iloc[test_indices]
    print(f"Training on {len(train)} texts, testing on {len(test)} unseen texts...")

    model = build_pipeline()
    model.fit(train["text"], train["label"], clf__sample_weight=train["weight"].to_numpy())
    predictions = model.predict(test["text"])
    probabilities = model.predict_proba(test["text"])[:, 1]
    metrics = {
        **classification_metrics(test["label"], predictions, probabilities),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "dataset_samples": int(len(df)),
    }
    baseline_results = {}
    for name, baseline in baseline_models().items():
        baseline.fit(train["text"], train["label"])
        baseline_predictions = baseline.predict(test["text"])
        baseline_probabilities = baseline.predict_proba(test["text"])[:, 1]
        baseline_results[name] = classification_metrics(
            test["label"], baseline_predictions, baseline_probabilities
        )
        print(f"{name:>24}: accuracy={baseline_results[name]['accuracy']:.3f}, "
              f"f1={baseline_results[name]['f1']:.3f}, auc={baseline_results[name]['auc']:.3f}")
    metrics["baselines"] = baseline_results
    fpr, tpr, thresholds = roc_curve(test["label"], probabilities)
    PREDICTIONS.write_text(json.dumps({
        "labels": test["label"].astype(int).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
    }), encoding="utf-8")
    ROC.write_text(json.dumps({
        "fpr": fpr.astype(float).tolist(),
        "tpr": tpr.astype(float).tolist(),
        "thresholds": thresholds.astype(float).tolist(),
        "auc": metrics["auc"],
    }), encoding="utf-8")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for key in ("accuracy", "precision", "recall", "f1", "auc", "false_positive_rate"):
        print(f"{key:>20}: {metrics[key]:.3f}")
    print("Saved:", MODEL)
    print("Saved:", PREDICTIONS)
    print("Saved:", ROC)


if __name__ == "__main__":
    main()
