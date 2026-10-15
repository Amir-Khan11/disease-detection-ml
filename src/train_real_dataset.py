"""Train and compare models on a real symptom-disease CSV.

Expected format: one row per case, binary (0/1) symptom columns, and one label column.
Example (Kaggle "Disease Symptom Prediction"): python src/train_real_dataset.py \
    --data data/raw/Training.csv --target prognosis
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV")
    ap.add_argument("--target", default="disease", help="Label column name")
    ap.add_argument("--out", default="reports", help="Output folder")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="Keep identical rows (default: drop them to avoid train/test leakage)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    df = pd.read_csv(args.data)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if not args.keep_duplicates:
        before = len(df); df = df.drop_duplicates(); print(f"Dropped {before - len(df)} duplicate rows")
    X, y = df.drop(columns=[args.target]), df[args.target]
    print(f"Rows: {len(df)} | Symptoms: {X.shape[1]} | Diseases: {y.nunique()}")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Naive Bayes": BernoulliNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    min_class = y.value_counts().min()
    cv = StratifiedKFold(n_splits=max(2, min(5, int(min_class))), shuffle=True, random_state=42)

    rows = []
    for name, model in models.items():
        s = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({"model": name,
                     "accuracy_mean": s["test_accuracy"].mean(), "accuracy_std": s["test_accuracy"].std(),
                     "f1_macro_mean": s["test_f1_macro"].mean()})
    results = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    print(results.round(4).to_string(index=False))
    results.to_csv(out / "model_comparison.csv", index=False)

    best_name = results.iloc[0]["model"]
    test_size = max(0.2, (y.nunique() + 1) / len(df))   # keep at least one sample per class in test
    stratify = y if min_class >= 2 and test_size < 0.5 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=42)
    best = models[best_name].fit(X_tr, y_tr)
    pred = best.predict(X_te)
    (out / "classification_report.txt").write_text(classification_report(y_te, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout(); plt.savefig(out / "real_confusion_matrix.png", dpi=150); plt.close()

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns).nlargest(15)[::-1]
        imp.plot.barh(figsize=(7, 6), title=f"Top symptoms ({best_name})")
        plt.tight_layout(); plt.savefig(out / "feature_importance.png", dpi=150); plt.close()
    print(f"Best model: {best_name} | held-out macro-F1: {f1_score(y_te, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()
