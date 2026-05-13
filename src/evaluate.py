"""
Evaluation and plotting utilities.
"""

from __future__ import annotations

import os

from config import MBTI_TYPES, OUTPUTS_DIR

os.environ.setdefault("MPLCONFIGDIR", os.path.join(str(OUTPUTS_DIR), ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(y_true, y_pred, model_name: str = "Model"):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(MBTI_TYPES))),
        target_names=MBTI_TYPES,
        output_dict=True,
        zero_division=0,
    )
    print(f"   {model_name}: accuracy={acc:.3f}, macro_f1={macro_f1:.3f}, weighted_f1={weighted_f1:.3f}")
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


def evaluate_dimensions(true_types, pred_types):
    dims = [("E/I", 0), ("N/S", 1), ("T/F", 2), ("J/P", 3)]
    scores = {}
    for name, pos in dims:
        score = sum(t[pos] == p[pos] for t, p in zip(true_types, pred_types)) / len(true_types)
        scores[name] = score
        print(f"   {name}: {score:.3f}")
    return scores


def build_comparison_table(results):
    return pd.DataFrame(
        [
            {
                "model": item["model"],
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                "weighted_f1": item["weighted_f1"],
            }
            for item in results
        ]
    ).sort_values("weighted_f1", ascending=False)


def _save_current(name):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"   Saved plot: {path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(MBTI_TYPES))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=MBTI_TYPES, yticklabels=MBTI_TYPES, cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_current("confusion_matrix_mlp.png")


def plot_f1_per_class(report, model_name: str = "Model"):
    rows = [
        {"type": label, "f1": report.get(label, {}).get("f1-score", 0)}
        for label in MBTI_TYPES
    ]
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="type", y="f1", color="#4C78A8")
    plt.ylim(0, 1)
    plt.title(f"{model_name} F1 by Class")
    _save_current("f1_per_class_mlp.png")


def plot_mlp_loss_curve(mlp):
    if not getattr(mlp, "loss_curve_", None):
        return
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_)
    plt.title("MLP Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    _save_current("mlp_loss_curve.png")


def plot_model_comparison(cmp_df):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=cmp_df, x="model", y="weighted_f1", color="#59A14F")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.title("Model Comparison")
    _save_current("model_comparison.png")
