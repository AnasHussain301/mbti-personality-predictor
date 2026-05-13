"""
Small EDA report used by main.py.
"""

from __future__ import annotations

import os

from config import OUTPUTS_DIR

os.environ.setdefault("MPLCONFIGDIR", os.path.join(str(OUTPUTS_DIR), ".matplotlib"))

import matplotlib.pyplot as plt
import seaborn as sns


def run_full_eda(df):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    counts = df["type"].value_counts().sort_index()
    print("   Class counts:")
    print(counts.to_string())

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#4C78A8")
    plt.title("MBTI Type Distribution")
    plt.xlabel("Type")
    plt.ylabel("Rows")
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "type_distribution.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"   Saved plot: {path}")
