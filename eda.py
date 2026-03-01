"""
Exploratory Data Analysis — Loan Default Dataset
=================================================
Author: Oluwadamilola Oyekan, Ph.D.

Run this script first to understand the data before model training.
Generates a full EDA report with charts saved to outputs/eda/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_PATH  = "Task_3_and_4_Loan_Data.csv"
TARGET_COL = "default"
EDA_DIR    = Path("outputs/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head())
print(df.describe())

numeric_cols     = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COL, errors="ignore").tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()


# ---------------------------------------------------------------------------
# 2. Target distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
counts = df[TARGET_COL].value_counts()
ax.bar(["No Default (0)", "Default (1)"], counts.values, color=["#2ecc71", "#e74c3c"])
ax.set_ylabel("Count")
ax.set_title(f"Target Distribution (Default Rate: {df[TARGET_COL].mean():.2%})")
for i, v in enumerate(counts.values):
    ax.text(i, v + len(df) * 0.005, str(v), ha="center", fontweight="bold")
plt.tight_layout()
fig.savefig(EDA_DIR / "target_distribution.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 3. Missing values heatmap
# ---------------------------------------------------------------------------
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) > 0:
    fig, ax = plt.subplots(figsize=(8, max(3, len(miss) * 0.4)))
    ax.barh(miss.index, miss.values / len(df) * 100, color="salmon")
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Values by Column")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "missing_values.png", dpi=150)
    plt.close()
else:
    print("No missing values found.")


# ---------------------------------------------------------------------------
# 4. Numeric feature distributions (split by default status)
# ---------------------------------------------------------------------------
n = len(numeric_cols)
if n > 0:
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        for label, grp in df.groupby(TARGET_COL)[col]:
            axes[i].hist(grp.dropna(), bins=40, alpha=0.6,
                         label=f"Default={label}", density=True)
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric Features by Default Status", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "numeric_distributions.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------
if n > 1:
    corr = df[numeric_cols + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(corr, annot=(n <= 15), fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 6. Box plots — numeric features vs default
# ---------------------------------------------------------------------------
if n > 0:
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        df.boxplot(column=col, by=TARGET_COL, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel(f"Default (0/1)")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("")
    fig.suptitle("Box Plots: Numeric Features vs Default", fontsize=13)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "boxplots.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 7. Categorical feature default rates
# ---------------------------------------------------------------------------
for col in categorical_cols[:6]:   # limit to first 6
    rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, len(rate) * 0.6), 4))
    rate.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Default Rate")
    ax.set_title(f"Default Rate by {col}")
    ax.axhline(df[TARGET_COL].mean(), color="red", linestyle="--", label="Overall mean")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(EDA_DIR / f"default_rate_{col}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 8. Summary stats by default status
# ---------------------------------------------------------------------------
summary = df.groupby(TARGET_COL)[numeric_cols].mean().T
summary.columns = ["Non-Default (0)", "Default (1)"]
summary["Ratio (D/ND)"] = summary["Default (1)"] / summary["Non-Default (0)"].replace(0, np.nan)
print("\nMean Feature Values by Default Status:\n")
print(summary.round(3).to_string())
summary.to_csv(EDA_DIR / "feature_means_by_default.csv")

print(f"\n[EDA COMPLETE] All charts saved to: {EDA_DIR}/")
