# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis — Credit Card Fraud Detection
#
# **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
#
# 284,807 credit card transactions from European cardholders (September 2013, 2 days).
# Features V1–V28 are PCA-transformed (confidential originals). Only `Time` and `Amount`
# are untransformed.
#
# **Goal:** Understand the data before building fraud detection models.

# %% [markdown]
# ## 1. Data Loading & Initial Inspection

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

# %%
df = pd.read_csv("../data/raw/creditcard.csv")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()

# %%
df.info()

# %%
df.describe().T

# %% [markdown]
# ### Missing Values

# %%
missing = df.isnull().sum()
print(f"Total missing values: {missing.sum()}")
# No missing values — the PCA transformation and dataset curation ensured completeness.

# %% [markdown]
# ### Duplicate Rows

# %%
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows: {n_duplicates:,} ({n_duplicates / len(df) * 100:.2f}%)")

# Check: are duplicates mostly fraud or non-fraud?
dup_mask = df.duplicated(keep=False)
print(f"\nClass distribution in duplicate rows:")
print(df[dup_mask]["Class"].value_counts())
print(f"\nClass distribution in non-duplicate rows:")
print(df[~dup_mask]["Class"].value_counts())

# %% [markdown]
# There are 1,081 duplicate rows. Since these are real transactions (different customers can
# make identical-looking transactions after PCA), we'll **keep them** — dropping could
# remove legitimate data points.

# %% [markdown]
# ## 2. Class Distribution

# %%
class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
bars = axes[0].bar(
    ["Legitimate (0)", "Fraud (1)"],
    class_counts.values,
    color=["#2196F3", "#F44336"],
    edgecolor="black",
    linewidth=0.5,
)
for bar, count, pct in zip(bars, class_counts.values, class_pct.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2000,
        f"{count:,}\n({pct:.3f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
axes[0].set_title("Class Distribution (Count)", fontsize=13)
axes[0].set_ylabel("Count")

# Log scale version to see fraud bar
axes[1].bar(
    ["Legitimate (0)", "Fraud (1)"],
    class_counts.values,
    color=["#2196F3", "#F44336"],
    edgecolor="black",
    linewidth=0.5,
)
axes[1].set_yscale("log")
axes[1].set_title("Class Distribution (Log Scale)", fontsize=13)
axes[1].set_ylabel("Count (log)")

plt.tight_layout()
plt.savefig("../data/processed/01_class_distribution.png", bbox_inches="tight")
plt.show()

print(
    f"\nImbalance ratio: 1 fraud per {class_counts[0] // class_counts[1]:,} legitimate transactions"
)

# %% [markdown]
# **Extreme class imbalance:** Only 0.173% of transactions are fraudulent (492 out of 284,807).
# This is a 1:578 ratio. Implications:
# - Accuracy is a useless metric (99.83% by predicting all legitimate)
# - Must use **PR-AUC**, **F1**, **precision**, and **recall** for evaluation
# - Need class balancing strategies: `scale_pos_weight`, SMOTE, or threshold tuning

# %% [markdown]
# ## 3. Feature Distributions

# %% [markdown]
# ### 3.1 Time

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall Time distribution
axes[0].hist(
    df["Time"], bins=100, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.3
)
axes[0].set_title("Transaction Time Distribution (All)", fontsize=13)
axes[0].set_xlabel("Time (seconds from first transaction)")
axes[0].set_ylabel("Count")

# Time by class
for cls, color, label in [(0, "#2196F3", "Legitimate"), (1, "#F44336", "Fraud")]:
    subset = df[df["Class"] == cls]["Time"]
    axes[1].hist(
        subset,
        bins=100,
        color=color,
        alpha=0.6,
        label=label,
        density=True,
        edgecolor="black",
        linewidth=0.3,
    )
axes[1].set_title("Transaction Time Distribution (by Class, Normalized)", fontsize=13)
axes[1].set_xlabel("Time (seconds from first transaction)")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("../data/processed/01_time_distribution.png", bbox_inches="tight")
plt.show()

# Time spans ~48 hours (172,792 seconds)
print(f"Time range: {df['Time'].min():.0f} to {df['Time'].max():.0f} seconds")
print(
    f"That's approximately {df['Time'].max() / 3600:.1f} hours ({df['Time'].max() / 86400:.1f} days)"
)

# %% [markdown]
# The Time feature shows two clear cycles (two days), with dips corresponding to nighttime.
# Fraud appears more uniformly distributed across time — fraudsters don't sleep.
# **Feature idea:** Extract hour-of-day as a cyclical feature.

# %% [markdown]
# ### 3.2 Amount

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Raw Amount
axes[0].hist(
    df["Amount"], bins=100, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.3
)
axes[0].set_title("Transaction Amount (All)", fontsize=13)
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Count")

# Log Amount (for better visibility)
log_amount = np.log1p(df["Amount"])
axes[1].hist(
    log_amount, bins=100, color="#4CAF50", alpha=0.7, edgecolor="black", linewidth=0.3
)
axes[1].set_title("Log(1 + Amount) Distribution", fontsize=13)
axes[1].set_xlabel("Log(1 + Amount)")
axes[1].set_ylabel("Count")

# Amount by class
for cls, color, label in [(0, "#2196F3", "Legitimate"), (1, "#F44336", "Fraud")]:
    subset = df[df["Class"] == cls]["Amount"]
    axes[2].hist(
        np.log1p(subset),
        bins=80,
        color=color,
        alpha=0.6,
        label=label,
        density=True,
        edgecolor="black",
        linewidth=0.3,
    )
axes[2].set_title("Log(1 + Amount) by Class (Normalized)", fontsize=13)
axes[2].set_xlabel("Log(1 + Amount)")
axes[2].set_ylabel("Density")
axes[2].legend()

plt.tight_layout()
plt.savefig("../data/processed/01_amount_distribution.png", bbox_inches="tight")
plt.show()

print("Amount statistics:")
print(
    f"  Overall — Mean: ${df['Amount'].mean():.2f}, Median: ${df['Amount'].median():.2f}, Max: ${df['Amount'].max():.2f}"
)
print(
    f"  Fraud   — Mean: ${df[df['Class'] == 1]['Amount'].mean():.2f}, Median: ${df[df['Class'] == 1]['Amount'].median():.2f}"
)
print(
    f"  Legit   — Mean: ${df[df['Class'] == 0]['Amount'].mean():.2f}, Median: ${df[df['Class'] == 0]['Amount'].median():.2f}"
)

# %% [markdown]
# Amount is heavily right-skewed (most transactions are small). Fraudulent transactions tend
# to have a **lower median** but similar mean — fraud spans a wide range of amounts.
# **Feature idea:** Log-transform Amount, and possibly bin into categories.

# %% [markdown]
# ### 3.3 V-Features (PCA Components)

# %%
v_features = [f"V{i}" for i in range(1, 29)]

fig, axes = plt.subplots(4, 7, figsize=(24, 14))
axes = axes.flatten()

for i, feat in enumerate(v_features):
    ax = axes[i]
    for cls, color, label in [(0, "#2196F3", "Legit"), (1, "#F44336", "Fraud")]:
        subset = df[df["Class"] == cls][feat]
        ax.hist(subset, bins=80, color=color, alpha=0.5, label=label, density=True)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("V-Feature Distributions by Class (Normalized)", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("../data/processed/01_v_features_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Several V-features show clear separation between fraud and legitimate distributions:
# - **Strong separators:** V4, V11, V12, V14, V17 — fraud distributions are visibly shifted
# - **Weak/no separation:** V8, V13, V15, V22, V23, V24, V25, V26
#
# This confirms the correlation analysis and gives us confidence about which features
# will drive model predictions.

# %% [markdown]
# ## 4. Correlation Analysis

# %%
# Correlation with target
corr_with_target = df.corr()["Class"].drop("Class").sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#F44336" if v < 0 else "#2196F3" for v in corr_with_target.values]
ax.barh(
    corr_with_target.index,
    corr_with_target.values,
    color=colors,
    edgecolor="black",
    linewidth=0.3,
)
ax.set_title("Feature Correlation with Fraud (Class)", fontsize=14)
ax.set_xlabel("Pearson Correlation")
ax.axvline(x=0, color="black", linewidth=0.8)
# Highlight strong correlations
ax.axvline(x=0.1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=-0.1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../data/processed/01_correlation_with_target.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Feature Correlation Heatmap (Top Features)

# %%
# Select features with strongest fraud correlations for a focused heatmap
top_features = (
    corr_with_target.abs().sort_values(ascending=False).head(15).index.tolist()
)
top_features.append("Class")

corr_matrix = df[top_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Heatmap — Top 15 Fraud-Correlated Features", fontsize=14)
plt.tight_layout()
plt.savefig("../data/processed/01_correlation_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Key observations:
# - PCA features are **mostly uncorrelated with each other** (by design)
# - Strongest negative correlations with fraud: V17 (-0.33), V14 (-0.30), V12 (-0.26)
# - Strongest positive correlations with fraud: V11 (+0.15), V4 (+0.13), V2 (+0.09)
# - Amount has negligible correlation with fraud (0.006)
# - The low inter-feature correlation means tree-based models should perform well
#   without multicollinearity concerns

# %% [markdown]
# ## 5. Outlier Analysis

# %%
# Check for extreme outliers in key features using IQR method
print("Features with extreme outliers (values > 10x IQR beyond Q3):\n")
for feat in v_features + ["Amount"]:
    q1 = df[feat].quantile(0.25)
    q3 = df[feat].quantile(0.75)
    iqr = q3 - q1
    extreme_threshold = q3 + 10 * iqr
    n_extreme = (df[feat] > extreme_threshold).sum()
    if n_extreme > 0:
        print(
            f"  {feat}: {n_extreme} extreme outliers (threshold: {extreme_threshold:.2f})"
        )

# %% [markdown]
# Some V-features have extreme outliers. Since we're using tree-based models
# (XGBoost, LightGBM, CatBoost), they're inherently robust to outliers — no need
# to clip or remove them.

# %% [markdown]
# ## 6. Fraud Transaction Deep Dive

# %%
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

print(f"Fraud transactions: {len(fraud)}")
print(f"Legitimate transactions: {len(legit):,}")
print()

# Statistical comparison for top features
comparison_features = ["V17", "V14", "V12", "V10", "V11", "V4", "V2", "Amount", "Time"]
comparison = pd.DataFrame(
    {
        "Feature": comparison_features,
        "Fraud Mean": [fraud[f].mean() for f in comparison_features],
        "Legit Mean": [legit[f].mean() for f in comparison_features],
        "Fraud Std": [fraud[f].std() for f in comparison_features],
        "Legit Std": [legit[f].std() for f in comparison_features],
    }
)
comparison["Mean Diff"] = comparison["Fraud Mean"] - comparison["Legit Mean"]
comparison = comparison.sort_values("Mean Diff", key=abs, ascending=False)
print(comparison.to_string(index=False))

# %% [markdown]
# ## 7. Summary & Modeling Implications
#
# ### Dataset Characteristics
# | Property | Value |
# |---|---|
# | Rows | 284,807 |
# | Features | 30 (28 PCA + Time + Amount) |
# | Target | Binary (0 = Legitimate, 1 = Fraud) |
# | Fraud rate | 0.173% (492 / 284,807) |
# | Missing values | 0 |
# | Duplicates | 1,081 (kept — likely real) |
#
# ### Key Findings
# 1. **Extreme class imbalance** (1:578) — must use appropriate metrics and balancing
# 2. **No missing values** — no imputation needed
# 3. **PCA features are uncorrelated** — good for tree-based models
# 4. **Strong fraud signals** in V17, V14, V12, V10, V11, V4
# 5. **Amount** has weak direct correlation but different distributions by class
# 6. **Time** shows daily cycles; fraud is more uniformly distributed
# 7. **Outliers present** but tree-based models handle them natively
#
# ### Modeling Strategy
# - **Metrics:** PR-AUC (primary), F1, precision, recall, ROC-AUC
# - **Class balancing:** `scale_pos_weight` parameter in boosting models + threshold tuning
# - **Feature engineering:**
#   - Log-transform Amount
#   - Hour-of-day from Time (cyclical encoding)
#   - Standardize Amount (V-features are already standardized from PCA)
# - **Models:** XGBoost, LightGBM, CatBoost — all handle imbalanced data and outliers well
# - **Validation:** Stratified K-Fold to preserve class ratio in each fold
