"""
Data preparation and fixed evaluation harness for aora.

One-time setup: generates eda_report.md from data in data/ directory.
Runtime utilities: auto-detects task type, provides fixed train/test split
and evaluation metrics that agents CANNOT modify.

Usage:
    uv run prepare.py            # run EDA and print dataset summary
    uv run prepare.py --no-eda   # skip writing eda_report.md
"""

import os
import sys
import math
import time
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (fixed -- do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600        # seconds per experiment (5 minutes)
TEST_SIZE = 0.2          # fixed 80/20 train/test split
RANDOM_STATE = 42        # global reproducibility seed
DATA_DIR = "data"        # directory where the user places their data file
MAX_CARDINALITY = 50     # categorical features with more unique values get dropped
MOSTLY_MISSING_THRESHOLD = 0.6  # features with >60% missing values are flagged

# Target column name candidates (checked in order, case-insensitive)
TARGET_CANDIDATES = ['Target']

# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def _find_data_file():
    """Find first CSV or Parquet file in DATA_DIR. Raises if none found."""
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Data directory '{DATA_DIR}' not found. "
            "Create it and place your CSV or Parquet file inside."
        )
    candidates = []
    for fname in sorted(os.listdir(DATA_DIR)):
        lower = fname.lower()
        if lower.endswith((".csv", ".parquet", ".tsv")) and not fname.startswith("."):
            candidates.append(os.path.join(DATA_DIR, fname))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV or Parquet files found in '{DATA_DIR}'. "
            "Place your data file there and re-run."
        )
    return candidates[0]


def load_data():
    """
    Load the first CSV/Parquet file found in data/.
    Returns a pandas DataFrame with all original columns.
    """
    path = _find_data_file()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    print(f"Loaded data: {path}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_target_col(df):
    """
    Detect target column. Priority:
      1. TARGET_COL environment variable
      2. Column name matching TARGET_CANDIDATES (case-insensitive)
      3. Last column
    """
    env_override = os.environ.get("TARGET_COL")
    if env_override:
        if env_override not in df.columns:
            raise ValueError(
                f"TARGET_COL env var '{env_override}' not found in columns: {list(df.columns)}"
            )
        return env_override

    lower_to_orig = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_orig:
            return lower_to_orig[candidate]

    # Fall back to last column
    return df.columns[-1]


def _detect_task_type(series):
    """
    Infer classification vs regression from a target Series.
    Priority:
      1. TASK_TYPE environment variable ("classification" or "regression")
      2. Non-numeric dtype (string, object, bool, category) -> classification
      3. Integer-like numeric dtype with <=20 unique values -> classification
      4. Otherwise -> regression
    """
    env_override = os.environ.get("TASK_TYPE", "").lower()
    if env_override in ("classification", "regression"):
        return env_override

    # Non-numeric types are always classification
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    # Bool/category
    if series.dtype.name in ("category", "bool", "boolean"):
        return "classification"
    # Integer-like with few unique values
    if pd.api.types.is_integer_dtype(series) and series.nunique() <= 20:
        return "classification"
    return "regression"


def _detect_feature_types(df, target_col):
    """
    Split feature columns into numeric and categorical, dropping:
    - the target column
    - columns that are all-NaN
    - high-cardinality object columns (> MAX_CARDINALITY unique values)
    Returns (numeric_features, categorical_features, dropped_features).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = []
    categorical_features = []
    dropped_features = []

    for col in feature_cols:
        s = df[col]
        if s.isna().all():
            dropped_features.append((col, "all_nan"))
            continue
        if pd.api.types.is_numeric_dtype(s):
            numeric_features.append(col)
        elif s.nunique() > MAX_CARDINALITY:
            dropped_features.append((col, f"high_cardinality({s.nunique()})"))
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features, dropped_features


def get_task_info():
    """
    Auto-detect and return task metadata dict:
      - task_type: "classification" or "regression"
      - target_col: name of the target column
      - primary_metric: "f1_macro" (classification) or "r2" (regression)
      - numeric_features: list of numeric feature column names
      - categorical_features: list of categorical feature column names
      - dropped_features: list of (name, reason) tuples for dropped columns
      - n_classes: int (classification only)
      - class_names: list (classification only)
      - data_path: path to the data file used
    """
    df = load_data()
    target_col = _detect_target_col(df)
    task_type = _detect_task_type(df[target_col])
    numeric_feats, cat_feats, dropped = _detect_feature_types(df, target_col)

    info = {
        "task_type": task_type,
        "target_col": target_col,
        "primary_metric": "f1_macro" if task_type == "classification" else "r2",
        "numeric_features": numeric_feats,
        "categorical_features": cat_feats,
        "dropped_features": dropped,
        "data_path": _find_data_file(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }

    if task_type == "classification":
        classes = sorted(df[target_col].dropna().astype(str).unique())
        info["n_classes"] = len(classes)
        info["class_names"] = classes

    return info


# ---------------------------------------------------------------------------
# Train/test split (fixed -- always called the same way)
# ---------------------------------------------------------------------------

def get_train_test_split():
    """
    Return (X_train, X_test, y_train, y_test) with fixed seed and split ratio.
    For classification: stratified split. For regression: random split.
    The test set is immutable ground truth -- agents should NOT evaluate on it
    during tuning; use cross-validation on X_train instead.
    """
    from sklearn.model_selection import train_test_split

    df = load_data()
    info = get_task_info()
    target_col = info["target_col"]
    feature_cols = info["numeric_features"] + info["categorical_features"]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    stratify = y if info["task_type"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric contract)
# ---------------------------------------------------------------------------

def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Evaluate a fitted sklearn-compatible pipeline against the fixed test set.
    Returns a dict with:
      - primary_score: float (higher is always better)
      - primary_metric: str
      - all additional metrics relevant to the task type
    """
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
        roc_auc_score, r2_score, mean_squared_error, mean_absolute_error,
    )

    info = get_task_info()
    task_type = info["task_type"]
    y_pred = pipeline.predict(X_test)

    metrics = {}

    if task_type == "classification":
        metrics["f1_macro"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision_macro"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_test, y_pred, average="macro", zero_division=0)

        # AUC: only for binary or when predict_proba is available
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
                n_classes = info.get("n_classes", 2)
                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        y_test, y_proba, multi_class="ovr", average="macro"
                    )
            except Exception:
                pass

        metrics["primary_score"] = metrics["f1_macro"]
        metrics["primary_metric"] = "f1_macro"

    else:  # regression
        metrics["r2"] = r2_score(y_test, y_pred)
        metrics["rmse"] = math.sqrt(mean_squared_error(y_test, y_pred))
        metrics["mae"] = mean_absolute_error(y_test, y_pred)

        metrics["primary_score"] = metrics["r2"]
        metrics["primary_metric"] = "r2"

    return metrics


# ---------------------------------------------------------------------------
# EDA report generation
# ---------------------------------------------------------------------------

def generate_eda_report(output_path="eda_report.md"):
    """
    Generate a markdown EDA report from the data file.
    Saves to output_path. Called by `uv run prepare.py`.
    """
    df = load_data()
    info = get_task_info()
    target_col = info["target_col"]
    task_type = info["task_type"]
    numeric_feats = info["numeric_features"]
    cat_feats = info["categorical_features"]
    dropped = info["dropped_features"]

    lines = []
    lines.append("# EDA Report")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nData file: `{info['data_path']}`")

    # -- Dataset overview --
    lines.append("\n## Dataset Overview")
    lines.append(f"\n| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Rows | {len(df):,} |")
    lines.append(f"| Columns | {len(df.columns)} |")
    lines.append(f"| Memory usage | {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB |")
    lines.append(f"| Target column | `{target_col}` |")
    lines.append(f"| Task type | **{task_type}** |")
    lines.append(f"| Primary metric | `{info['primary_metric']}` (higher is better) |")
    lines.append(f"| Numeric features | {len(numeric_feats)} |")
    lines.append(f"| Categorical features | {len(cat_feats)} |")
    if dropped:
        lines.append(f"| Dropped features | {len(dropped)} |")

    # -- Missing values --
    lines.append("\n## Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0].sort_values("pct", ascending=False)

    if missing_df.empty:
        lines.append("\nNo missing values found.")
    else:
        lines.append(f"\n{len(missing_df)} columns have missing values:\n")
        lines.append("| Column | Missing Count | Missing % | Flag |")
        lines.append("|---|---|---|---|")
        for col, row in missing_df.iterrows():
            flag = "⚠ mostly missing" if row["pct"] / 100 > MOSTLY_MISSING_THRESHOLD else ""
            lines.append(f"| `{col}` | {int(row['count']):,} | {row['pct']}% | {flag} |")

    # -- Target distribution --
    lines.append("\n## Target Distribution")
    lines.append(f"\nTarget column: `{target_col}`")

    if task_type == "classification":
        vc = df[target_col].value_counts().sort_index()
        total = len(df)
        lines.append(f"\n| Class | Count | % |")
        lines.append("|---|---|---|")
        for cls, cnt in vc.items():
            lines.append(f"| `{cls}` | {cnt:,} | {cnt/total*100:.1f}% |")
        # Imbalance warning
        if len(vc) >= 2:
            ratio = vc.max() / vc.min()
            if ratio > 5:
                lines.append(f"\n> **Imbalance warning**: majority/minority ratio = {ratio:.1f}x. "
                              "Consider class weighting or resampling.")
    else:
        desc = df[target_col].describe()
        lines.append(f"\n| Statistic | Value |")
        lines.append("|---|---|")
        for stat, val in desc.items():
            try:
                lines.append(f"| {stat} | {float(val):.4f} |")
            except (TypeError, ValueError):
                lines.append(f"| {stat} | {val} |")
        try:
            skew = float(df[target_col].skew())
            lines.append(f"| skewness | {skew:.4f} |")
            if abs(skew) > 1:
                lines.append(f"\n> **Skew warning**: target skewness = {skew:.2f}. "
                             "Consider log/sqrt transformation.")
        except Exception:
            pass

    # -- Numeric features --
    highly_skewed = []
    if numeric_feats:
        lines.append("\n## Numeric Features")
        desc = df[numeric_feats].describe().T
        desc["skew"] = df[numeric_feats].skew()
        desc["missing_pct"] = (df[numeric_feats].isnull().mean() * 100).round(1)
        lines.append(f"\n{desc[['count','mean','std','min','50%','max','skew','missing_pct']].to_markdown()}")

        # Flag skewed features
        highly_skewed = [c for c in numeric_feats if abs(df[c].skew()) > 2]
        if highly_skewed:
            lines.append(f"\n> **Highly skewed features** (|skew| > 2): {', '.join(f'`{c}`' for c in highly_skewed)}")

    # -- Categorical features --
    if cat_feats:
        lines.append("\n## Categorical Features")
        lines.append("\n| Feature | Unique Values | Top Value | Top Freq % | Missing % |")
        lines.append("|---|---|---|---|---|")
        for col in cat_feats:
            s = df[col]
            n_unique = s.nunique()
            top = s.value_counts().index[0] if not s.value_counts().empty else "N/A"
            top_freq = s.value_counts().iloc[0] / len(s) * 100 if not s.value_counts().empty else 0
            miss_pct = s.isnull().mean() * 100
            lines.append(f"| `{col}` | {n_unique} | `{top}` | {top_freq:.1f}% | {miss_pct:.1f}% |")

    # -- Correlations (numeric only) --
    if len(numeric_feats) >= 2:
        lines.append("\n## Feature Correlations")
        corr = df[numeric_feats].corr()
        # Top pairs (upper triangle, sorted by absolute correlation)
        pairs = []
        for i, c1 in enumerate(numeric_feats):
            for c2 in numeric_feats[i+1:]:
                pairs.append((c1, c2, corr.loc[c1, c2]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = pairs[:15]
        lines.append("\nTop 15 correlated feature pairs:\n")
        lines.append("| Feature A | Feature B | Correlation |")
        lines.append("|---|---|---|")
        for c1, c2, r in top_pairs:
            lines.append(f"| `{c1}` | `{c2}` | {r:.4f} |")

        # Correlation with target (for numeric targets or encoded classification)
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_corr = df[numeric_feats + [target_col]].corr()[target_col].drop(target_col)
            target_corr = target_corr.abs().sort_values(ascending=False).head(10)
            lines.append(f"\n### Top features correlated with `{target_col}`:\n")
            lines.append("| Feature | |Correlation| |")
            lines.append("|---|---|")
            for feat, val in target_corr.items():
                lines.append(f"| `{feat}` | {val:.4f} |")

    # -- Dropped features --
    if dropped:
        lines.append("\n## Dropped Features")
        lines.append("\nThese features were excluded automatically:\n")
        lines.append("| Feature | Reason |")
        lines.append("|---|---|")
        for col, reason in dropped:
            lines.append(f"| `{col}` | {reason} |")

    # -- Agent recommendations --
    lines.append("\n## Agent Recommendations")
    recs = []
    if missing_df is not None and not missing_df.empty:
        recs.append("- Missing values detected: experiment with different imputation strategies "
                    "(median, mean, KNN, iterative).")
    if highly_skewed if numeric_feats else []:
        recs.append("- Highly skewed numeric features: try log1p or power transforms.")
    if cat_feats:
        recs.append("- Categorical features present: try target encoding, ordinal encoding, or "
                    "leave-one-out encoding in addition to one-hot.")
    if task_type == "classification":
        recs.append("- Classification task: try LogisticRegression (baseline), RandomForest, "
                    "GradientBoosting, XGBoost, LightGBM, and SVM.")
        recs.append("- Use `cross_val_score` on X_train for tuning — never peek at X_test.")
    else:
        recs.append("- Regression task: try Ridge/Lasso (baseline), RandomForest, GradientBoosting, "
                    "XGBoost, LightGBM.")
        recs.append("- Consider polynomial features or interaction terms for numeric columns.")
    recs.append("- Use `RandomizedSearchCV` with `n_iter` limits to stay within the time budget.")

    lines.extend(recs)
    lines.append("\n---\n*End of EDA Report*")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"EDA report written to: {output_path}")
    return report


# ---------------------------------------------------------------------------
# Main (one-time setup)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and generate EDA for autoresearch-ds")
    parser.add_argument("--no-eda", action="store_true", help="Skip writing eda_report.md")
    args = parser.parse_args()

    print("=" * 60)
    print("autoresearch-ds: data preparation")
    print("=" * 60)
    print()

    t0 = time.time()

    # Load and inspect
    df = load_data()
    info = get_task_info()

    print(f"\nTask type:          {info['task_type']}")
    print(f"Target column:      {info['target_col']}")
    print(f"Primary metric:     {info['primary_metric']} (higher is better)")
    print(f"Numeric features:   {len(info['numeric_features'])}")
    print(f"Categorical features: {len(info['categorical_features'])}")
    if info["dropped_features"]:
        print(f"Dropped features:   {len(info['dropped_features'])}")
        for col, reason in info["dropped_features"]:
            print(f"  - {col}: {reason}")
    if info["task_type"] == "classification":
        print(f"Classes ({info['n_classes']}):       {info['class_names']}")

    print()

    # Validate split
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Train set:  {X_train.shape[0]} rows x {X_train.shape[1]} features")
    print(f"Test set:   {X_test.shape[0]} rows x {X_test.shape[1]} features")

    if not args.no_eda:
        print()
        generate_eda_report()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Ready to run experiments: uv run pipeline.py")
