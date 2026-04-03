"""
Autoresearch-DS experiment script.
This is the ONLY file the agent modifies.

Everything is fair game: preprocessing, feature engineering, model selection,
hyperparameter tuning, cross-validation strategy, ensembles, etc.

What you CANNOT do:
- Modify prepare.py (fixed evaluation harness, train/test split, metrics)
- Install new packages (only use what's in pyproject.toml)
- Peek at X_test during tuning -- use cross-validation on X_train only

Usage: uv run pipeline.py
"""

import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

from prepare import (  # noqa: E402
    TIME_BUDGET,
    RANDOM_STATE,
    get_task_info,
    get_train_test_split,
    evaluate_pipeline,
)


# ---------------------------------------------------------------------------
# Logging helpers (do not remove -- used throughout for observability)
# ---------------------------------------------------------------------------

def _section(title):
    """Print a clearly visible section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _bullet(label, value, indent=2):
    """Print a labeled key-value bullet."""
    print(f"{' ' * indent}• {label}: {value}")


def _list_features(features, label, indent=4, max_show=20):
    """Print a named feature list, truncating if very long."""
    pad = " " * indent
    print(f"\n{' ' * (indent - 2)}  {label} ({len(features)}):")
    if not features:
        print(f"{pad}(none)")
        return
    for feat in features[:max_show]:
        print(f"{pad}- {feat}")
    if len(features) > max_show:
        print(f"{pad}  ... and {len(features) - max_show} more")


def _print_top_features(names, feat_scores, label="importance", top_n=15):
    """Print ranked feature scores (importances, coefficients, etc.)."""
    pairs = sorted(
        zip(names, feat_scores), key=lambda x: x[1], reverse=True
    )[:top_n]
    if not pairs:
        return
    max_score = max(s for _, s in pairs)
    print(f"\n  Top {min(top_n, len(pairs))} features by {label}:")
    for rank, (name, score) in enumerate(pairs, 1):
        bar_len = int(score / max_score * 20) if max_score > 0 else 0
        bar = "\u2588" * bar_len
        print(f"    {rank:3d}. {name:<40} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

_section("TASK SETUP")

info = get_task_info()
task_type = info["task_type"]
target_col = info["target_col"]
numeric_features = info["numeric_features"]
categorical_features = info["categorical_features"]

_bullet("Task type", task_type.upper())
_bullet("Target column", target_col)
_bullet("Primary metric", f"{info['primary_metric']} (higher is better)")
_bullet("Time budget", f"{TIME_BUDGET}s")
_bullet("Data path", info["data_path"])

if task_type == "classification":
    _bullet("Classes", f"{info['n_classes']}  →  {info['class_names']}")

_list_features(numeric_features, "Numeric features")
_list_features(categorical_features, "Categorical features")

if info.get("dropped_features"):
    print(f"\n  Dropped features ({len(info['dropped_features'])}):")
    for col, reason in info["dropped_features"]:
        print(f"    - {col}  [reason: {reason}]")

X_train, X_test, y_train, y_test = get_train_test_split()
_bullet("Split", f"{len(X_train)} train / {len(X_test)} test rows")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Agent: modify this section to change imputation, scaling, or encoding.
# After each change, add a print() explaining what you changed and why.

_section("PREPROCESSING")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
print("  Numeric pipeline:")
print("    1. SimpleImputer(strategy='median')  — robust to outliers")
print("    2. StandardScaler()                  — zero mean, unit variance")

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
print("  Categorical pipeline:")
print("    1. SimpleImputer(strategy='most_frequent')")
print("    2. OneHotEncoder(handle_unknown='ignore')")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)
n_num = len(numeric_features)
n_cat = len(categorical_features)
print(f"\n  Applying to {n_num} numeric + {n_cat} categorical features")
print("  Remainder columns: dropped")


# ---------------------------------------------------------------------------
# Feature selection / engineering (optional)
# ---------------------------------------------------------------------------
# Agent: add feature selection or engineering here.
# Log every decision so the output stays interpretable, e.g.:
#
#   from sklearn.feature_selection import SelectFromModel
#   selector = SelectFromModel(
#       RandomForestClassifier(n_estimators=50), threshold="mean"
#   )
#   # After fitting, print which features were kept:
#   selected = [f for f, keep in zip(feat_names, selector.get_support()) if keep]
#   print(f"  SelectFromModel kept {len(selected)} features:")
#   for f in selected:
#       print(f"    - {f}")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# Agent: swap out the model below. Add a print() block describing:
#   - Why you chose this model
#   - Key hyperparameters and their values
#   - Any trade-offs vs. the previous model

_section("MODEL SELECTION")

if task_type == "classification":
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model_type = "LogisticRegression"
    print(f"  Model:  {model_type}")
    print("  Reason: linear baseline — interpretable, fast, reference point")
    print(
        f"  Params: max_iter=1000, C=1.0 (default), "
        f"solver='lbfgs', random_state={RANDOM_STATE}"
    )
else:
    from sklearn.linear_model import Ridge  # noqa: E402
    model = Ridge(random_state=RANDOM_STATE)
    model_type = "Ridge"
    print(f"  Model:  {model_type}")
    print("  Reason: linear baseline — fast, stable, interpretable coefs")
    print(f"  Params: alpha=1.0 (default), random_state={RANDOM_STATE}")


# ---------------------------------------------------------------------------
# Full pipeline assembly
# ---------------------------------------------------------------------------

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Agent: add RandomizedSearchCV, cross-validation, or staged training here.
# Use n_jobs=-1 and cap n_iter to stay within TIME_BUDGET.
# After any search, print the best params and CV scores:
#
#   print(f"  Best params: {search.best_params_}")
#   print(f"  Best CV score: {search.best_score_:.4f}")
#   print(
#       f"  CV scores: mean={cv_scores.mean():.4f}"
#       f"  std={cv_scores.std():.4f}"
#   )

_section("TRAINING")

t_train_start = time.time()
print("  Fitting pipeline on training set...")
full_pipeline.fit(X_train, y_train)
training_seconds = time.time() - t_train_start
print(f"  Done in {training_seconds:.1f}s")


# ---------------------------------------------------------------------------
# Post-fit introspection (feature importances / coefficients)
# ---------------------------------------------------------------------------

_section("INTROSPECTION")

n_features_orig = len(numeric_features) + len(categorical_features)

# Count expanded features after preprocessing (one-hot may increase count)
try:
    preprocessed = full_pipeline.named_steps["preprocessor"]
    n_features_used = preprocessed.transform(X_train[:1]).shape[1]
    print(f"  Input features:  {n_features_orig} original columns")
    print(
        f"  Output features: {n_features_used} "
        "after encoding/transformation"
    )
except Exception:  # noqa: BLE001
    n_features_used = n_features_orig
    print(f"  Features: {n_features_used}")

# Extract feature names from ColumnTransformer
feat_names: list = []
try:
    preprocessed = full_pipeline.named_steps["preprocessor"]
    feat_names = list(preprocessed.get_feature_names_out())
except Exception:  # noqa: BLE001
    pass

# Display feature importance or coefficients
fitted_model = full_pipeline.named_steps["model"]

if feat_names and hasattr(fitted_model, "feature_importances_"):
    importances = list(fitted_model.feature_importances_)
    _print_top_features(feat_names, importances, label="importance")

elif feat_names and hasattr(fitted_model, "coef_"):
    coef = fitted_model.coef_
    if coef.ndim > 1:
        # Multi-class: average absolute coefficient across classes
        coef_scores = list(abs(coef).mean(axis=0))
        _print_top_features(
            feat_names, coef_scores,
            label="mean |coefficient| (multi-class)"
        )
    else:
        coef_scores = list(abs(coef))
        _print_top_features(feat_names, coef_scores, label="|coefficient|")

else:
    print("  (model does not expose feature_importances_ or coef_)")


# ---------------------------------------------------------------------------
# Evaluation (do not change this block)
# ---------------------------------------------------------------------------

_section("EVALUATION")

metrics = evaluate_pipeline(full_pipeline, X_test, y_test)
total_seconds = time.time() - t_start

print(f"  {metrics['primary_metric']}: {metrics['primary_score']:.6f}")
secondary = {
    k: v for k, v in metrics.items()
    if k not in ("primary_score", "primary_metric")
}
for key, val in secondary.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.6f}")
    else:
        print(f"  {key}: {val}")
print(
    f"\n  Total time: {total_seconds:.1f}s"
    f"  (training: {training_seconds:.1f}s)"
)


# ---------------------------------------------------------------------------
# Output (fixed format -- agent must preserve this block exactly)
# grep command: grep "^score:\|^model_type:" run.log
# ---------------------------------------------------------------------------

print()
print("---")
print(f"score:            {metrics['primary_score']:.6f}")
print(f"metric:           {metrics['primary_metric']}")
print(f"task_type:        {task_type}")
print(f"model_type:       {model_type}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"n_features_used:  {n_features_used}")
print(f"n_features_orig:  {n_features_orig}")

for key, val in metrics.items():
    if key not in ("primary_score", "primary_metric"):
        if isinstance(val, float):
            print(f"{key}:{'':8}{val:.6f}")
        else:
            print(f"{key}: {val}")
