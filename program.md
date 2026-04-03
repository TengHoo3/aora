# autoresearch-ds

This is an experiment to have an LLM agent do its own data science research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on the dataset and today's date (e.g. `titanic-apr1`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Verify data exists**: Check that `data/` contains a CSV or Parquet file. If not, tell the human to place their dataset there.
4. **Read the in-scope files in full**:
   - `README.md` — repository context and design.
   - `prepare.py` — fixed constants, auto-detection logic, evaluation harness. **Do not modify.**
   - `pipeline.py` — your primary experiment file. Preprocessing, model, training loop.
   - `eda_report.md` — generated EDA report (if it exists). This is your primary source of domain knowledge about the dataset. If it doesn't exist, run `uv run prepare.py` first.
   - `train.py` — neural network fallback. Read it but **do not run or modify** until pipeline.py has been exhausted (see Neural Network Fallback section).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

Each experiment runs from scratch: preprocess → train → evaluate. Since the data science pipeline can complete quickly on small/medium datasets, each run is expected to finish well within the **5 minute time budget** (`TIME_BUDGET = 300` in `prepare.py`). Use cross-validation on the training set for hyperparameter tuning, and rely on the fixed `evaluate_pipeline()` call at the end for the reported score.

**What you CAN do in `pipeline.py`:**
- Preprocessing: imputation strategies, scaling, encoding, feature transformations
- Feature engineering: polynomial features, interaction terms, binning, log transforms
- Feature selection: variance threshold, SelectKBest, RFE, SHAP-based selection
- Model selection: any model available in the installed packages
- Hyperparameter tuning: `RandomizedSearchCV` or `GridSearchCV` (watch the time budget)
- Cross-validation strategy: KFold, StratifiedKFold, etc.
- Ensembles: VotingClassifier/Regressor, StackingClassifier/Regressor
- Dimensionality reduction: PCA, factor analysis

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. The `evaluate_pipeline()` function and the train/test split are ground truth — do not circumvent them.
- Modify `program.md` or `README.md`. These are human-maintained files.
- Peek at `X_test` during training or hyperparameter tuning. Use cross-validation on `X_train` only.
- Install new packages or add dependencies. Use only what's already in `pyproject.toml`.
- Change the output format at the bottom of `pipeline.py` or `train.py`. The `---` block must remain parseable.

**The goal is simple: get the highest score.** Since classification uses `f1_macro` and regression uses `r2`, higher is always better for both. The fixed test set and `evaluate_pipeline()` call are immutable — this is the ground truth.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement with a lot of complexity is usually not worth it. Removing a step and getting the same score is a win.

**The first run**: Your very first run should always establish the baseline — run `pipeline.py` as-is without modifications.

## Available Models in pipeline.py (from pyproject.toml)

**Classification:**
- `sklearn.linear_model.LogisticRegression` (baseline)
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.GradientBoostingClassifier`
- `sklearn.ensemble.ExtraTreesClassifier`
- `sklearn.svm.SVC`
- `xgboost.XGBClassifier`
- `lightgbm.LGBMClassifier`
- `sklearn.ensemble.VotingClassifier`, `StackingClassifier`

**Regression:**
- `sklearn.linear_model.Ridge`, `Lasso`, `ElasticNet` (baseline)
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.GradientBoostingRegressor`
- `sklearn.ensemble.ExtraTreesRegressor`
- `xgboost.XGBRegressor`
- `lightgbm.LGBMRegressor`
- `sklearn.ensemble.VotingRegressor`, `StackingRegressor`

## Neural Network Fallback (train.py)

`train.py` provides a PyTorch `TabularMLP` — a feedforward neural network as an alternative to tree-based methods. It uses the same `prepare.py` harness and outputs the same `---` format.

**Escalate to `train.py` ONLY when all of the following are true:**
1. You have completed the full pipeline.py experiment progression (baseline → feature engineering → model selection → tuning → ensembles).
2. The best `pipeline.py` score is still below the escalation threshold:
   - Classification: `f1_macro < 0.70`
   - Regression: `r2 < 0.60`
3. You have a specific reason to believe a neural network will help (e.g. very large dataset where tree methods plateau, complex interactions, or high-dimensional numeric data).

**If you escalate:**
- Switch from `uv run pipeline.py > run.log` to `uv run train.py > run.log`.
- You may modify `train.py` just like `pipeline.py` — same rules apply.
- The `results.tsv` log continues with the same columns. Use `train.py` model names like `TabularMLP(256-128-64)`.
- If `train.py` still underperforms, return to `pipeline.py` with the best ensemble found so far.

**What to tune in `train.py`:**
- `HIDDEN_DIMS`: width and depth of the MLP
- `DROPOUT`, `LR`, `WEIGHT_DECAY`, `BATCH_SIZE`
- The `TabularMLP` class: add residual connections, attention, entity embeddings for categoricals

## Suggested Experiment Progression

Follow this roughly; you don't need to be rigid, but starting from fundamentals prevents premature optimization:

1. **Baseline**: Run `pipeline.py` as-is (LogisticRegression or Ridge). Record score.
2. **Feature engineering**: Try different imputation strategies (KNN, iterative), encoding (target encoding, ordinal), transforms (log1p for skewed), interaction terms. Stay with the same simple model so you can isolate the impact.
3. **Model selection**: Try RF, GBM, XGBoost, LightGBM with default hyperparameters. Find the strongest model family.
4. **Hyperparameter tuning**: Use `RandomizedSearchCV` on the best model family. Limit `n_iter` to stay within `TIME_BUDGET`.
5. **Ensembles**: Combine top-performing models with `VotingClassifier`/`StackingClassifier`.
6. **Advanced**: Feature selection (SHAP values, RFE), dimensionality reduction (PCA), target encoding, class imbalance handling (class weights, SMOTE from `imbalanced-learn`).
7. **Neural fallback** *(only if steps 1–6 fail the escalation threshold)*: Switch to `train.py`, tune `TabularMLP` architecture and hyperparameters.

## Output Format

Both `pipeline.py` and `train.py` print an identical summary block at the end:

```
---
score:            0.854200
metric:           f1_macro
task_type:        classification
model_type:       RandomForestClassifier
training_seconds: 12.3
total_seconds:    14.1
n_features_used:  48
n_features_orig:  30
```

Extract the key metric:

```bash
grep "^score:" run.log
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	score	model_type	status	description
```

1. git commit hash (short, 7 chars)
2. score achieved (e.g. 0.854200) — use 0.000000 for crashes
3. model type (e.g. `RandomForestClassifier` or `TabularMLP(256-128-64)`)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	score	model_type	status	description
a1b2c3d	0.721400	LogisticRegression	keep	baseline
b2c3d4e	0.783200	RandomForestClassifier	keep	switch to RF default params
c3d4e5f	0.779000	GradientBoostingClassifier	discard	GBM worse than RF
d4e5f6g	0.801500	RandomForestClassifier	keep	RF + log1p skewed features
e5f6g7h	0.000000	XGBClassifier	crash	XGB param error
f6g7h8i	0.689000	TabularMLP(256-128-64)	discard	neural fallback — worse than RF
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/titanic-apr1`).

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Read `eda_report.md` to stay grounded in the dataset characteristics
3. Tune the active experiment file (`pipeline.py` or `train.py`) with an idea
4. `git commit`
5. Run: `uv run pipeline.py > run.log 2>&1` (or `uv run train.py > run.log 2>&1` if escalated)
6. Read results: `grep "^score:\|^model_type:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` and attempt a fix.
8. Record results in `results.tsv`
9. If `score` improved (higher), keep the commit and advance
10. If `score` is equal or worse, `git reset` to discard and revert

**Timeout**: If a run exceeds 10 minutes, kill it, treat as failure, and revert. This usually means your hyperparameter search `n_iter` is too high or your CV folds are too many — reduce them.

**Crashes**: Fix trivial errors (import, typo). If the idea is fundamentally broken (OOM, incompatible transformer), log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". You are autonomous. If you run out of ideas, re-read `eda_report.md`, revisit near-misses in `results.tsv`, try combining previous winning approaches, or go deeper on the best model family's hyperparameters. The loop runs until the human interrupts you, period.
