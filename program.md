# autoresearch-ds

This is an experiment to have an LLM agent do its own data science research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a task name** with the user (e.g. `titanic`, `housing-prices`). You will use this to create a unique task directory.

2. **Create a task directory** with the format `tasks/<taskname>_<YYYYMMDD_HHMMSS>/`:
   ```
   tasks/titanic_20260404_143022/
   ├── pipeline.py      ← copy from repo root (your working copy)
   ├── train.py         ← copy from repo root
   ├── eda_report.md    ← generated here
   └── results.tsv      ← experiment log for this task
   ```
   Copy `pipeline.py` and `train.py` from the repo root into this directory as the starting point. All your edits and results for this task stay in this directory — never in the root files.

3. **Verify data exists**: Check that `data/` contains a CSV or Parquet file.

4. **Read the in-scope files in full**:
   - `README.md` — repository context and design.
   - `prepare.py` — fixed constants, auto-detection logic, evaluation harness. **Do not modify. Ever.**
   - `tasks/<taskname>_<datetime>/pipeline.py` — your working copy for this task.
   - `tasks/<taskname>_<datetime>/train.py` — neural network fallback for this task.
   - `tasks/<taskname>_<datetime>/eda_report.md` — if it exists. If not, run `uv run prepare.py` from the repo root first, then copy the generated file into your task directory.

5. **Initialize results.tsv**: Create `tasks/<taskname>_<datetime>/results.tsv` with just the header row:
   ```
   commit	score	model_type	status	description
   ```

6. **Confirm and go**: Confirm the task directory is set up, then kick off experimentation.

### Task directory rules
- **You only edit files inside your task directory.** Never edit `pipeline.py` or `train.py` in the repo root.
- `prepare.py` in the repo root is the single fixed harness for all tasks. Always run it from the repo root: `uv run prepare.py`.
- All experiment runs use the task-directory copies: `uv run tasks/<taskname>_<datetime>/pipeline.py`.
- At the end of a good run, record the result in the task's `results.tsv`. The best `pipeline.py` naturally persists in the task directory.
- To **resume** a past task: read its task directory and `results.tsv` to understand where you left off.

### train.py escalation (MPS)
`train.py` uses Apple Metal (MPS) and cannot run inside OpenClaw's sandbox. Call the host bridge instead:
```bash
curl -s -X POST http://host.docker.internal:8765/run_train
# Results written to tasks/<taskname>_<datetime>/run.log
# Read score:
curl -s http://host.docker.internal:8765/score
```

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

## Running Experiments

All experiments run from the **repo root** (so `prepare.py` and `data/` are always accessible). The active files are inside your task directory.

```bash
# pipeline.py (OpenClaw sandbox handles isolation)
uv run tasks/<taskname>_<datetime>/pipeline.py \
    > tasks/<taskname>_<datetime>/run.log 2>&1

# Read score
grep "^score:\|^model_type:" tasks/<taskname>_<datetime>/run.log

# train.py (MPS — always via host bridge, not directly)
curl -s -X POST http://host.docker.internal:8765/run_train
curl -s http://host.docker.internal:8765/score
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/titanic-apr1`).

LOOP FOREVER (working inside `tasks/<taskname>_<datetime>/`):

1. Read `tasks/<taskname>_<datetime>/eda_report.md` to stay grounded in the dataset.
2. Tune `tasks/<taskname>_<datetime>/pipeline.py` with an experimental idea.
3. **Run the experiment** (from the repo root so `prepare.py` and `data/` are accessible):
   ```bash
   uv run tasks/<taskname>_<datetime>/pipeline.py > tasks/<taskname>_<datetime>/run.log 2>&1
   ```
   For `train.py` escalation (MPS, runs on host via bridge):
   ```bash
   curl -s -X POST http://host.docker.internal:8765/run_train
   ```
4. **Read results:**
   ```bash
   grep "^score:\|^model_type:" tasks/<taskname>_<datetime>/run.log
   ```
5. If grep output is empty, the run crashed. Read `tail -n 50 tasks/<taskname>_<datetime>/run.log` and fix.
6. Record result in `tasks/<taskname>_<datetime>/results.tsv`.
7. If `score` improved, keep the edit and advance.
8. If `score` is equal or worse, revert `pipeline.py` in the task directory to its previous state.

**Timeout**: If a run exceeds 10 minutes, kill it, treat as failure, and revert. This usually means your hyperparameter search `n_iter` is too high or your CV folds are too many — reduce them.

**Crashes**: Fix trivial errors (import, typo). If the idea is fundamentally broken (OOM, incompatible transformer), log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". You are autonomous. If you run out of ideas, re-read `eda_report.md`, revisit near-misses in `results.tsv`, try combining previous winning approaches, or go deeper on the best model family's hyperparameters. The loop runs until the human interrupts you, period.
