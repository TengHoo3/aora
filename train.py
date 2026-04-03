"""
Autoresearch-DS (AORA) neural network

Use this file ONLY after pipeline.py (sklearn/XGBoost/LightGBM) is exhausted.

When to escalate here from pipeline.py:
  - Classification f1_macro < 0.70 after full hyperparameter tuning
  - Regression r2 < 0.60 after full hyperparameter tuning
  - Dataset > 100K rows where tree methods have clearly plateaued
  - High-dimensional numeric features with complex interactions

What you CAN modify:
  - HIDDEN_DIMS, DROPOUT, BATCH_SIZE, LR, WEIGHT_DECAY (hyperparameters)
  - TabularMLP architecture: add residual connections, attention layers,
    entity embeddings for categoricals, or deeper/wider networks

What you CANNOT do:
  - Modify prepare.py (fixed evaluation harness, train/test split, metrics)
  - Peek at X_test during training — use the internal val split only
  - Change the output block format (--- section)

Usage:  uv run train.py
Runs on: CPU, MPS (Apple Silicon), or CUDA — auto-detected.
"""

import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline as SklearnPipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")

from prepare import (  # noqa: E402
    TIME_BUDGET,
    RANDOM_STATE,
    get_task_info,
    get_train_test_split,
    evaluate_pipeline,
)


# ---------------------------------------------------------------------------
# Logging helpers (same style as pipeline.py)
# ---------------------------------------------------------------------------

def _section(title):
    """Print a clearly visible section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _bullet(label, value, indent=2):
    """Print a labeled key-value bullet."""
    print(f"{' ' * indent}• {label}: {value}")


# ---------------------------------------------------------------------------
# Hyperparameters (agent: modify these)
# ---------------------------------------------------------------------------

HIDDEN_DIMS = [256, 128, 64]    # hidden layer sizes (agent: try [512, 256, 128])
DROPOUT = 0.3                   # dropout probability
BATCH_SIZE = 256                # training batch size
LR = 1e-3                       # initial AdamW learning rate
WEIGHT_DECAY = 1e-4             # AdamW weight decay
MAX_EPOCHS = 200                # maximum epochs (early stopping usually wins)
PATIENCE = 20                   # early stopping patience in epochs
VAL_FRACTION = 0.15             # fraction of train set held out for val


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    """
    Feedforward MLP for tabular classification and regression.

    Layout: InputBN → [Linear → BN → GELU → Dropout] * n → Linear(output)

    Agent: extend this class to add:
      - Residual connections: store prev output and add to current layer output
      - Wider/deeper: change HIDDEN_DIMS
      - Entity embeddings: add nn.Embedding for each categorical feature
        (replace OrdinalEncoder with raw integer indices in preprocessing)
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class TorchTabularWrapper:
    """
    Wraps a trained PyTorch model in the sklearn .predict() interface so
    evaluate_pipeline() from prepare.py works unchanged.
    """

    def __init__(
        self, model, preprocessor, task_type,
        label_encoder=None, device="cpu"
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.label_encoder = label_encoder
        self.device = device

    def _to_tensor(self, X):
        arr = np.ascontiguousarray(
            self.preprocessor.transform(X).astype(np.float32)
        )
        return torch.from_numpy(arr).to(self.device)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(X))
        if self.task_type == "classification":
            preds = logits.argmax(dim=1).cpu().numpy()
            return self.label_encoder.inverse_transform(preds)
        return logits.squeeze(-1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(X))
        return torch.softmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(RANDOM_STATE)

# Device: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

_section("TASK SETUP")

info = get_task_info()
task_type = info["task_type"]
target_col = info["target_col"]
numeric_features = info["numeric_features"]
categorical_features = info["categorical_features"]

_bullet("Task type", task_type.upper())
_bullet("Target column", target_col)
_bullet("Primary metric", f"{info['primary_metric']} (higher is better)")
_bullet("Device", str(device).upper())
_bullet("Time budget", f"{TIME_BUDGET}s")

if task_type == "classification":
    _bullet("Classes", f"{info['n_classes']}  →  {info['class_names']}")

X_train, X_test, y_train, y_test = get_train_test_split()
_bullet("Split", f"{len(X_train)} train / {len(X_test)} test rows")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Agent: OrdinalEncoder is used (not OneHot) — MLPs handle ordinal fine and
# it keeps input_dim small. Swap to OneHotEncoder if you add entity embeddings.

_section("PREPROCESSING")

numeric_transformer = SklearnPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = SklearnPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1,
    )),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)
print("  Numeric:     SimpleImputer(median) → StandardScaler")
print("  Categorical: SimpleImputer(mode) → OrdinalEncoder")

preprocessor.fit(X_train)
input_dim = preprocessor.transform(X_train[:1]).shape[1]
_bullet("Input dim", f"{input_dim} features after preprocessing")


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

if task_type == "classification":
    label_enc = LabelEncoder()
    y_train_np = label_enc.fit_transform(y_train).astype(np.int64)
    n_classes = len(label_enc.classes_)
    output_dim = n_classes
else:
    label_enc = None
    y_train_np = np.asarray(y_train, dtype=np.float32)
    output_dim = 1


# ---------------------------------------------------------------------------
# Internal validation split (from X_train only — never peek at X_test)
# ---------------------------------------------------------------------------

n_val = max(1, int(len(X_train) * VAL_FRACTION))
n_tr = len(X_train) - n_val
rng = np.random.default_rng(RANDOM_STATE)
shuffle_idx = rng.permutation(len(X_train))
tr_idx, val_idx = shuffle_idx[:n_tr], shuffle_idx[n_tr:]

X_np = preprocessor.transform(X_train).astype(np.float32)

def _to_device(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


X_tr_t = _to_device(X_np[tr_idx])
X_val_t = _to_device(X_np[val_idx])
y_tr_t = _to_device(y_train_np[tr_idx])
y_val_t = _to_device(y_train_np[val_idx])

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
)
_bullet(
    "Train/val (internal)",
    f"{len(tr_idx)} train / {len(val_idx)} val"
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_section("MODEL")

model = TabularMLP(
    input_dim=input_dim,
    hidden_dims=HIDDEN_DIMS,
    output_dim=output_dim,
    dropout=DROPOUT,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
arch_str = " → ".join(str(h) for h in HIDDEN_DIMS)
print(f"  Architecture:  {input_dim} → {arch_str} → {output_dim}")
_bullet("Parameters", f"{n_params:,}")
_bullet("Dropout", DROPOUT)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS,
)
criterion = (
    nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
)
_bullet("Optimizer", f"AdamW(lr={LR}, wd={WEIGHT_DECAY})")
_bullet("LR schedule", f"CosineAnnealingLR(T_max={MAX_EPOCHS})")
_bullet("Loss", criterion.__class__.__name__)
_bullet("Early stop", f"patience={PATIENCE} epochs on val loss")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

_section("TRAINING")

t_train_start = time.time()
best_val_loss = float("inf")
best_state = None
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # Time-budget guard — leave 10% for eval overhead
    if time.time() - t_start >= TIME_BUDGET * 0.9:
        print(f"\n  Time budget reached at epoch {epoch}. Stopping.")
        break

    # Train
    model.train()
    train_loss_sum = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        if task_type == "regression":
            out = out.squeeze(-1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * len(xb)
    scheduler.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        if task_type == "regression":
            val_out = val_out.squeeze(-1)
        val_loss = criterion(val_out, y_val_t).item()

    avg_train_loss = train_loss_sum / len(tr_idx)

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"  epoch {epoch:4d}/{MAX_EPOCHS}"
            f"  train={avg_train_loss:.4f}"
            f"  val={val_loss:.4f}"
        )

    # Early stopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

training_seconds = time.time() - t_train_start
_bullet("Best val loss", f"{best_val_loss:.6f}")
_bullet("Training time", f"{training_seconds:.1f}s")


# ---------------------------------------------------------------------------
# Evaluation (do not change this block)
# ---------------------------------------------------------------------------

_section("EVALUATION")

wrapper = TorchTabularWrapper(
    model=model,
    preprocessor=preprocessor,
    task_type=task_type,
    label_encoder=label_enc,
    device=device,
)

metrics = evaluate_pipeline(wrapper, X_test, y_test)
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
# Output (fixed format — same as pipeline.py, do not change)
# grep command: grep "^score:\|^model_type:" run.log
# ---------------------------------------------------------------------------

n_features_orig = len(numeric_features) + len(categorical_features)
model_type = f"TabularMLP({'-'.join(str(h) for h in HIDDEN_DIMS)})"

print()
print("---")
print(f"score:            {metrics['primary_score']:.6f}")
print(f"metric:           {metrics['primary_metric']}")
print(f"task_type:        {task_type}")
print(f"model_type:       {model_type}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"n_features_used:  {input_dim}")
print(f"n_features_orig:  {n_features_orig}")

for key, val in metrics.items():
    if key not in ("primary_score", "primary_metric"):
        if isinstance(val, float):
            print(f"{key}:{'':8}{val:.6f}")
        else:
            print(f"{key}: {val}")
