# How to run aora

Step-by-step guide to go from zero to a running autonomous data science experiment.

---

## What you need

- macOS (Apple Silicon or Intel)
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — OpenClaw uses this for its built-in sandbox
- [Ollama](https://ollama.com) — local LLM inference

---

## How the stack works

```
HOST macOS
├── ollama serve             ← runs the local LLM (never touches your files)
├── uv run host_bridge.py   ← MPS runner for train.py (port 8765)
└── openclaw (on host)
      ├── calls Ollama for inference (localhost:11434)
      ├── calls host_bridge for train.py MPS runs (localhost:8765)
      └── sandbox (Docker, managed by OpenClaw)
              └── runs pipeline.py, reads/writes tasks/ only
```

**Blast radius:** OpenClaw's workspace is locked to the `aora/` directory (`openclaw.json`), mounted read-only.
Only `pipeline.py`, `train.py`, and `tasks/` are bind-mounted writable inside the Docker sandbox.
`.env` and `openclaw.json` are hard-denied via `fs.deny`; `prepare.py` is protected by the read-only mount.

---

## One-time setup (do this once)

**1. Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Install Python deps**
```bash
cd aora
uv sync
```

**3. Install Ollama and pull a coding model**
```bash
brew install ollama    # or download from ollama.com
ollama pull qwen2.5-coder:14b
```
Verify: `ollama list` — model name must appear exactly as you'll use it.

**4. Install OpenClaw**
The quickest path via Ollama (installs OpenClaw + web search plugin automatically):
```bash
ollama launch openclaw
```
Or install directly and configure manually — see [OpenClaw docs](https://docs.openclaw.ai).

**5. Point OpenClaw at this repo**
OpenClaw reads `openclaw.json` from its workspace directory. The file is already in the repo root and sets:
- **workspace**: `aora/` directory, mounted read-only at `/agent`; `pipeline.py`, `train.py`, and `tasks/` are bind-mounted writable
- **sandbox**: always-on Docker (scope: agent — one persistent container); bridge network so exec can reach `host.docker.internal:8765`
- **tools**: `minimal` profile + explicit allow for `exec`, `read`, `write`, `group:web`; `delete`, `apply_patch`, browser, messaging, automation blocked
- **protected files**: `.env`, `openclaw.json` via `fs.deny`; `prepare.py` via read-only workspace mount

When you open OpenClaw in (or point it at) the `aora/` directory, it picks up this config automatically.

**6. Place your dataset**
```bash
cp /path/to/your/data.csv data/
```
Supported: `.csv`, `.tsv`, `.parquet`. One file in `data/` at a time.

---

## Manual sanity check (run before autonomous experiments)

**Step 1: generate EDA**
```bash
uv run prepare.py
```
Expected: prints task type, target column, class names, train/test split sizes. Generates `eda_report.md`.

**Step 2: baseline pipeline**
```bash
uv run pipeline.py
```
Expected: `TASK SETUP` → `PREPROCESSING` → `MODEL SELECTION` → `TRAINING` → `INTROSPECTION` → `EVALUATION` → `---` block with `score:` and `model_type:`.

**Step 3: verify grep**
```bash
uv run pipeline.py > run.log 2>&1
grep "^score:\|^model_type:" run.log
```
Expected: exactly two lines. If blank, check `tail -20 run.log`.

**Step 4: verify train.py (neural fallback)**
```bash
uv run train.py > run.log 2>&1
grep "^score:\|^model_type:" run.log
```
Expected: same two-line format, `model_type: TabularMLP(...)`.

---

## Starting an autonomous run

You need **2 terminals** running simultaneously:

| Terminal | What runs |
|----------|-----------|
| 1 | `ollama serve` |
| 2 | `uv run host_bridge.py` |

OpenClaw itself runs as a background service (started by `ollama launch openclaw` or `openclaw gateway start`).

**Terminal 1 — start Ollama**
```bash
ollama serve
```

**Terminal 2 — start host bridge (MPS runner for train.py)**
```bash
cd aora
uv run host_bridge.py
```
Expected output:
```
aora host_bridge — MPS runner
Workspace: .../aora
Port: 8765
Auth: disabled
```
Verify:
```bash
curl http://localhost:8765/health
```

**Start the agent**
Open OpenClaw (TUI, UI, or messaging channel) and send:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will:
1. Ask you for a task name (e.g. `titanic`)
2. Create `tasks/titanic_20260404_143022/` with copies of `pipeline.py` and `train.py`
3. Run `uv run prepare.py` and copy `eda_report.md` into the task directory
4. Run `uv run tasks/titanic_.../pipeline.py` baseline
5. Loop: tune → run → keep/discard → repeat (entirely within the task directory)

---

## Task directory structure

Each experiment gets its own dated directory — nothing is lost:

```
tasks/
├── titanic_20260404_143022/
│   ├── pipeline.py       ← evolves during the run (best state persists here)
│   ├── train.py          ← neural fallback (only if escalated)
│   ├── eda_report.md     ← dataset analysis
│   ├── results.tsv       ← full experiment log for this task
│   └── run.log           ← last run output (gitignored)
│
└── housing_20260405_090100/
    ├── pipeline.py
    └── ...
```

**What is tracked in git:**
- `tasks/*/pipeline.py` ✓
- `tasks/*/train.py` ✓
- `tasks/*/results.tsv` ✓
- `tasks/*/run.log` ✗ (gitignored)
- `tasks/*/eda_report.md` ✗ (gitignored — regenerate with `uv run prepare.py`)
- `tasks/*/data/` ✗ (gitignored — data stays in `data/`)

**To resume a past task**, tell the agent the task directory name. It reads `results.tsv` to understand where it left off.

---

## OpenClaw sandbox and blast radius

| Layer | Boundary |
|-------|----------|
| **File access** | `aora/` workspace read-only at `/agent`. `pipeline.py`, `train.py`, `tasks/` bind-mounted writable. Cannot touch `~/Documents`, SSH keys, etc. |
| **Code execution** | OpenClaw sandbox container (Docker, bridge network). Working directory: `/agent`. |
| **Blocked files** | `.env`, `openclaw.json` — `fs.deny`; `prepare.py` — protected by read-only workspace mount |
| **Network** | Container uses bridge network. `exec` reaches host bridge at `host.docker.internal:8765`. `group:web` tools run on host (no container network needed). |
| **train.py** | Always runs on host via bridge (MPS). Never inside sandbox. |

---

## Configuring OpenClaw tools

Tool access is set in `openclaw.json` (already in the repo). Current config:
- **Profile**: `minimal` (base: `session_status` only)
- **Explicitly allowed**: `exec`, `read`, `write`, `group:web` (web_search, web_fetch)
- **Denied**: `delete`, `apply_patch`, browser, messaging, nodes, automation, image generation

To add or remove tools, edit `openclaw.json`. See [OpenClaw tools docs](https://docs.openclaw.ai/tools).

**Web search** is enabled by default. For local models it may require:
```bash
ollama signin
openclaw plugins install @ollama/openclaw-web-search
```
See: [Ollama × OpenClaw](https://docs.ollama.com/integrations/openclaw)

---

## Monitoring a run

```bash
# Watch the active task log live
tail -f tasks/<taskname>_<datetime>/run.log

# Check current score
grep "^score:\|^model_type:" tasks/<taskname>_<datetime>/run.log

# Check bridge health
curl http://localhost:8765/health

# Read last train.py score (if escalated)
curl http://localhost:8765/score
```

---

## After a run

```bash
# Full experiment history for a task
cat tasks/<taskname>_<datetime>/results.tsv

# Best pipeline code
cat tasks/<taskname>_<datetime>/pipeline.py
```

The best `pipeline.py` is already saved in the task directory — nothing to do.
If you want it tracked in git it's already not gitignored.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uv sync` fails | Check Python ≥ 3.10: `python --version` |
| `grep "^score:"` returns blank | Run crashed — `tail -20 tasks/.../run.log` |
| `curl localhost:8765/health` fails | host_bridge not running — start Terminal 2 |
| `train.py` runs on CPU | Expected — MPS detected by the subprocess. Check `Device: MPS` in run.log |
| OpenClaw can't find model | `ollama list` — model name must match exactly |
| OpenClaw edits files outside `aora/` | Check `openclaw.json` workspace setting; update OpenClaw if config not respected |
| Sandbox not starting | Verify Docker Desktop is running |

---

## File responsibilities at a glance

| File | Who edits | What it does |
|------|-----------|--------------|
| `prepare.py` | Nobody (fixed) | EDA, fixed split, fixed metrics |
| `pipeline.py` | Template (root copy) | Starting point copied into each task dir |
| `train.py` | Template (root copy) | Neural fallback starting point |
| `tasks/<name>/pipeline.py` | Agent | Active experiment for that task |
| `tasks/<name>/train.py` | Agent (if escalated) | Neural fallback for that task |
| `host_bridge.py` | Nobody | Runs train.py on host with MPS |
| `openclaw.json` | Human only | Workspace, tools, sandbox config |
| `program.md` | Human | Agent operating rules |
| `data/` | Human | Dataset (one file at a time) |
