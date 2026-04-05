"""
host_bridge.py — Host-side MPS runner for train.py.

Runs on macOS host (NOT in Docker). Accepts HTTP requests from the OpenClaw
container and executes train.py locally with Apple MPS acceleration.

Why this exists:
  Docker on macOS uses a Linux VM. Apple Metal/MPS cannot pass through to
  Linux containers. train.py must run on the host to use MPS.

Security model:
  - Binds to 127.0.0.1 only — not reachable from outside the machine
  - docker-compose maps host.docker.internal to host IP for containers
  - Only whitelisted actions: run_train, kill_train, log, score, health
  - No arbitrary shell execution
  - No file writes outside the workspace
  - Optional: set BRIDGE_SECRET in .env for token auth

Usage:
  uv run host_bridge.py          # start on port 8765
  BRIDGE_PORT=9000 uv run host_bridge.py

Endpoints:
  POST /run_train   — run uv run train.py, return exit code
  GET  /log         — return last run.log contents
  GET  /health      — liveness check
"""

import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent.resolve()
LOG_PATH = WORKSPACE / "outputs" / "run.log"
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 8765))
BRIDGE_SECRET = os.environ.get("BRIDGE_SECRET", "")
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", 600))

app = FastAPI(title="aora host bridge", docs_url=None, redoc_url=None)

# Mutex: only one train.py run at a time
_run_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# Auth middleware (optional — enabled if BRIDGE_SECRET is set)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def check_secret(request: Request, call_next):
    if BRIDGE_SECRET and request.url.path != "/health":
        token = request.headers.get("X-Bridge-Secret", "")
        if token != BRIDGE_SECRET:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "unauthorized"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — called by docker-compose healthcheck."""
    mps_available = False
    try:
        import torch as _torch  # noqa: PLC0415
        mps_available = (
            hasattr(_torch.backends, "mps")
            and _torch.backends.mps.is_available()
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "workspace": str(WORKSPACE),
        "mps_available": mps_available,
    }


@app.post("/run_train")
def run_train():
    """
    Run uv run train.py in the workspace.
    Blocks until complete or MAX_RUN_SECONDS exceeded.
    Returns: exit_code, duration_seconds, log_tail (last 50 lines).
    """
    global _current_proc

    if not _run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A train.py run is already in progress.",
        )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.time()
        with LOG_PATH.open("w") as log_file:
            _current_proc = subprocess.Popen(
                ["uv", "run", "train.py"],
                cwd=WORKSPACE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                exit_code = _current_proc.wait(timeout=MAX_RUN_SECONDS)
            except subprocess.TimeoutExpired:
                _current_proc.kill()
                _current_proc.wait()
                exit_code = -1
            finally:
                _current_proc = None

        duration = round(time.time() - t0, 1)
        log_lines = _read_log_tail(50)

        return {
            "exit_code": exit_code,
            "duration_seconds": duration,
            "log_tail": log_lines,
            "log_path": str(LOG_PATH),
        }

    finally:
        _run_lock.release()


@app.post("/kill_train")
def kill_train():
    """Kill an in-progress train.py run."""
    if _current_proc is None:
        return {"status": "no_run_in_progress"}
    _current_proc.kill()
    return {"status": "killed"}


@app.get("/log")
def get_log(tail: int = 100):
    """Return the last N lines of outputs/run.log."""
    if not LOG_PATH.exists():
        return {"lines": [], "note": "No run.log found yet."}
    return {"lines": _read_log_tail(tail)}


@app.get("/score")
def get_score():
    """
    Parse and return the score from the last run.log.
    Same format as: grep "^score:\\|^model_type:" run.log
    """
    if not LOG_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No run.log found.",
        )
    result = {}
    for line in LOG_PATH.read_text().splitlines():
        for key in ("score", "metric", "model_type", "task_type",
                    "training_seconds", "total_seconds"):
            if line.startswith(f"{key}:"):
                result[key] = line.split(":", 1)[1].strip()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not parse score from run.log. Run may have crashed.",
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_log_tail(n: int) -> list[str]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(errors="replace").splitlines()
    return lines[-n:]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  aora host_bridge — MPS runner")
    print("=" * 60)
    print(f"  Workspace:  {WORKSPACE}")
    print(f"  Log path:   {LOG_PATH}")
    print(f"  Port:       {BRIDGE_PORT}")
    auth_status = "enabled" if BRIDGE_SECRET else "disabled (set BRIDGE_SECRET)"
    print(f"  Auth:       {auth_status}")
    print(f"  Max run:    {MAX_RUN_SECONDS}s")
    print()

    # Verify MPS at startup
    try:
        import torch as _torch  # noqa: PLC0415
        if _torch.backends.mps.is_available():
            print("  MPS: available ✓")
        else:
            print("  MPS: not available — train.py will fall back to CPU")
    except Exception:  # noqa: BLE001
        print("  MPS: check skipped (torch import issue in this env)")
        print("       train.py subprocess will auto-detect device at runtime")
    print()

    uvicorn.run(
        app,
        host="127.0.0.1",   # localhost only — not exposed to network
        port=BRIDGE_PORT,
        log_level="warning",
    )
