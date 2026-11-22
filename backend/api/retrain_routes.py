"""
Read-only view of the retraining job: scheduler state, simulated-day progress and the run log.
Like the registry endpoints, nothing here can start a run or change what serves decisions.
"""

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(prefix="/retrain", tags=["retraining"])


def _data_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "retrain_data_dir", "") or ".")


def _read_log(data_dir: Path):
    path = data_dir / "retrain_log.jsonl"
    if not path.exists():
        return []
    runs = []
    for line in path.read_text().splitlines():
        try:
            runs.append(json.loads(line))
        except ValueError:
            continue  # a half-written or corrupt line must not take the page down
    return runs


@router.get("/status", summary="Scheduler state and latest run")
def status(request: Request):
    data_dir = _data_dir(request)
    runs = _read_log(data_dir)
    scheduler = getattr(request.app.state, "retrain_scheduler", None)
    state_file = data_dir / "state.json"
    next_day = None
    if state_file.exists():
        try:
            next_day = json.loads(state_file.read_text()).get("next_day")
        except ValueError:
            pass
    return {
        "scheduler": scheduler.status() if scheduler else {"enabled": False},
        "last_run": runs[-1] if runs else None,
        "next_simulated_day": next_day,
    }


@router.get("/log", summary="Retraining runs, newest first")
def log(request: Request, limit: int = 30):
    runs = _read_log(_data_dir(request))
    return {"runs": list(reversed(runs))[: max(1, min(limit, 200))]}
