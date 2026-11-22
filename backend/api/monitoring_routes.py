"""
Read-only monitoring endpoints: data drift and flagged accounts.

These only report what the daily job already recorded. Nothing here can hold, block or
close an account; flags are recommendations for a human to act on.
"""

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "retrain_data_dir", "") or ".")


def _drift_lines(data_dir: Path):
    path = data_dir / "drift_log.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a half-written line must not take the page down
    return out


@router.get("/drift/latest", summary="Full drift report from the most recent run")
def drift_latest(request: Request):
    lines = _drift_lines(_dir(request))
    return lines[-1] if lines else {"drift": None}


@router.get("/drift/history", summary="Compact drift history, newest first")
def drift_history(request: Request, limit: int = 30):
    rows = []
    for line in reversed(_drift_lines(_dir(request))):
        d, a = line.get("drift") or {}, line.get("anomalies") or {}
        rows.append({
            "job_id": line.get("job_id"), "at": line.get("at"), "sim_days": line.get("sim_days"),
            "status": d.get("status"), "max_psi": d.get("max_psi"),
            "score_psi": (d.get("score") or {}).get("psi"),
            "flagged_features": [f["feature"] for f in d.get("features", []) if f.get("severity") == "significant"],
            "accounts_flagged": a.get("flagged"), "accounts_checked": a.get("accounts_checked"),
            "by_severity": a.get("by_severity"),
        })
    return {"history": rows[: max(1, min(limit, 200))]}


@router.get("/anomalies/latest", summary="Flagged accounts for a simulated day (default: newest)")
def anomalies_latest(request: Request, day: Optional[int] = None, limit: int = 100,
                     severity: Optional[List[str]] = Query(default=None)):
    folder = _dir(request) / "anomalies"
    files = sorted(folder.glob("day_*.json")) if folder.exists() else []
    if day is not None:
        files = [f for f in files if f.name == f"day_{day:04d}.json"]
    if not files:
        return {"day": day, "summary": None, "records": []}
    try:
        data = json.loads(files[-1].read_text())
    except ValueError:
        return {"day": day, "summary": None, "records": []}
    records = data.get("records", [])
    if severity:
        records = [r for r in records if r.get("severity") in severity]
    return {"day": data.get("day"), "summary": data.get("summary"),
            "records": records[: max(1, min(limit, 500))]}
