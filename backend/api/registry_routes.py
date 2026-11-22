"""
Read-only endpoints for the model registry.

Deliberately no write endpoints: promotion and rollback are done through
`python -m ml.contest.run`, so nothing reachable over HTTP can change which
model serves decisions.
"""

from fastapi import APIRouter, HTTPException, Request

from ml.registry.store import RegistryError

router = APIRouter(prefix="/registry", tags=["registry"])


def _registry(request: Request):
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(status_code=503, detail="Model registry is not available.")
    return reg


@router.get("/champion", summary="Model currently designated for serving")
def champion(request: Request):
    return {"champion": _registry(request).champion()}


@router.get("/runs", summary="Contest runs, newest first")
def runs(request: Request):
    return {"runs": list(reversed(_registry(request).list_runs()))}


@router.get("/runs/{run_id}", summary="One contest run")
def run_detail(run_id: str, request: Request):
    try:
        return _registry(request).get_run(run_id)
    except RegistryError:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")


@router.get("/events", summary="Promotion and rollback audit trail, newest first")
def events(request: Request):
    return {"events": list(reversed(_registry(request).events()))}
