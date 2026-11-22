"""
Case-management API: the analyst alert queue and simulated account actions.

Reading is open. Every action that changes something (hold, dismiss, release, close) needs a
named analyst in the request body and, when MS_ANALYST_TOKEN is set on the server, a matching
X-Analyst-Token header. Without a token the demo allows actions, as the rest of this
prototype has no authentication; set the token before exposing it anywhere.
"""

import hmac
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ml.cases.store import CaseError, CaseStore, Conflict, NotFound

router = APIRouter(prefix="/cases", tags=["cases"])


class Decision(BaseModel):
    decided_by: str = Field(..., min_length=2, max_length=80, description="Who is making this decision.")
    note: str = Field(default="", max_length=500)


def _store(request: Request) -> CaseStore:
    store = getattr(request.app.state, "cases_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Case management is not available.")
    return store


def require_analyst(request: Request):
    token = getattr(request.app.state, "analyst_token", None)
    if not token:
        return
    supplied = request.headers.get("X-Analyst-Token", "")
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="A valid analyst token is required for this action.")


def _run(fn, *args):
    try:
        return fn(*args)
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except CaseError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------- reads

@router.get("/summary", summary="Counts for the queue badge and dashboard")
def summary(request: Request):
    return _store(request).summary()


@router.get("/alerts", summary="Alert queue, critical first")
def alerts(request: Request, status: Optional[str] = None, limit: int = 100,
           severity: Optional[List[str]] = Query(default=None)):
    return {"alerts": _store(request).list_alerts(status=status, severities=severity, limit=limit)}


@router.get("/alerts/{alert_id}", summary="One alert with its account, notices and history")
def alert_detail(alert_id: int, request: Request):
    return _run(_store(request).get_alert, alert_id)


@router.get("/accounts", summary="Accounts that have been acted on")
def accounts(request: Request, status: Optional[str] = None):
    return {"accounts": _store(request).list_accounts(status=status)}


@router.get("/events", summary="Audit trail, newest first")
def events(request: Request, limit: int = 50, customer_id: Optional[str] = None):
    return {"events": _store(request).events(limit=limit, customer_id=customer_id)}


# ----------------------------------------------------------------- actions

@router.post("/alerts/{alert_id}/hold", dependencies=[Depends(require_analyst)],
             summary="Place a temporary hold (simulated) after reviewing an alert")
def hold(alert_id: int, body: Decision, request: Request):
    return _run(_store(request).hold, alert_id, body.decided_by, body.note)


@router.post("/alerts/{alert_id}/dismiss", dependencies=[Depends(require_analyst)],
             summary="Dismiss an alert as a false alarm")
def dismiss(alert_id: int, body: Decision, request: Request):
    return _run(_store(request).dismiss, alert_id, body.decided_by, body.note)


@router.post("/accounts/{customer_id}/release", dependencies=[Depends(require_analyst)],
             summary="Lift a temporary hold (a reason is required)")
def release(customer_id: str, body: Decision, request: Request):
    return _run(_store(request).release, customer_id, body.decided_by, body.note)


@router.post("/accounts/{customer_id}/close", dependencies=[Depends(require_analyst)],
             summary="Close a held account (final; a reason is required)")
def close(customer_id: str, body: Decision, request: Request):
    return _run(_store(request).close, customer_id, body.decided_by, body.note)
