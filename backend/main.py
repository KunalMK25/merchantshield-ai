"""
MerchantShield AI backend entry point.

Startup behavior: loads the frozen model + explainer + metadata exactly once
(see services/model_loader.py) and opens the audit DB connection. If the model
artifact is missing or corrupt, this raises during startup -- the process will
not come up in a half-working state. If ONLY the model fails to load, /health
still comes up so orchestration tooling can see the degraded state and act on
it (this is deliberate: model_bundle is attached to app.state as None rather
than crashing the whole process, so the failure is visible via /health and
/model/info returns a clean 503 rather than every request 500ing).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.services.model_loader import load_model_bundle, ModelUnavailableError
from backend.services.audit_service import AuditStore
from backend.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merchantshield")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_bundle = load_model_bundle(settings.MODEL_PATH, settings.MODEL_METADATA_PATH)
        logger.info(f"Model loaded from {settings.MODEL_PATH}")
    except ModelUnavailableError as e:
        # Do not crash the process -- come up degraded so /health reports it clearly
        # rather than every deploy tool seeing an opaque process crash.
        logger.error(f"MODEL FAILED TO LOAD: {e}")
        app.state.model_bundle = None

    try:
        app.state.audit_store = AuditStore(settings.AUDIT_DB_URL, settings.AUDIT_DB_DIR)
        logger.info(f"Audit store ready at {settings.AUDIT_DB_PATH}")
    except Exception as e:
        logger.error(f"AUDIT STORE FAILED TO INITIALIZE: {e}")
        app.state.audit_store = None

    yield

    # no special teardown needed for sqlite/joblib-loaded objects


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.include_router(router)

# CORS is only needed when the frontend runs as a separate dev-server process
# (`npm run dev` on a different port) against this API during development. The
# production build (Phase 9) is served same-origin via StaticFiles below, where
# CORS plays no role at all. Left permissive here since this is a prototype with
# no auth/session cookies to protect; tightening this is a config change, not a
# code change, if this were ever deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Turns Pydantic/FastAPI's validation errors into a clean, structured JSON body
    instead of the (already-JSON, but verbose/nested) default, and guarantees no
    stack trace ever reaches the client for a malformed request.
    """
    messages = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        messages.append(f"{loc}: {err.get('msg')}")
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": "; ".join(messages) or "Invalid request body."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort handler: any exception that escapes a route (a genuine bug) still
    returns clean structured JSON, never a raw traceback to the client. The real
    traceback is logged server-side for debugging.
    """
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


@app.get("/api", tags=["ops"], summary="API root", description="Basic service identification.")
def api_root():
    return {"service": settings.API_TITLE, "version": settings.API_VERSION, "docs": "/docs"}


# Serve the built frontend (frontend/dist, produced by `npm run build`) same-origin
# at "/" so no CORS is needed in production. Mounted LAST, after all API routes,
# so it never shadows /health, /model/info, /risk/*, /audit-log, /docs, etc.
# If the frontend hasn't been built yet, the API still runs fine on its own --
# this is a soft dependency, not a hard requirement to start the backend.
#
# IMPORTANT FOR FUTURE EDITS: this StaticFiles mount is a catch-all for "/" and
# everything under it. FastAPI/Starlette matches routes in REGISTRATION ORDER, so
# any API route added to this file (or to backend/api/routes.py's router) AFTER
# this mount() call would be silently shadowed by the static handler and never
# reached. All API routes must be registered (via app.include_router / app.get /
# app.post, etc.) BEFORE this line. This mount should stay the last thing
# registered in this file.
_FRONTEND_DIST = os.path.join(settings.PROJECT_ROOT, "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", tags=["ops"])
    def root_no_frontend():
        return {
            "service": settings.API_TITLE, "version": settings.API_VERSION, "docs": "/docs",
            "note": "Frontend build not found at frontend/dist -- run `npm run build` in frontend/.",
        }
