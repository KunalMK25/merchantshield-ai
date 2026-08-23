from fastapi import APIRouter, Request, HTTPException

from backend.schemas.transaction import RiskRequest
from backend.schemas.response import (
    HealthResponse, ModelInfoResponse, RiskScoreResponse, RiskExplainResponse,
    RiskEvaluateResponse, ContributionItem,
)
from backend.services import risk_service
from backend.services.risk_service import FeatureGenerationError
from backend.services.audit_service import AuditPersistenceError
from ml.evaluation.decision_engine import InvalidTransactionError
from ml.features.build_features import FEATURE_COLUMNS

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"],
            summary="Liveness/readiness check",
            description="Returns whether the API process is up and whether the frozen "
                        "LightGBM model was loaded successfully at startup.")
def health(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.metadata.get("model_name") if bundle else None,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["ops"],
            summary="Frozen model metadata",
            description="Returns the frozen model's version, decision threshold, feature list, "
                        "and the validation/test metrics recorded when the threshold was selected "
                        "(Phase 5). This is read-only, static metadata -- it does not run inference.")
def model_info(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; service is degraded.")
    meta = bundle.metadata
    return ModelInfoResponse(
        model_version=meta.get("model_name", "lgbm_v1"),
        model_file=meta.get("model_file", ""),
        decision_threshold=meta.get("selected_threshold", 0.40),
        selection_rule=meta.get("selection_rule", ""),
        validation_metrics_at_threshold=meta.get("validation_metrics_at_threshold", {}),
        test_metrics_at_frozen_threshold=meta.get("test_metrics_at_frozen_threshold", {}),
        feature_columns=FEATURE_COLUMNS,
    )


def _get_bundle_or_503(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; service is degraded.")
    return bundle


@router.post("/risk/score", response_model=RiskScoreResponse, tags=["risk"],
             summary="Fraud probability + risk score (lightweight)",
             description="Runs feature engineering and model inference only. Returns the fraud "
                         "probability and the 0-100 risk score/category. Does NOT run SHAP "
                         "explanation, does NOT run the decision engine, and does NOT write an "
                         "audit record -- use /risk/evaluate for the full decision flow.")
def risk_score(payload: RiskRequest, request: Request):
    bundle = _get_bundle_or_503(request)
    try:
        result = risk_service.score_only(bundle, payload)
    except FeatureGenerationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return RiskScoreResponse(**result)


@router.post("/risk/explain", response_model=RiskExplainResponse, tags=["risk"],
             summary="Fraud probability + full SHAP explanation",
             description="Runs feature engineering, inference, and SHAP explanation. Returns the "
                         "grounded, per-feature contribution breakdown and the correctly-framed "
                         "human-readable reasons (never claims a transaction was 'flagged' unless "
                         "its probability actually crosses the frozen decision threshold). Does "
                         "NOT run the decision engine and does NOT write an audit record.")
def risk_explain(payload: RiskRequest, request: Request):
    bundle = _get_bundle_or_503(request)
    try:
        result = risk_service.explain_only(bundle, payload)
    except FeatureGenerationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    result["contributions"] = [ContributionItem(**c) for c in result["contributions"]]
    return RiskExplainResponse(**result)


@router.post("/risk/evaluate", response_model=RiskEvaluateResponse, tags=["risk"],
             summary="Full risk decision (score + explain + decide + audit)",
             description="The complete pipeline: feature engineering -> LightGBM inference -> "
                         "risk score/category -> SHAP explanation -> deterministic decision "
                         "engine -> audit persistence. This is the endpoint a real integration "
                         "should call for an actual risk decision on a transaction. If audit "
                         "persistence fails, the decision is still returned "
                         "(audit_persisted=false) rather than losing the risk decision -- see "
                         "docs/api.md for the failure-recovery rationale.")
def risk_evaluate(payload: RiskRequest, request: Request):
    bundle = _get_bundle_or_503(request)

    try:
        decision, explanation, request_id = risk_service.evaluate_full(bundle, payload)
    except FeatureGenerationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except InvalidTransactionError as e:
        raise HTTPException(status_code=422, detail=f"Decision engine rejected input: {e}")
    except Exception as e:
        # Anything else here is an unexpected internal failure in a trusted,
        # already-tested component -- surface as a clean 500, no stack trace,
        # rather than letting FastAPI's default handler leak internals.
        raise HTTPException(status_code=500, detail=f"Internal risk evaluation failure: {e}")

    audit_store = getattr(request.app.state, "audit_store", None)
    audit_persisted = False
    audit_error = None
    if audit_store is not None:
        try:
            audit_store.record_decision(request_id, decision, decision.model_explanation, source=payload.source)
            audit_persisted = True
        except AuditPersistenceError as e:
            audit_error = str(e)
    else:
        audit_error = "Audit store unavailable."

    return RiskEvaluateResponse(
        request_id=request_id,
        transaction_id=decision.transaction_id,
        model_version=decision.model_version,
        fraud_probability=decision.fraud_probability,
        threshold=decision.threshold,
        risk_score=decision.risk_score,
        risk_category=decision.risk_category,
        action=decision.action,
        policy_rule_id=decision.policy_rule_id,
        policy_reason=decision.policy_reason,
        explanation_header=explanation["header"],
        reasons=explanation["reasons"],
        timestamp=decision.timestamp,
        audit_persisted=audit_persisted,
        audit_error=audit_error,
    )


@router.get("/audit-log", tags=["ops"],
            summary="Recent audit log entries",
            description="Returns the most recent persisted decision records (default 50), most "
                        "recent first. For prototype/demo inspection of the audit trail.")
def audit_log(request: Request, limit: int = 50):
    audit_store = getattr(request.app.state, "audit_store", None)
    if audit_store is None:
        raise HTTPException(status_code=503, detail="Audit store is not available.")
    try:
        return audit_store.get_recent(limit=min(limit, 500))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audit log: {e}")
