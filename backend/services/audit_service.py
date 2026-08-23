"""
Audit persistence.

WHAT'S STORED AND WHY (deliberately minimal -- see Phase 8 instruction against
storing unnecessary sensitive information):
- transaction_id, timestamp, model_version, fraud_probability, threshold,
  risk_score, risk_category, action, policy_rule_id, policy_reason, request_id:
  the exact fields needed to reconstruct WHAT the system decided and WHY, per
  the decision engine's own DecisionRecord.
- top_reasons_json: the grounded SHAP-derived explanation reasons, stored as a
  JSON string, kept in its OWN column separate from the decision fields above --
  this is the "clearly separate audit data from transient explanation output"
  requirement: the decision record is the durable governance artifact, the
  explanation is attached alongside it but is conceptually a distinct kind of
  data (derived, human-readable narrative vs. the hard decision facts).
- source ("demo" | "manual"): purely descriptive request-origin metadata (Phase 9.5
  remediation) so a reviewer of the audit log can tell a dashboard demo-scenario
  evaluation apart from a hand-entered one. Set by the API layer from the request
  payload, never read or used by the decision engine -- it has no bearing on the
  decision itself.

WHAT'S DELIBERATELY NOT STORED:
- customer_id, device_id, geo_region, payment_method, or any other raw
  transaction PII/context. The audit trail's job is to prove what the RISK
  SYSTEM decided and why, not to duplicate a full copy of merchant transaction
  data (which would belong in the merchant's own systems, not a fraud-scoring
  prototype's audit log).
"""

import json
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    transaction_id = Column(String(64), nullable=False, index=True)
    model_version = Column(String(32), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_category = Column(String(16), nullable=False)
    action = Column(String(32), nullable=False)
    policy_rule_id = Column(String(64), nullable=False)
    policy_reason = Column(Text, nullable=False)
    decision_timestamp = Column(String(64), nullable=False)  # timestamp of the decision itself
    top_reasons_json = Column(Text, nullable=True)           # explanation, stored separately
    source = Column(String(16), nullable=False, default="manual")  # "demo" | "manual"
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class AuditPersistenceError(RuntimeError):
    """Raised when an audit record cannot be written."""


class AuditStore:
    def __init__(self, db_url: str, db_dir: str = None):
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def record_decision(self, request_id: str, decision_record, top_reasons: list, source: str = "manual") -> None:
        """
        decision_record: ml.evaluation.decision_engine.DecisionRecord
        source: "demo" or "manual" -- descriptive only, see module docstring.
        Raises AuditPersistenceError on failure -- caller decides whether that
        should fail the whole request or just be surfaced as a warning.
        """
        session = self.Session()
        try:
            entry = AuditLogEntry(
                request_id=request_id,
                transaction_id=decision_record.transaction_id,
                model_version=decision_record.model_version,
                fraud_probability=decision_record.fraud_probability,
                threshold=decision_record.threshold,
                risk_score=decision_record.risk_score,
                risk_category=decision_record.risk_category,
                action=decision_record.action,
                policy_rule_id=decision_record.policy_rule_id,
                policy_reason=decision_record.policy_reason,
                decision_timestamp=decision_record.timestamp,
                top_reasons_json=json.dumps(top_reasons or []),
                source=source if source in ("demo", "manual") else "manual",
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            raise AuditPersistenceError(f"Failed to persist audit record: {e}") from e
        finally:
            session.close()

    def get_recent(self, limit: int = 50) -> list:
        session = self.Session()
        try:
            rows = (session.query(AuditLogEntry)
                    .order_by(AuditLogEntry.id.desc())
                    .limit(limit)
                    .all())
            return [
                dict(
                    request_id=r.request_id, transaction_id=r.transaction_id,
                    model_version=r.model_version, fraud_probability=r.fraud_probability,
                    threshold=r.threshold, risk_score=r.risk_score, risk_category=r.risk_category,
                    action=r.action, policy_rule_id=r.policy_rule_id, policy_reason=r.policy_reason,
                    decision_timestamp=r.decision_timestamp,
                    top_reasons=json.loads(r.top_reasons_json) if r.top_reasons_json else [],
                    source=r.source or "manual",
                    created_at=r.created_at.isoformat(),
                )
                for r in rows
            ]
        finally:
            session.close()
