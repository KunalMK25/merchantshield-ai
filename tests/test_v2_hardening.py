"""
V2 edge-case tests: model output validation, signal quality, SHAP failure handling, feature bounds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backend.services.risk_service import (
    _validate_model_probability, _validate_engineered_features,
    _calculate_signal_quality, ModelOutputError, FeatureGenerationError
)
from backend.schemas.transaction import TransactionInput, RiskRequest
from ml.evaluation.decision_engine import make_decision


# =========================================================================
# V2 Tests: Model output validation
# =========================================================================

class TestModelOutputValidation:
    """V2 hardening: probability validation after model prediction"""

    def test_valid_probability_accepted(self):
        """Normal probabilities are accepted"""
        assert _validate_model_probability(0.0) == 0.0
        assert _validate_model_probability(0.5) == 0.5
        assert _validate_model_probability(1.0) == 1.0
        assert _validate_model_probability(0.0001) > 0
        assert _validate_model_probability(0.9999) < 1.0

    def test_nan_probability_rejected(self):
        """NaN probability raises ModelOutputError"""
        with pytest.raises(ModelOutputError, match="NaN"):
            _validate_model_probability(float("nan"))

    def test_positive_infinity_rejected(self):
        """Positive infinity raises ModelOutputError"""
        with pytest.raises(ModelOutputError, match="infinite"):
            _validate_model_probability(float("inf"))

    def test_negative_infinity_rejected(self):
        """Negative infinity raises ModelOutputError"""
        with pytest.raises(ModelOutputError, match="infinite"):
            _validate_model_probability(float("-inf"))

    def test_probability_above_one_rejected(self):
        """Probability > 1.0 raises ModelOutputError"""
        with pytest.raises(ModelOutputError, match="out-of-range"):
            _validate_model_probability(1.0001)
        with pytest.raises(ModelOutputError, match="out-of-range"):
            _validate_model_probability(2.0)

    def test_probability_below_zero_rejected(self):
        """Probability < 0.0 raises ModelOutputError"""
        with pytest.raises(ModelOutputError, match="out-of-range"):
            _validate_model_probability(-0.0001)
        with pytest.raises(ModelOutputError, match="out-of-range"):
            _validate_model_probability(-1.0)


# =========================================================================
# V2 Tests: Feature bounds validation
# =========================================================================

class TestFeatureBoundsValidation:
    """V2 hardening: pathological feature detection"""

    def test_extreme_velocity_60min_rejected(self):
        """velocity_60min > 1000 (pathological)"""
        row = pd.Series({
            "velocity_60min": 1001,
            "velocity_30min": 100,
            "velocity_5min": 10,
            "prior_txn_count": 100,
            "amount_zscore": 2.0,
            "time_since_prev_txn_min": 1000,
        })
        with pytest.raises(FeatureGenerationError, match="velocity_60min"):
            _validate_engineered_features(row)

    def test_acceptable_velocity_60min_accepted(self):
        """velocity_60min <= 1000 (acceptable)"""
        row = pd.Series({
            "velocity_60min": 1000,
            "velocity_30min": 100,
            "velocity_5min": 10,
            "prior_txn_count": 100,
            "amount_zscore": 2.0,
            "time_since_prev_txn_min": 1000,
        })
        # Should not raise
        _validate_engineered_features(row)

    def test_extreme_prior_txn_count_rejected(self):
        """prior_txn_count > 100,000 (data quality issue)"""
        row = pd.Series({
            "velocity_60min": 100,
            "velocity_30min": 50,
            "velocity_5min": 10,
            "prior_txn_count": 100_001,
            "amount_zscore": 2.0,
            "time_since_prev_txn_min": 1000,
        })
        with pytest.raises(FeatureGenerationError, match="prior_txn_count"):
            _validate_engineered_features(row)

    def test_acceptable_prior_txn_count_accepted(self):
        """prior_txn_count <= 100,000 (acceptable)"""
        row = pd.Series({
            "velocity_60min": 100,
            "velocity_30min": 50,
            "velocity_5min": 10,
            "prior_txn_count": 100_000,
            "amount_zscore": 2.0,
            "time_since_prev_txn_min": 1000,
        })
        # Should not raise
        _validate_engineered_features(row)

    def test_extreme_zscore_rejected(self):
        """amount_zscore with non-finite values (NaN/Inf) should be rejected"""
        row = pd.Series({
            "velocity_60min": 100,
            "velocity_30min": 50,
            "velocity_5min": 10,
            "prior_txn_count": 100,
            "amount_zscore": float("inf"),
            "time_since_prev_txn_min": 1000,
        })
        with pytest.raises(FeatureGenerationError, match="non-finite"):
            _validate_engineered_features(row)

    def test_acceptable_zscore_accepted(self):
        """amount_zscore: extremely high values are acceptable if finite (legitimate fraud signal)"""
        row = pd.Series({
            "velocity_60min": 100,
            "velocity_30min": 50,
            "velocity_5min": 10,
            "prior_txn_count": 100,
            "amount_zscore": 2604.19,  # Legitimate extreme z-score from high transaction vs low history
            "time_since_prev_txn_min": 1000,
        })
        # Should not raise
        _validate_engineered_features(row)

    def test_negative_zscore_accepted(self):
        """negative amount_zscore (even extreme) should be accepted if finite"""
        row = pd.Series({
            "velocity_60min": 100,
            "velocity_30min": 50,
            "velocity_5min": 10,
            "prior_txn_count": 100,
            "amount_zscore": -2000.0,  # Extreme but legitimate (very small amount vs large history)
            "time_since_prev_txn_min": 1000,
        })
        # Should not raise
        _validate_engineered_features(row)

    def test_nan_zscore_rejected(self):
        """amount_zscore with NaN value should be rejected"""
        row = pd.Series({
            "velocity_60min": 100,
            "velocity_30min": 50,
            "velocity_5min": 10,
            "prior_txn_count": 100,
            "amount_zscore": float("nan"),
            "time_since_prev_txn_min": 1000,
        })
        with pytest.raises(FeatureGenerationError, match="non-finite"):
            _validate_engineered_features(row)


# =========================================================================
# V2 Tests: Signal quality calculation
# =========================================================================

class TestSignalQuality:
    """V2 historical context indicator"""

    def test_signal_quality_zero_prior_transactions(self):
        """No prior history → MINIMAL"""
        sq = _calculate_signal_quality(0)
        assert sq["level"] == "MINIMAL"
        assert sq["prior_transaction_count"] == 0
        assert "No prior transaction" in sq["message"]

    def test_signal_quality_one_prior_transaction(self):
        """1 prior → LIMITED"""
        sq = _calculate_signal_quality(1)
        assert sq["level"] == "LIMITED"
        assert sq["prior_transaction_count"] == 1

    def test_signal_quality_two_prior_transactions(self):
        """2 prior → LIMITED"""
        sq = _calculate_signal_quality(2)
        assert sq["level"] == "LIMITED"
        assert sq["prior_transaction_count"] == 2

    def test_signal_quality_three_prior_transactions(self):
        """3 prior → MODERATE"""
        sq = _calculate_signal_quality(3)
        assert sq["level"] == "MODERATE"
        assert sq["prior_transaction_count"] == 3

    def test_signal_quality_nine_prior_transactions(self):
        """9 prior → MODERATE"""
        sq = _calculate_signal_quality(9)
        assert sq["level"] == "MODERATE"
        assert sq["prior_transaction_count"] == 9

    def test_signal_quality_ten_prior_transactions(self):
        """10 prior → ESTABLISHED"""
        sq = _calculate_signal_quality(10)
        assert sq["level"] == "ESTABLISHED"
        assert sq["prior_transaction_count"] == 10

    def test_signal_quality_many_prior_transactions(self):
        """1000 prior → ESTABLISHED"""
        sq = _calculate_signal_quality(1000)
        assert sq["level"] == "ESTABLISHED"
        assert sq["prior_transaction_count"] == 1000


# =========================================================================
# V2 Tests: Decision independence from SHAP
# =========================================================================

class TestDecisionIndependentOfShap:
    """V2 architecture: decision must not depend on SHAP success"""

    def test_decision_created_regardless_of_shap_failure(self):
        """make_decision() works even if model_explanation is None"""
        decision = make_decision(
            transaction_id="txn_test_001",
            model_probability=0.45,
            amount=5000.0,
            model_explanation=None,  # SHAP failed
        )
        assert decision.transaction_id == "txn_test_001"
        assert decision.fraud_probability == 0.45
        assert decision.action == "STEP_UP_VERIFICATION"
        assert decision.model_explanation is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



# =========================================================================
# V2 Tests: Audit persistence with prior_txn_count
# =========================================================================

class TestAuditPersistenceV2:
    """V2 hardening: audit log captures prior_transaction_count"""

    def test_audit_record_includes_prior_txn_count(self, client):
        """Audit record should include prior_txn_count in response"""
        from tests.test_api import _txn, _prior_history

        payload = {
            "transaction": _txn(transaction_id="txn_audit_v2_test"),
            "prior_transactions": _prior_history(),
        }
        r = client.post("/risk/evaluate", json=payload)
        assert r.status_code == 200
        body = r.json()

        # V2 fields should be present
        assert "prior_transaction_count" in body
        assert body["prior_transaction_count"] == 6  # _prior_history returns 6


# =========================================================================
# V2 Tests: Schema migration
# =========================================================================

class TestSchemaMigration:
    """V2 hardening: SQLite schema migration for prior_transaction_count column"""

# =========================================================================
# V2 Tests: Schema migration
# =========================================================================

class TestSchemaMigration:
    """V2 hardening: SQLite schema migration for prior_transaction_count column"""

    def test_migration_creates_correct_schema(self):
        """AuditStore initialization should create table with prior_transaction_count"""
        from backend.services.audit_service import AuditStore
        import tempfile
        import os
        import sqlite3

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test_schema.db")
            db_url = f"sqlite:///{db_path}"

            # Initialize AuditStore
            store = AuditStore(db_url, db_dir=tmpdir)

            # Verify table exists and has prior_transaction_count
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(audit_log)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()

            assert "prior_transaction_count" in columns
            assert "request_id" in columns
            assert "transaction_id" in columns
        finally:
            import shutil
            if os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except:
                    pass

    def test_audit_store_accepts_prior_txn_count_parameter(self):
        """record_decision should accept and store prior_txn_count"""
        from backend.services.audit_service import AuditStore
        from ml.evaluation.decision_engine import DecisionRecord
        import tempfile
        import os

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test_prior.db")
            db_url = f"sqlite:///{db_path}"

            store = AuditStore(db_url, db_dir=tmpdir)

            decision = DecisionRecord(
                transaction_id="txn_prior_test",
                model_version="lgbm_v1",
                fraud_probability=0.35,
                threshold=0.40,
                risk_score=35,
                risk_category="LOW",
                action="ALLOW",
                policy_rule_id="LOW_ALLOW",
                policy_reason="Low fraud probability.",
                timestamp="2026-05-20T10:05:00Z",
                model_explanation=None,
            )

            # This should not raise an error
            store.record_decision(
                request_id="req_test",
                decision_record=decision,
                top_reasons=[],
                source="manual",
                prior_txn_count=10,
            )
        finally:
            import shutil
            if os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except:
                    pass


# =========================================================================
# V2 Tests: Response schema includes new fields
# =========================================================================

class TestResponseSchemaV2:
    """V2 hardening: API response includes new V2 fields"""

    def test_evaluate_response_includes_signal_quality(self, client):
        """RiskEvaluateResponse should include signal_quality field"""
        from tests.test_api import _txn, _prior_history

        payload = {
            "transaction": _txn(transaction_id="txn_schema_test"),
            "prior_transactions": _prior_history(),
        }
        r = client.post("/risk/evaluate", json=payload)
        assert r.status_code == 200
        body = r.json()

        # V2 fields
        assert "signal_quality" in body
        assert "level" in body["signal_quality"]
        assert "message" in body["signal_quality"]
        assert "prior_transaction_count" in body["signal_quality"]

        # V2 response fields
        assert "prior_transaction_count" in body
        assert "audit_persisted" in body
        assert "audit_error" in body

    def test_signal_quality_reflects_prior_txn_count(self, client):
        """Signal quality level should match prior_transaction_count"""
        from tests.test_api import _txn, _prior_history

        # Test with 0 prior transactions
        payload = {
            "transaction": _txn(transaction_id="txn_sq_test_0"),
            "prior_transactions": [],
        }
        r = client.post("/risk/evaluate", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["prior_transaction_count"] == 0
        assert body["signal_quality"]["level"] == "MINIMAL"

        # Test with 6 prior transactions
        payload = {
            "transaction": _txn(transaction_id="txn_sq_test_6"),
            "prior_transactions": _prior_history(),
        }
        r = client.post("/risk/evaluate", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["prior_transaction_count"] == 6
        assert body["signal_quality"]["level"] == "MODERATE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
