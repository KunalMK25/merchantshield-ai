"""
Request schemas.

VALIDATION SCOPE, DELIBERATELY LIMITED: these schemas validate SHAPE and TYPE
(strict types, non-empty strings, positive amounts, valid timestamps, valid enum
values for fields the feature pipeline actually branches on). They do NOT decide
what counts as "risky" -- that judgment belongs entirely to the model + decision
engine downstream. A schema rejecting a request means "this isn't a well-formed
transaction," never "this transaction looks fraudulent."
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

STATUS_VALUES = ("success", "failed")
PAYMENT_METHOD_VALUES = ("card", "upi", "netbanking", "wallet")


class TransactionInput(BaseModel):
    """
    Raw transaction fields, matching the raw event-log schema the existing feature
    pipeline (ml/features/build_features.py) already expects. No engineered
    features are accepted here -- they are always computed server-side.
    """
    transaction_id: str = Field(..., min_length=1, max_length=64,
                                 description="Unique transaction identifier.")
    customer_id: str = Field(..., min_length=1, max_length=64,
                              description="Customer identifier this transaction belongs to.")
    merchant_id: str = Field(..., min_length=1, max_length=64)
    merchant_category: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime = Field(..., description="Transaction time, ISO-8601, UTC recommended.")
    amount: float = Field(..., gt=0, le=10_000_000,
                           description="Transaction amount in INR. Must be positive.")
    device_id: str = Field(..., min_length=1, max_length=128)
    geo_region: str = Field(..., min_length=1, max_length=64)
    payment_method: Literal[PAYMENT_METHOD_VALUES]  # type: ignore[valid-type]
    status: Literal[STATUS_VALUES]  # type: ignore[valid-type]
    account_created: datetime = Field(..., description="When the customer's account was created.")

    @field_validator("transaction_id", "customer_id", "merchant_id", "device_id", "geo_region")
    @classmethod
    def _no_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @model_validator(mode="after")
    def _account_created_not_after_transaction(self):
        if self.account_created > self.timestamp:
            raise ValueError("account_created cannot be after the transaction timestamp")
        return self


class RiskRequest(BaseModel):
    """
    Request body for /risk/score, /risk/explain, /risk/evaluate.

    `prior_transactions` is the customer's own recent transaction history, used
    ONLY to compute strictly-prior behavioral features (velocity, historical
    average amount, device/geo novelty, etc.) for `transaction` via the existing,
    unmodified feature pipeline. This is a deliberate design choice documented in
    docs/api.md: this prototype has no live transaction store, so real-time
    context must be supplied by the caller. If omitted, the transaction is scored
    as this customer's first-ever observed transaction.
    """
    transaction: TransactionInput
    prior_transactions: List[TransactionInput] = Field(default_factory=list, max_length=500)
    source: Literal["demo", "manual"] = Field(
        default="manual",
        description="Where this request originated: 'demo' for the dashboard's built-in synthetic "
                    "scenarios, 'manual' for a hand-entered transaction (also the default for any "
                    "caller that doesn't specify it, e.g. a direct API integration). Recorded on the "
                    "audit trail for /risk/evaluate only -- purely descriptive metadata, never used "
                    "by the decision engine.",
    )

    @model_validator(mode="after")
    def _prior_transactions_must_be_valid_context(self):
        txn = self.transaction
        for i, prior in enumerate(self.prior_transactions):
            if prior.customer_id != txn.customer_id:
                raise ValueError(
                    f"prior_transactions[{i}].customer_id ('{prior.customer_id}') does not match "
                    f"transaction.customer_id ('{txn.customer_id}') -- prior transactions must "
                    f"belong to the same customer being scored."
                )
            if prior.timestamp >= txn.timestamp:
                raise ValueError(
                    f"prior_transactions[{i}].timestamp ({prior.timestamp}) is not strictly before "
                    f"transaction.timestamp ({txn.timestamp}) -- feeding same-or-future transactions "
                    f"as 'prior' context would leak information the real-time system would not "
                    f"actually have at decision time."
                )
            if prior.transaction_id == txn.transaction_id:
                raise ValueError(
                    f"prior_transactions[{i}] has the same transaction_id as the transaction being "
                    f"scored -- a transaction cannot be its own prior context."
                )
        return self
