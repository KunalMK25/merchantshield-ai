"""
False-positive / false-negative cost model.

IMPORTANT — these are CONFIGURABLE ASSUMPTIONS, not measured ground truth:

FP_COST: the operational cost of wrongly flagging a legitimate transaction
    (support burden, customer friction, possible churn). We do not have real
    merchant support-cost data, so we use a flat assumed figure and say so
    everywhere this number is shown.

FN_COST: the cost of letting a fraudulent transaction through. We do NOT claim
    this equals the transaction amount — a merchant rarely loses 100% of a
    fraudulent transaction's value (chargeback processes, partial recovery,
    insurance, etc. vary a lot). We model it as a configurable FRACTION of the
    transaction amount, explicitly labeled as an assumption, and we run a
    sensitivity analysis (see threshold_analysis.py) across multiple fractions
    rather than presenting one number as truth.
"""

from dataclasses import dataclass


@dataclass
class CostAssumptions:
    fp_cost: float = 50.0            # flat assumed cost per false positive (INR)
    fn_cost_fraction: float = 0.5    # assumed fraction of txn amount lost per false negative


def expected_cost(y_true, y_pred, amounts, assumptions: CostAssumptions):
    """
    Returns a breakdown dict. y_true/y_pred are 0/1 arrays, amounts is the
    transaction amount array aligned with them.
    """
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_mask = (y_pred == 1) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    tp_count = int(tp_mask.sum())
    tn_count = int(tn_mask.sum())

    fp_total_cost = fp_count * assumptions.fp_cost
    fn_total_cost = float((amounts[fn_mask] * assumptions.fn_cost_fraction).sum())

    return dict(
        fp_count=fp_count,
        fn_count=fn_count,
        tp_count=tp_count,
        tn_count=tn_count,
        fp_cost_assumed=assumptions.fp_cost,
        fn_cost_fraction_assumed=assumptions.fn_cost_fraction,
        fp_total_cost=round(fp_total_cost, 2),
        fn_total_cost=round(fn_total_cost, 2),
        total_expected_cost=round(fp_total_cost + fn_total_cost, 2),
    )
