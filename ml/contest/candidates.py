"""
Candidate models for the contest.

Each candidate declares whether it is tree-based. Only tree-based models may be
promoted to serving, because the API explains every decision with SHAP tree values.
"""

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

EARLY_STOP = 50


def _pos_weight(y):
    pos = int(y.sum())
    return (len(y) - pos) / max(pos, 1)


def _fit_logreg(X, y, Xv, yv, seed):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
    return m.fit(X, y)


def _fit_rf(X, y, Xv, yv, seed):
    return RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced", n_jobs=-1, random_state=seed).fit(X, y)


def _fit_lgbm(X, y, Xv, yv, seed):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, max_depth=6,
                           scale_pos_weight=_pos_weight(y), subsample=0.8, colsample_bytree=0.8,
                           min_child_samples=20, random_state=seed, verbose=-1)
    # first_metric_only: stop on AUC, not on the logloss that scale_pos_weight inflates.
    m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="auc",
          callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True, verbose=False)])
    return m


def _fit_xgb(X, y, Xv, yv, seed):
    m = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=_pos_weight(y), eval_metric="aucpr",
                      early_stopping_rounds=EARLY_STOP, tree_method="hist", n_jobs=-1,
                      random_state=seed)
    return m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)


def _fit_catboost(X, y, Xv, yv, seed):
    m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                           auto_class_weights="Balanced", eval_metric="AUC", random_seed=seed,
                           verbose=False, allow_writing_files=False, thread_count=-1)
    return m.fit(X, y, eval_set=(Xv, yv), early_stopping_rounds=EARLY_STOP)


# name -> (fit function, tree_based)
CANDIDATES = {
    "logistic_regression": (_fit_logreg, False),
    "random_forest": (_fit_rf, True),
    "lightgbm": (_fit_lgbm, True),
    "xgboost": (_fit_xgb, True),
    "catboost": (_fit_catboost, True),
}
