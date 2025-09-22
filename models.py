# models.py
from __future__ import annotations
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class ModelFactory:
    """XGB / XGBRF / LGBM / CatBoost / RF 생성기 (일관된 기본값 적용)"""
    def __init__(self, default_n_estimators: int = 500, random_state: int = 42):
        self.default_n = int(default_n_estimators)
        self.random_state = int(random_state)

    @staticmethod
    def names() -> List[str]:
        return ["xgb", "xgbrf", "lgbm", "cat", "rf"]

    def create(self, name: str, params: Dict[str, Any]) -> object:
        n = name.lower()
        p = dict(params or {})

        if n == "xgb":
            base = dict(
                n_estimators=self.default_n,
                tree_method="hist",
                eval_metric="auc",
                n_jobs=-1,
                random_state=self.random_state,
                objective=p.pop("objective", "binary:logistic"),
            )
            base.update(p)
            return XGBClassifier(**base)

        if n == "xgbrf":
            base = dict(
                n_estimators=self.default_n,
                subsample=0.8,
                colsample_bynode=0.8,
                tree_method="hist",
                eval_metric="auc",
                n_jobs=-1,
                random_state=self.random_state,
                objective=p.pop("objective", "binary:logistic"),
            )
            base.update(p)
            return XGBRFClassifier(**base)

        if n == "lgbm":
            base = dict(
                n_estimators=self.default_n,
                objective=p.pop("objective", "binary"),
                boosting_type="gbdt",
                n_jobs=-1,
                random_state=self.random_state,
            )
            base.update(p)
            return LGBMClassifier(**base)

        if n == "cat":
            base = dict(
                iterations=self.default_n,
                loss_function=p.pop("loss_function", "Logloss"),
                verbose=False,
                random_state=self.random_state,
            )
            base.update(p)
            return CatBoostClassifier(**base)

        if n == "rf":
            base = dict(
                n_estimators=max(500, self.default_n),
                n_jobs=-1,
                class_weight=p.pop("class_weight", "balanced"),
                random_state=self.random_state,
            )
            base.update(p)
            return RandomForestClassifier(**base)
        raise ValueError(f"Unknown model: {name} (choose from {self.names()})")
