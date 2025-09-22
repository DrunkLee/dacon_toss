from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from metric import CompetitionMetric
from models import ModelFactory
from utils import class_print, print_sys_usage

class Trainer:
    def __init__(self, model_name:str, params:Dict[str,Any], n_splits:int=5, metric:str="auc", seed:int=42):
        self.model_name = model_name
        self.params = dict(params)
        self.n_splits = n_splits
        self.metric = metric
        self.factory = ModelFactory(random_state=seed)
        self.seed = seed
        self.models_: List[Any] = []
        self.full_model_ = None
        self.features_ = None
        self.oof_pred = Optional[np.ndarray] = None
        self.cv_score_ = Optional[float] = None
        
    @staticmethod
    def _spw(y: np.ndarray) -> float:
        pos = (y==1).sum()
        neg = (y==0).sum()
        return max(1e-6, neg/max(1,pos))
    
    def _metric(self, y, p):
        return f1_score(y, (p>=0.5).astype(int), average="macro") if self.metric=="f1" else roc_auc_score(y,p)
    
    def fit_cv(self, X: pd.DataFrame, y: pd.Series, refit_full:bool = True):
        class_print(self, f"CV 시작: {self.model_name} fold={self.n_splits}")
        self.feature_ = list(X.columns)
        p = dict(self.params)
        if self.model_name.lower() in {'xgb', 'lgbm', "cat"} and "scale_pos_weight" not in p:
            p["scale_pos_weight"] = self._spw(y.values)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.models_.clear()
        self.oof_pred = np.zeros(len(y), dtype=float)
        scores = []
        
        for fold, (tr,va) in enumerate(skf.split(X,y), 1):
            model = self.factory.create(self.model_name, p)
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xva, yva = X.iloc[va], y.iloc[va]
            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xva)[:,1] if hasattr(model, "predict_proba") else model.predict(Xva)
            self.oof_pred[va]=prob
            s = self._metric(yva, prob)
            scores.append(s)
            class_print(self, f"Fold {fold}/{self.n_splits} {self.metric}={s:.5f}")
        
        self.cv_score_=float(np.mean(scores))
        class_print(self, f"CV 완료 mean={self.cv_score_:.5f} std={np.std(scores):.5f}")
        CompetitionMetric(verbose=True).eval(y, self.oof_pred)
        
        if refit_full:
            class_print(self, "전체 데이터 재학습")
            self.full_model_ = self.factory.create(self.model_name, p)
            self.full_model_.fit(X,y)
            class_print(self, "재학습 완료")
            print_sys_usage("after refit")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        m = self.full_model_ or (self.models_[-1] if self.models_ else None)
        assert m is not None, "fit_cv 먼저 호출하세요"
        return m.predict_proba(X[self.features_])[:,1] if hasattr(m, "predict_proba") else m.predict(X[self.features_])
    
    def save(self, path:str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    @classmethod
    def load(cls, path:str) -> "Trainer":
        pass
    
    def tune(self, X:pd.DataFrame, y:pd.Series, n_trials:int=40):
        pass