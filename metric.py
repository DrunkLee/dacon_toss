from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Any
from sklearn.metrics import average_precision_score
import numpy as np
import cupy as cp
import warnings

@dataclass(frozen=True)
class Metrics:
    score: float
    ap: float
    wll: float
    
class CompetitionMetric:
    '''대회 메트릭 계산 클래스'''
    def __init__(self,
                 eps: float = 1e-18, 
                 class_weights: Tuple[float, float] = (0.5, 0.5),
                 verbose: bool = True
                 ):
        self.eps = float(eps)
        self.w0, self.w1 = map(float, class_weights)
        self.verbose = verbose
    
    def _to_numpy_1d(self, arr: Any) -> np.ndarray:
        '''입력으로 들어온 배열을 1D numpy ndarray로 변환'''
        if cp is not None and isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
        if hasattr(arr,"values"):
            arr = arr.values
        a = np.asarray(arr)
        if a.ndim != 1:
            a = a.reshape(-1)
        return a.astype(np.float64, copy=False)
    
    def _compute_ap(self, y_true_np: np.ndarray, y_pred_np: np.ndarray) -> float:
        '''average precision 계산'''
        ap = float(average_precision_score(y_true_np, y_pred_np))
        return ap
    
    def _compute_wll(self, y_true_np: np.ndarray, y_pred_np: np.ndarray) -> float:
        '''weighted logloss 클래스별 50:50 기본'''
        mask_1 = (y_true_np == 1)
        mask_0 = ~mask_1
        ll_0 = -float(np.mean(np.log(1.0 - y_pred_np[mask_0])))
        ll_1 = -float(np.mean(np.log(y_pred_np[mask_1])))
        wll = self.w0 * ll_0 + self.w1 * ll_1
        return wll
    
    def eval(self, y_true: Any, y_pred_proba: Any) -> Metrics:
        '''true, pred를 받아 점수를 계산'''
        y_true_np = self._to_numpy_1d(y_true)
        y_pred_np = self._to_numpy_1d(y_pred_proba)
        
        y_pred_np = np.clop(y_pred_np, self.eps, 1.0 - self.eps)
        
        ap = self._compute_ap(y_true_np, y_pred_np)
        wll = self._compute_wll(y_true_np, y_pred_np)
        score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
        if self.verbose:
            pos_ratio = float(np.mean(y_true_np))
            print(f"[Metric] AP={ap:.6f} | WLL={wll:.6f} | Score={score:.6f} | "
                  f"PosRatio={pos_ratio:.6f}")
        return Metrics(score=score, ap=ap, wll=wll)
        
        