"""Evaluation metrics: discrimination (AUROC, AUPRC) and threshold-based operating-point metrics with
Wilson 95% confidence intervals and likelihood ratios (log-method intervals).

All confidence intervals here are computed in closed form; no bootstrapping is performed. AUROC is
reported as a point estimate (the manuscript does not claim CIs or a formal between-arm test).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

Z = 1.959963984540054  # 95% normal quantile


# --------------------------------------------------------------------------------------------------
# interval helpers
# --------------------------------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _lr_ci(point: float, log_se: float, z: float = Z) -> tuple[float, float]:
    return (point * math.exp(-z * log_se), point * math.exp(z * log_se))


def likelihood_ratios(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Positive/negative likelihood ratios with log-method 95% CIs (Simel et al., 1991)."""
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    out = {}
    # LR+ = sens / (1 - spec)
    if 0 < spec < 1 and sens > 0:
        lr_pos = sens / (1 - spec)
        se = math.sqrt((1 - sens) / tp + spec / fp) if tp and fp else float("nan")
        out["lr_pos"], out["lr_pos_ci"] = lr_pos, _lr_ci(lr_pos, se)
    else:
        out["lr_pos"], out["lr_pos_ci"] = float("nan"), (float("nan"), float("nan"))
    # LR- = (1 - sens) / spec
    if spec > 0 and 0 < sens < 1:
        lr_neg = (1 - sens) / spec
        se = math.sqrt(sens / fn + (1 - spec) / tn) if fn and tn else float("nan")
        out["lr_neg"], out["lr_neg_ci"] = lr_neg, _lr_ci(lr_neg, se)
    else:
        out["lr_neg"], out["lr_neg_ci"] = float("nan"), (float("nan"), float("nan"))
    return out


# --------------------------------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------------------------------
def discrimination(y_true, scores) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


# --------------------------------------------------------------------------------------------------
# operating point
# --------------------------------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    tp: int
    fp: int
    fn: int
    tn: int
    sensitivity: tuple
    specificity: tuple
    ppv: tuple
    npv: tuple
    accuracy: tuple
    f1: float
    lr_pos: float
    lr_pos_ci: tuple
    lr_neg: float
    lr_neg_ci: tuple

    def as_dict(self):
        return asdict(self)


def _prop(k, n):
    return (k / n if n else float("nan"), *wilson_ci(k, n))


def operating_point(y_true, scores, threshold: float) -> OperatingPoint:
    """Confusion-matrix-derived metrics at a fixed probability threshold, with Wilson CIs."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)

    sens = _prop(tp, tp + fn)
    spec = _prop(tn, tn + fp)
    ppv = _prop(tp, tp + fp)
    npv = _prop(tn, tn + fn)
    acc = _prop(tp + tn, tp + fp + fn + tn)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    lr = likelihood_ratios(tp, fp, fn, tn)

    return OperatingPoint(
        tp=tp, fp=fp, fn=fn, tn=tn,
        sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, accuracy=acc, f1=f1,
        lr_pos=lr["lr_pos"], lr_pos_ci=lr["lr_pos_ci"],
        lr_neg=lr["lr_neg"], lr_neg_ci=lr["lr_neg_ci"],
    )


def fmt_ci(triple) -> str:
    p, lo, hi = triple
    return f"{p:.2f} ({lo:.2f}–{hi:.2f})"
