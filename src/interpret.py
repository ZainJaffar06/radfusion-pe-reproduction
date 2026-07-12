"""SHAP attribution for the EHR arm.

The EHR model is a linear (elastic-net logistic) estimator, so SHAP recovers — rather than extends —
the standardized model coefficients; the attribution is reported for transparency and to audit which
features drive the prediction.

AUDIT NOTE: in RadFusion the dominant EHR feature is a pulmonary-circulation diagnosis (ICD-10
I26–I28) whose coding overlaps the PE label itself. A high-ranking such feature is consistent with
temporal label leakage rather than genuine pre-test signal. This script prints the top features so
that overlap can be inspected directly.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def explain_ehr(model, X_background, X_explain, feature_names, out_dir: Path, top_k: int = 20):
    """Compute SHAP values for the EHR pipeline and save summary + beeswarm plots.

    `model` is the fitted sklearn Pipeline (scaler + linear clf); we wrap its `predict_proba` so SHAP
    explains the same probability that feeds the fusion.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    f = lambda data: model.predict_proba(data)[:, 1]
    # KernelExplainer/Explainer works for arbitrary pipelines; a modest background keeps it tractable.
    background = shap.sample(X_background, min(100, len(X_background)), random_state=0)
    explainer = shap.Explainer(f, background, feature_names=list(feature_names))
    shap_values = explainer(X_explain)

    # global importance (heatmap-style summary)
    plt.figure()
    shap.plots.bar(shap_values, max_display=top_k, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_importance.png", dpi=300); plt.close()

    # beeswarm (direction + value dependence)
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=top_k, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_beeswarm.png", dpi=300); plt.close()

    # print ranked features so label-overlap can be inspected
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    print("\nTop EHR SHAP features (inspect for overlap with the PE label / ICD-10 I26–I28):")
    for rank, i in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_names[i]:<45s} mean|SHAP|={mean_abs[i]:.4f}")
    return shap_values
