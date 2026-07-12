"""End-to-end reproduction: regenerate the manuscript's Table 1 (discrimination), Table 2
(operating points), Table 3 (comparison to the benchmark), and the SHAP figures.

Imaging arm = RadFusion-provided vision-model score (Vision.csv `pred`). EHR arm = re-implemented
elastic-net logistic (Table 1) with a support-vector variant (Table 2 operating point). Fusion =
average / maximum of the imaging and EHR probabilities.

This script does not train an imaging model; the exploratory YOLOv8n-cls arm lives in
src/exploratory_yolo.py and is intentionally not part of this pipeline.
"""
from __future__ import annotations
import json
import numpy as np

from src import config as C
from src import data as D
from src import models as M
from src import fusion as F
from src import metrics as MET
from src import interpret as I

BENCHMARK_AUROC = 0.947  # Huang et al. 2020, best multimodal model (PENet imaging arm), for Table 3


def _subset_mask(df, non_subsegmental: bool):
    """Full cohort, or the non-subsegmental subset (exclude studies whose PE is subsegmental only)."""
    if not non_subsegmental:
        return np.ones(len(df), dtype=bool)
    pe_type = df[C.PE_TYPE_COL].astype(str).str.lower()
    is_subsegmental_only = pe_type.eq("subsegmental")
    return (~is_subsegmental_only).to_numpy()


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = D.assemble_dataset()
    feats = D.feature_columns(df)

    tr = df[df[C.SPLIT_COL] == "train"]
    te = df[df[C.SPLIT_COL] == "test"]
    Xtr, ytr = tr[feats].to_numpy(), tr[C.LABEL_COL].to_numpy()
    Xte, yte = te[feats].to_numpy(), te[C.LABEL_COL].to_numpy()
    print(f"train={len(tr)}  test={len(te)}  test prevalence={yte.mean():.3f}  features={len(feats)}")

    # ---- EHR arms ----
    enet = M.fit_elastic_net(Xtr, ytr)
    svc = M.fit_svc(Xtr, ytr)
    emr_prob = enet.predict_proba(Xte)          # elastic-net -> Table 1 EHR discrimination
    emr_prob_svc = svc.predict_proba(Xte)       # SVC        -> Table 2 EHR operating point
    vision_prob = te[C.VISION_PRED_COL].to_numpy()  # PROVIDED PENet score (imaging arm)

    # ---- late fusion ----
    fuse_avg = F.average_fusion(vision_prob, emr_prob)
    fuse_max = F.maximum_fusion(vision_prob, emr_prob)

    # ================= Table 1: discrimination (AUROC / AUPRC) =================
    table1 = {}
    for label, non_sub in [("full cohort", False), ("non-subsegmental", True)]:
        m = _subset_mask(te, non_sub)
        table1[label] = {
            "imaging (provided PENet)": MET.discrimination(yte[m], vision_prob[m]),
            "EHR (elastic-net)": MET.discrimination(yte[m], emr_prob[m]),
            "late fusion (average)": MET.discrimination(yte[m], fuse_avg[m]),
            "late fusion (maximum)": MET.discrimination(yte[m], fuse_max[m]),
        }
    print("\n=== Table 1: AUROC / AUPRC by modality and analysis ===")
    print(json.dumps(table1, indent=2))

    # ================= Table 2: operating points (fixed 0.5 threshold) =================
    op_ehr_svc = MET.operating_point(yte, emr_prob_svc, C.THRESHOLD)   # specificity-favoring
    op_fuse_max = MET.operating_point(yte, fuse_max, C.THRESHOLD)      # sensitivity-favoring
    print("\n=== Table 2: operating-point metrics (Wilson 95% CI), fixed 0.5 threshold ===")
    for name, op in [("EHR (SVC)", op_ehr_svc), ("Late fusion (max)", op_fuse_max)]:
        print(f"\n{name}  [TP={op.tp} FP={op.fp} FN={op.fn} TN={op.tn}, n={op.tp+op.fp+op.fn+op.tn}]")
        print(f"  sensitivity {MET.fmt_ci(op.sensitivity)}   specificity {MET.fmt_ci(op.specificity)}")
        print(f"  PPV {MET.fmt_ci(op.ppv)}   NPV {MET.fmt_ci(op.npv)}")
        print(f"  accuracy {MET.fmt_ci(op.accuracy)}   F1 {op.f1:.2f}")
        print(f"  LR+ {op.lr_pos:.1f} ({op.lr_pos_ci[0]:.1f}–{op.lr_pos_ci[1]:.1f})   "
              f"LR- {op.lr_neg:.2f} ({op.lr_neg_ci[0]:.2f}–{op.lr_neg_ci[1]:.2f})")

    # ================= Table 3: comparison to the benchmark =================
    this_work_auroc = table1["full cohort"]["late fusion (average)"]["auroc"]
    print("\n=== Table 3: reproduction vs benchmark (full held-out test) ===")
    print(f"  Huang et al. 2020 (PENet imaging arm): AUROC {BENCHMARK_AUROC:.3f}")
    print(f"  This work (average late fusion, provided PENet scores + re-implemented EHR): "
          f"AUROC {this_work_auroc:.2f}")

    # ================= SHAP (EHR arm) =================
    try:
        I.explain_ehr(enet.estimator, Xtr, Xte, feats, C.RESULTS_DIR)
    except Exception as e:  # SHAP is optional; don't fail the whole run
        print(f"[warn] SHAP step skipped: {e}")

    # persist machine-readable results
    with open(C.RESULTS_DIR / "results.json", "w") as fh:
        json.dump({
            "table1": table1,
            "table2": {"EHR_SVC": op_ehr_svc.as_dict(), "fusion_max": op_fuse_max.as_dict()},
            "table3": {"benchmark_auroc": BENCHMARK_AUROC, "this_work_auroc": this_work_auroc},
        }, fh, indent=2)
    print(f"\nWrote {C.RESULTS_DIR/'results.json'} and SHAP plots to {C.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
