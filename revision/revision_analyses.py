"""Revision analyses for the Cureus RadFusion reproduction manuscript.

Reproduces the reported numbers with the configuration that generated them (the RadFusion reference
tutorial configuration), then adds everything the editors asked for:
  * cohort table (n, % PE-positive, PE subtype) per split
  * preprocessing feature counts
  * Table 1 AUROC/AUPRC with stratified bootstrap 95% CIs (incl. max fusion on non-subsegmental)
  * paired DeLong tests between arms
  * Table 2 operating points with n/N
  * label-leakage ablation: refit the EHR arm without prespecified feature groups
  * SHAP (exact linear SHAP) ranking with shares by leakage group
  * descriptive prevalence of leakage-candidate features by label
  * sensitivity analyses (train-only standardization; cross-validated grid search)
  * Figures 2-5 and S1 without embedded titles
Run:  python revision/revision_analyses.py   (expects the RadFusion CSVs in ./data)
"""
from __future__ import annotations
import json, platform, datetime, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED = 42
N_BOOT = 2000
THRESH = 0.5
EHR_FILES = ["Demographics.csv", "OUT_MED.csv", "INP_MED.csv", "ICD.csv", "LABS.csv", "Vitals.csv"]

# ---------------------------------------------------------------------------------------------
# prespecified leakage-candidate feature groups (exact column stems; inpatient + outpatient)
# ---------------------------------------------------------------------------------------------
ANTICOAG_STEMS = [
    "HEPARIN AND RELATED PREPARATIONS",
    "ANTICOAGULANTS,COUMARIN TYPE",
    "DIRECT FACTOR XA INHIBITORS",
    "THROMBIN INHIBITORS, SELECTIVE, DIRECT, REVERSIBLE",
    "THROMBIN INHIBITORS,SEL,DIRECT,REVERS-HIRUDIN TYPE",
    "THROMBOLYTIC ENZYMES",
]


def leak_group(col: str) -> str | None:
    c = col.replace("Outpatient_", "")
    stem = c.rsplit(":", 1)[0]
    if stem == "DISEASES OF PULMONARY CIRCULATION":
        return "Pulmonary-circulation diagnoses"
    if stem in ANTICOAG_STEMS:
        return "Anticoagulant/thrombolytic medications"
    if stem == "d-dimer":
        return "D-dimer testing"
    if stem == "DISEASES OF VEINS AND LYMPHATICS, AND OTHER DISEASES OF CIRCULATORY SYSTEM":
        return "Venous-disease diagnoses"
    return None


GROUP_ORDER = ["Pulmonary-circulation diagnoses", "Anticoagulant/thrombolytic medications",
               "D-dimer testing", "Venous-disease diagnoses"]


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load(train_only_scaling: bool = False):
    vision = pd.read_csv(DATA / "Vision.csv").drop(columns=["Unnamed: 0"], errors="ignore").set_index("idx")
    counts = {}
    proc = []
    for f in EHR_FILES:
        df = pd.read_csv(DATA / f)
        raw_rows = len(df)
        df = df.drop_duplicates("idx")  # 51 studies appear twice with byte-identical rows
        df = df.drop(columns=[c for c in df.columns if c == "split" or c.startswith("Unnamed")]).set_index("idx")
        n_raw = df.shape[1]
        df = df.loc[:, df.nunique() > 1]  # zero-variance removal
        n_kept = df.shape[1]
        if train_only_scaling:
            tr = df.loc[df.index.intersection(vision.index[vision.split == "train"])]
            mu, sd = tr.mean(), tr.std().replace(0, 1.0)
        else:
            mu, sd = df.mean(), df.std()
        df = (df - mu) / sd
        counts[f.replace(".csv", "")] = {"rows_in_file": raw_rows, "studies": len(df),
                                         "features_raw": n_raw, "features_after_zero_variance": n_kept}
        proc.append(df)
    emr = pd.concat(proc, axis=1)
    missing = int(emr.isna().sum().sum())
    emr = emr.fillna(0.0)
    df = vision.join(emr, how="inner")
    return df, list(emr.columns), counts, missing


def enet():
    return LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.99,
                              class_weight="balanced", max_iter=1000, random_state=0)


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, preds):
    """Fast DeLong (Sun & Xu 2014). Returns aucs, covariance matrix."""
    y = np.asarray(y); preds = np.atleast_2d(preds)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(p) for p in pos]); ty = np.array([_midrank(p) for p in neg])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, s


def delong_test(y, a, b):
    aucs, s = delong(y, np.vstack([a, b]))
    diff = aucs[0] - aucs[1]
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(var)
    return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff,
            "diff_ci": (diff - 1.96 * se, diff + 1.96 * se), "z": z, "p": p}


def delong_ci(y, a):
    aucs, s = delong(y, a[None, :])
    se = np.sqrt(s[0, 0])
    return aucs[0], aucs[0] - 1.96 * se, min(1.0, aucs[0] + 1.96 * se), se


def boot_ci(y, score, fn=roc_auc_score, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        vals.append(fn(y[idx], score[idx]))
    vals = np.array(vals)
    return fn(y, score), np.percentile(vals, 2.5), np.percentile(vals, 97.5), vals.std(ddof=1)


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0, c - h), min(1, c + h)


def op_point(y, s, t=THRESH):
    yp = (np.asarray(s) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    lrp = sens / (1 - spec); lrn = (1 - sens) / spec
    se_p = np.sqrt((1 - sens) / tp + spec / fp); se_n = np.sqrt(sens / fn + (1 - spec) / tn)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Sensitivity": (tp, tp + fn, *wilson(tp, tp + fn)),
        "Specificity": (tn, tn + fp, *wilson(tn, tn + fp)),
        "PPV": (tp, tp + fp, *wilson(tp, tp + fp)),
        "NPV": (tn, tn + fn, *wilson(tn, tn + fn)),
        "Accuracy": (tp + tn, tp + fp + fn + tn, *wilson(tp + tn, tp + fp + fn + tn)),
        "F1": 2 * tp / (2 * tp + fp + fn),
        "LR+": (lrp, lrp * np.exp(-1.96 * se_p), lrp * np.exp(1.96 * se_p)),
        "LR-": (lrn, lrn * np.exp(-1.96 * se_n), lrn * np.exp(1.96 * se_n)),
    }


def fmt(p, lo, hi, d=2):
    return f"{p:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------------------------
# plotting helpers (no embedded titles)
# ---------------------------------------------------------------------------------------------
COLORS = {"Imaging (provided PENet score)": "#E08A1E", "EHR (elastic-net)": "#C8423B",
          "Late fusion (average)": "#2F6DB3", "Late fusion (maximum)": "#5B9A5B"}


def roc_figure(y, arms: dict, path: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=300)
    for name, s in arms.items():
        fpr, tpr, _ = roc_curve(y, s)
        a, lo, hi, _ = boot_ci(y, s)
        ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name}: AUROC {a:.2f} ({lo:.2f}–{hi:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.5")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate (1 − specificity)"); ax.set_ylabel("True-positive rate (sensitivity)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ---------------------------------------------------------------------------------------------
def main():
    run_date = datetime.date.today().isoformat()
    versions = {"Python": platform.python_version(), "NumPy": np.__version__, "pandas": pd.__version__,
                "SciPy": scipy.__version__, "scikit-learn": sklearn.__version__, "SHAP": shap.__version__,
                "matplotlib": matplotlib.__version__, "run_date": run_date}
    R = {"versions": versions}

    df, F, counts, n_missing = load()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    R["preprocessing"] = {"per_table": counts, "total_features": len(F), "missing_values_imputed": n_missing,
                          "studies_joined": len(df)}

    # ---- cohort table
    cohort = []
    for name, d in [("Training", tr), ("Validation", va), ("Test", te), ("Total", df)]:
        n = len(d); pos = int(d.label.sum())
        pt = d.pe_type.fillna("none").value_counts()
        cohort.append({"Split": name, "Studies, n": n,
                       "PE-positive, n (%)": f"{pos} ({100*pos/n:.1f})",
                       "PE-negative, n (%)": f"{n-pos} ({100*(n-pos)/n:.1f})",
                       "Central, n (%)": f"{pt.get('central',0)} ({100*pt.get('central',0)/n:.1f})",
                       "Segmental, n (%)": f"{pt.get('segmental',0)} ({100*pt.get('segmental',0)/n:.1f})",
                       "Subsegmental, n (%)": f"{pt.get('subsegmental',0)} ({100*pt.get('subsegmental',0)/n:.1f})"})
    cohort = pd.DataFrame(cohort); cohort.to_csv(OUT / "table_cohort.csv", index=False)
    R["cohort"] = cohort.to_dict("records")

    Xtr, ytr, Xte, yte = tr[F].values, tr.label.values, te[F].values, te.label.values
    vis = te.pred.values

    # ---- primary EHR model (reference configuration)
    clf = enet().fit(Xtr, ytr)
    ehr = clf.predict_proba(Xte)[:, 1]
    avg, mx = (ehr + vis) / 2, np.maximum(ehr, vis)
    R["model"] = {"elastic_net": {"C": 0.1, "l1_ratio": 0.99, "class_weight": "balanced", "solver": "saga",
                                  "max_iter": 1000, "nonzero_coefficients": int((clf.coef_ != 0).sum())}}

    # ---- SVC (Table 2)
    svc = SVC(probability=True, random_state=0).fit(Xtr, ytr)
    ehr_svc = svc.predict_proba(Xte)[:, 1]
    R["model"]["svc"] = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None, "probability": True}

    # ---- Table 1
    nonsub = (te.pe_type != "subsegmental").values
    arms = {"Imaging (provided PENet score)": vis, "EHR (elastic-net)": ehr,
            "Late fusion (average)": avg, "Late fusion (maximum)": mx}
    t1 = []
    for sub, m in [("Full test cohort", np.ones(len(te), bool)), ("Non-subsegmental subset", nonsub)]:
        y = yte[m]
        for name, s in arms.items():
            a, lo, hi, _ = boot_ci(y, s[m])
            ap, alo, ahi, _ = boot_ci(y, s[m], fn=average_precision_score)
            da, dlo, dhi, _ = delong_ci(y, s[m])
            t1.append({"Analysis": sub, "n (PE+/PE−)": f"{m.sum()} ({int(y.sum())}/{int((1-y).sum())})",
                       "Arm": name, "AUROC": a, "AUROC_lo": lo, "AUROC_hi": hi,
                       "AUROC_DeLong_lo": dlo, "AUROC_DeLong_hi": dhi,
                       "AUPRC": ap, "AUPRC_lo": alo, "AUPRC_hi": ahi})
    t1 = pd.DataFrame(t1); t1.to_csv(OUT / "table1_discrimination.csv", index=False)
    R["table1"] = t1.to_dict("records")

    # ---- DeLong paired tests
    pairs = [("Late fusion (average)", "EHR (elastic-net)"), ("Late fusion (average)", "Imaging (provided PENet score)"),
             ("EHR (elastic-net)", "Imaging (provided PENet score)"), ("Late fusion (maximum)", "EHR (elastic-net)"),
             ("Late fusion (average)", "Late fusion (maximum)"), ("Late fusion (maximum)", "Imaging (provided PENet score)")]
    dl = []
    for sub, m in [("Full test cohort", np.ones(len(te), bool)), ("Non-subsegmental subset", nonsub)]:
        for a, b in pairs:
            r = delong_test(yte[m], arms[a][m], arms[b][m])
            dl.append({"Analysis": sub, "Comparison": f"{a} vs {b}", "AUROC A": r["auc_a"], "AUROC B": r["auc_b"],
                       "Difference": r["diff"], "Diff_lo": r["diff_ci"][0], "Diff_hi": r["diff_ci"][1],
                       "z": r["z"], "p": r["p"]})
    dl = pd.DataFrame(dl); dl.to_csv(OUT / "delong_tests.csv", index=False)
    R["delong"] = dl.to_dict("records")

    # bootstrap-vs-DeLong SE agreement (check)
    R["se_check"] = {name: {"bootstrap_se": boot_ci(yte, s)[3], "delong_se": delong_ci(yte, s)[3]}
                     for name, s in arms.items()}

    # ---- Table 2
    ops = {"EHR (SVC)": op_point(yte, ehr_svc), "Late fusion (maximum)": op_point(yte, mx)}
    R["table2"] = ops

    # ---- leakage feature map & descriptive prevalence
    groups = {g: [c for c in F if leak_group(c) == g] for g in GROUP_ORDER}
    fmap = [{"feature": c, "group": g} for g, cs in groups.items() for c in cs]
    pd.DataFrame(fmap).to_csv(OUT / "leakage_feature_map.csv", index=False)
    raw = {}
    for f in ["ICD.csv", "INP_MED.csv", "OUT_MED.csv", "LABS.csv"]:
        d = pd.read_csv(DATA / f).drop_duplicates("idx").set_index("idx"); raw[f] = d
    rawall = pd.concat([raw[f].drop(columns=[c for c in raw[f].columns if c == "split" or c.startswith("Unnamed")])
                        for f in raw], axis=1)
    rawall = rawall.loc[:, ~rawall.columns.duplicated()]
    prev = []
    presence_cols = {
        "Pulmonary-circulation diagnosis (presence)": ["DISEASES OF PULMONARY CIRCULATION:presence"],
        "Any anticoagulant/thrombolytic (inpatient or outpatient)": [c for c in rawall.columns if leak_group(c) == "Anticoagulant/thrombolytic medications" and c.endswith("Binary")],
        "D-dimer measured": ["d-dimer:Binary"],
        "Venous-disease diagnosis (presence)": ["DISEASES OF VEINS AND LYMPHATICS, AND OTHER DISEASES OF CIRCULATORY SYSTEM:presence"],
    }
    lab = df.label
    for name, cols in presence_cols.items():
        flag = (rawall.loc[df.index, cols].fillna(0) > 0).any(axis=1)
        for sname, idxs in [("Training", tr.index), ("Test", te.index)]:
            f_, l_ = flag.loc[idxs], lab.loc[idxs]
            a, b = int(f_[l_ == 1].sum()), int((l_ == 1).sum())
            c_, d_ = int(f_[l_ == 0].sum()), int((l_ == 0).sum())
            orr, p = stats.fisher_exact([[a, b - a], [c_, d_ - c_]])
            prev.append({"Feature": name, "Split": sname, "PE-positive, n/N (%)": f"{a}/{b} ({100*a/b:.1f})",
                         "PE-negative, n/N (%)": f"{c_}/{d_} ({100*c_/d_:.1f})", "Odds ratio": orr, "Fisher p": p})
    prev = pd.DataFrame(prev); prev.to_csv(OUT / "leakage_feature_prevalence.csv", index=False)
    R["prevalence"] = prev.to_dict("records")

    # ---- SHAP (exact linear SHAP, interventional, log-odds; background = training set)
    expl = shap.LinearExplainer(clf, Xtr)
    sv = expl(Xte); sv.feature_names = F
    mean_abs = np.abs(sv.values).mean(axis=0)
    total = mean_abs.sum()
    order = np.argsort(mean_abs)[::-1]
    shap_rank = pd.DataFrame({"rank": np.arange(1, len(F) + 1), "feature": np.array(F)[order],
                              "mean_abs_shap": mean_abs[order], "share_of_total_pct": 100 * mean_abs[order] / total,
                              "leakage_group": [leak_group(F[i]) or "" for i in order]})
    shap_rank["cumulative_share_pct"] = shap_rank.share_of_total_pct.cumsum()
    shap_rank.to_csv(OUT / "shap_ranking.csv", index=False)
    grp_share = {g: float(100 * mean_abs[[F.index(c) for c in cs]].sum() / total) for g, cs in groups.items()}
    R["shap"] = {"top20": shap_rank.head(20).to_dict("records"), "group_share_pct": grp_share,
                 "n_nonzero_features": int((mean_abs > 0).sum()), "total_mean_abs": float(total)}

    # ---- leakage ablation
    abl_specs = [("Full model (reference)", [])] + [(f"Without {g[0].lower()+g[1:]}", groups[g]) for g in GROUP_ORDER[:3]] + [
        ("Without all three prespecified groups", sum((groups[g] for g in GROUP_ORDER[:3]), [])),
        ("Without all three + venous-disease diagnoses", sum((groups[g] for g in GROUP_ORDER), []))]
    abl = []; abl_preds = {}
    for name, drop in abl_specs:
        keep = [c for c in F if c not in drop]
        ki = [F.index(c) for c in keep]
        m_ = enet().fit(Xtr[:, ki], ytr)
        e_ = m_.predict_proba(Xte[:, ki])[:, 1]
        a_ = (e_ + vis) / 2
        abl_preds[name] = (e_, a_)
        row = {"Model": name, "Features removed": len(drop), "Features used": len(keep)}
        for arm, s, ref in [("EHR", e_, ehr), ("Average fusion", a_, avg)]:
            a, lo, hi, _ = boot_ci(yte, s)
            row[f"{arm} AUROC"] = a; row[f"{arm} lo"] = lo; row[f"{arm} hi"] = hi
            if drop:
                t = delong_test(yte, s, ref)
                row[f"{arm} Δ vs full"] = t["diff"]; row[f"{arm} Δ lo"] = t["diff_ci"][0]
                row[f"{arm} Δ hi"] = t["diff_ci"][1]; row[f"{arm} p"] = t["p"]
        # non-subsegmental too
        row["EHR AUROC nonsub"] = roc_auc_score(yte[nonsub], e_[nonsub])
        row["Fusion AUROC nonsub"] = roc_auc_score(yte[nonsub], a_[nonsub])
        abl.append(row)
    # compare ablated fusion vs imaging alone
    for row in abl:
        e_, a_ = abl_preds[row["Model"]]
        t = delong_test(yte, a_, vis)
        row["Fusion vs imaging Δ"] = t["diff"]; row["Fusion vs imaging p"] = t["p"]
    abl = pd.DataFrame(abl); abl.to_csv(OUT / "table4_leakage_ablation.csv", index=False)
    R["ablation"] = abl.to_dict("records")

    # ---- sensitivity analyses
    sens = []
    df2, F2, _, _ = load(train_only_scaling=True)
    tr2, te2 = df2[df2.split == "train"], df2[df2.split == "test"]
    c2 = enet().fit(tr2[F2].values, tr2.label.values)
    e2 = c2.predict_proba(te2[F2].values)[:, 1]
    sens.append({"Analysis": "Standardization fitted on training split only",
                 "EHR AUROC": roc_auc_score(te2.label, e2),
                 "Average fusion AUROC": roc_auc_score(te2.label, (e2 + te2.pred.values) / 2)})
    # cross-validated grid search on train (5-fold, ROC-AUC)
    gs = GridSearchCV(LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                                         max_iter=2000, random_state=0),
                      {"C": [0.01, 0.1, 1.0], "l1_ratio": [0.5, 0.9, 0.99]}, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1).fit(Xtr, ytr)
    e3 = gs.best_estimator_.predict_proba(Xte)[:, 1]
    sens.append({"Analysis": f"Elastic-net tuned by 5-fold CV grid search (selected C={gs.best_params_['C']}, l1_ratio={gs.best_params_['l1_ratio']})",
                 "EHR AUROC": roc_auc_score(yte, e3), "Average fusion AUROC": roc_auc_score(yte, (e3 + vis) / 2)})
    R["grid_search"] = {"best_params": gs.best_params_, "best_cv_auroc": gs.best_score_,
                        "cv_results": [{"C": p["C"], "l1_ratio": p["l1_ratio"], "mean_cv_auroc": s}
                                       for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])]}
    # grid-search on the fully ablated feature set
    keep = [i for i, c in enumerate(F) if leak_group(c) not in GROUP_ORDER[:3]]
    gs2 = GridSearchCV(LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                                          max_iter=2000, random_state=0),
                       {"C": [0.01, 0.1, 1.0], "l1_ratio": [0.5, 0.9, 0.99]}, scoring="roc_auc",
                       cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1).fit(Xtr[:, keep], ytr)
    e4 = gs2.best_estimator_.predict_proba(Xte[:, keep])[:, 1]
    sens.append({"Analysis": f"Grid-searched elastic-net without all three prespecified groups (C={gs2.best_params_['C']}, l1_ratio={gs2.best_params_['l1_ratio']})",
                 "EHR AUROC": roc_auc_score(yte, e4), "Average fusion AUROC": roc_auc_score(yte, (e4 + vis) / 2)})
    # SVC discrimination for transparency
    sens.append({"Analysis": "SVC EHR estimator (Table 2 operating point model)", "EHR AUROC": roc_auc_score(yte, ehr_svc),
                 "Average fusion AUROC": roc_auc_score(yte, (ehr_svc + vis) / 2)})
    sens = pd.DataFrame(sens); sens.to_csv(OUT / "sensitivity_analyses.csv", index=False)
    R["sensitivity"] = sens.to_dict("records")

    # ---- figures
    roc_figure(yte, arms, FIG / "Figure_2_ROC_full_test_cohort.png")
    roc_figure(yte[nonsub], {k: v[nonsub] for k, v in arms.items()}, FIG / "Figure_3_ROC_nonsubsegmental.png")

    plt.rcParams.update({"font.size": 10})
    shap.plots.beeswarm(sv, max_display=20, show=False, plot_size=(11, 7.5))
    plt.gcf().savefig(FIG / "Figure_4_SHAP_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close("all")

    # Figure 5: ablation forest plot
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=300)
    ypos = np.arange(len(abl))[::-1]
    for off, arm, col in [(0.15, "EHR", "#C8423B"), (-0.15, "Average fusion", "#2F6DB3")]:
        ax.errorbar(abl[f"{arm} AUROC"], ypos + off,
                    xerr=[abl[f"{arm} AUROC"] - abl[f"{arm} lo"], abl[f"{arm} hi"] - abl[f"{arm} AUROC"]],
                    fmt="o", color=col, capsize=3, label=f"{arm} arm" if arm == "EHR" else "Average late fusion")
    a_img, lo_img, hi_img, _ = boot_ci(yte, vis)
    ax.axvspan(lo_img, hi_img, color="#E08A1E", alpha=0.15, label=f"Imaging alone (AUROC {a_img:.2f}, 95% CI)")
    ax.axvline(a_img, color="#E08A1E", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(abl.Model); ax.set_xlabel("AUROC (95% bootstrap CI), full test cohort")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "Figure_5_leakage_ablation.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Figure S1: global mean |SHAP| (top 30) with cumulative share
    top = shap_rank.head(30)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8.5), dpi=300)
    cols = ["#C8423B" if g else "#6B7C93" for g in top.leakage_group]
    ax.barh(top.feature, top.mean_abs_shap, color=cols)
    ax.set_xlabel("Mean |SHAP value| on the test set (log-odds units)")
    ax.tick_params(axis="y", labelsize=7.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C8423B", label="Prespecified leakage-candidate group"),
                       Patch(color="#6B7C93", label="Other feature")], loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "Figure_S1_global_SHAP_importance.png", dpi=300); plt.close(fig)

    with open(OUT / "results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: R[k] for k in ["versions", "preprocessing", "model"]}, indent=1, default=str))
    print(t1.round(3).to_string()); print(dl.round(4).to_string())
    for k, v in ops.items(): print(k, v)
    print(prev.to_string()); print(shap_rank.head(20).round(4).to_string()); print(grp_share)
    print(abl.round(3).T.to_string()); print(sens.round(3).to_string()); print(R["se_check"])


if __name__ == "__main__":
    main()
