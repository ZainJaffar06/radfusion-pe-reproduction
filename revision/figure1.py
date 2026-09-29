"""Figure 1: pipeline overview (no embedded title)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "Figure_1_pipeline_overview.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, GREY, ORANGE = ("#DCE6F2", "#4A74B4"), ("#E2F0E0", "#5A9A5A"), ("#EAEAEA", "#8A8A8A"), ("#FBE6D4", "#D9822B")

fig, ax = plt.subplots(figsize=(12, 5.2), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")


def box(x, y, w, h, text, col, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor=col[0], edgecolor=col[1], lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))


box(1, 31, 18, 9, "CTPA imaging\n(RadFusion)", BLUE)
box(24, 31, 22, 9, "Provided study-level\nPENet score\n(reused, not retrained)", GREY)
box(1, 17, 18, 9, "Structured EHR\n(RadFusion; 6 tables,\n1,505 features)", GREEN)
box(24, 17, 22, 9, "Elastic-net logistic\nmodel (re-implemented)", GREEN)
box(52, 24, 19, 9, "Late fusion\n(average / maximum)", BLUE)
box(77, 24, 21, 9, "Study-level\nPE prediction", BLUE)
box(24, 2, 74, 9, "Label-leakage audit: SHAP attribution  →  prevalence of candidate features by PE status  →\n"
    "feature-exclusion ablation (refit EHR model, recompute fusion, DeLong tests)", ORANGE, fs=10)
arrow(19.3, 35.5, 23.7, 35.5); arrow(19.3, 21.5, 23.7, 21.5)
arrow(46.3, 34, 51.7, 30.5); arrow(46.3, 23, 51.7, 26.5); arrow(71.3, 28.5, 76.7, 28.5)
arrow(35, 16.7, 35, 11.3)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(OUT)
