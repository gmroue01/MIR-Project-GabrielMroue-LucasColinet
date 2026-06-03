"""
Generate analysis plots for SIFT PCA results.
Run from project root: .venv/Scripts/python.exe notebook/SIFT_PCA_Analysis/generate_plots.py
"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = "notebook/SIFT_PCA_Analysis/results"

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "lines.linewidth": 2.2,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

cumvar      = np.load(f"{RESULTS_DIR}/cumvar.npy")
eigenvalues = np.load(f"{RESULTS_DIR}/eigenvalues.npy")
with open(f"{RESULTS_DIR}/matching_quality.json") as f:
    quality = json.load(f)

ORIG_SIZE  = 199.7
PCA_DIM    = 32
BASELINE   = quality["baseline_avg_inliers"]
dims_q     = sorted([int(k) for k in quality["dims"].keys()])
inliers    = [quality["dims"][str(d)]["avg_inliers"] for d in dims_q]
retention  = [quality["dims"][str(d)]["retention_pct"] for d in dims_q]


# ──────────────────────────────────────────────────────────────────────────────
# 01 — Variance expliquee
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ACP sur descripteurs SIFT 128D — Variance expliquee", fontsize=14, fontweight="bold")

comps = np.arange(1, len(cumvar) + 1)
ax1.plot(comps, cumvar * 100, color="#58a6ff", linewidth=2)
ax1.fill_between(comps, cumvar * 100, alpha=0.12, color="#58a6ff")

thresh_cfg = {
    0.80: ("#ffa657", 32),
    0.90: ("#3fb950", 53),
    0.95: ("#79c0ff", 74),
    0.99: ("#ff7b72", 108),
}
for t, (c, d) in thresh_cfg.items():
    ax1.axvline(d, linestyle="--", color=c, linewidth=1.0, alpha=0.85,
                label=f"{int(t*100)}% var -> {d}D")
    ax1.axhline(t * 100, linestyle="--", color=c, linewidth=0.5, alpha=0.3)

ax1.axvline(PCA_DIM, linestyle="-", color="#ffd700", linewidth=2.5, zorder=5,
            label=f"Choix retenu : {PCA_DIM}D")
val_at_32 = cumvar[PCA_DIM - 1] * 100
ax1.scatter([PCA_DIM], [val_at_32], color="#ffd700", s=90, zorder=6)
ax1.annotate(
    f"{val_at_32:.1f}% variance",
    (PCA_DIM, val_at_32),
    xytext=(PCA_DIM + 10, val_at_32 - 7),
    color="#ffd700", fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.2),
)
ax1.set_xlabel("Nombre de composantes ACP")
ax1.set_ylabel("Variance expliquee cumulee (%)")
ax1.set_title("Variance cumulee")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(True)
ax1.set_ylim(0, 101)
ax1.set_xlim(1, 128)

explained = eigenvalues / eigenvalues.sum() * 100
bars = ax2.bar(np.arange(1, 65), explained[:64], color="#58a6ff", alpha=0.6, width=0.85)
for i in range(PCA_DIM):
    bars[i].set_color("#ffd700")
    bars[i].set_alpha(0.85)
ax2.axvline(PCA_DIM + 0.5, linestyle="-", color="#ffd700", linewidth=2.2,
            label=f"Coupure a {PCA_DIM}D")
ax2.set_xlabel("Composante ACP")
ax2.set_ylabel("Variance expliquee (%)")
ax2.set_title("Variance par composante (top 64)")
ax2.legend(fontsize=9)
ax2.grid(True, axis="y")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_variance_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("01_variance_analysis.png saved")


# ──────────────────────────────────────────────────────────────────────────────
# 02 — Taille de l index
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Taille de l index SIFT compresse selon la dimension ACP", fontsize=13, fontweight="bold")

labels_b = ["128D\n(original)", "32D\n(retenu)", "48D", "64D"]
sizes_b  = [199.7, 146.2, 214.5, 283.0]
col_b    = ["#ff7b72", "#3fb950", "#8b949e", "#8b949e"]

bars = ax.bar(labels_b, sizes_b, color=col_b, alpha=0.82, width=0.5)
ax.axhline(ORIG_SIZE, linestyle="--", color="#ff7b72", linewidth=1.8,
           label=f"Original uint8 : {ORIG_SIZE} MB")

for bar, sz in zip(bars, sizes_b):
    saving = ORIG_SIZE - sz
    sign = "-" if saving > 0 else "+"
    label = f"{sz:.1f} MB" if saving == 0 else f"{sz:.1f} MB\n({sign}{abs(saving):.1f} MB)"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
            label, ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Taille compressee (MB)")
ax.set_title("")
ax.legend(fontsize=9)
ax.grid(True, axis="y")
ax.set_ylim(0, 330)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_index_sizes.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("02_index_sizes.png saved")


# ──────────────────────────────────────────────────────────────────────────────
# 03 — Qualite de matching
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Impact ACP sur la qualite de matching SIFT-RANSAC (30 paires meme classe)",
    fontsize=13, fontweight="bold",
)

ax1.plot(dims_q, inliers, color="#58a6ff", marker="o", markersize=7, zorder=3)
ax1.axhline(BASELINE, linestyle="--", color="#ff7b72", linewidth=1.8,
            label=f"Baseline 128D sans ACP : {BASELINE:.1f}")
ax1.axvline(PCA_DIM, linestyle="-", color="#ffd700", linewidth=2.5, zorder=5,
            label=f"Choix retenu : {PCA_DIM}D")
ax1.fill_between(dims_q, inliers, BASELINE,
                 where=[v >= BASELINE for v in inliers],
                 alpha=0.15, color="#3fb950", label="Amelioration vs baseline")
ax1.fill_between(dims_q, inliers, BASELINE,
                 where=[v < BASELINE for v in inliers],
                 alpha=0.15, color="#ff7b72", label="Degradation vs baseline")
for x, y in zip(dims_q, inliers):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                 ha="center", fontsize=9)
ax1.set_xlabel("Dimension ACP")
ax1.set_ylabel("Inliers RANSAC moyens")
ax1.set_title("Inliers RANSAC moyens par dimension")
ax1.legend(fontsize=8)
ax1.grid(True)
ax1.set_xticks(dims_q)

bar_colors = ["#3fb950" if r >= 100 else "#58a6ff" if r >= 80 else "#ff7b72"
              for r in retention]
bars2 = ax2.bar([f"{d}D" for d in dims_q], retention, color=bar_colors, alpha=0.82, width=0.6)
ax2.axhline(100, linestyle="--", color="#ff7b72", linewidth=1.8, label="Baseline (100%)")

chosen_idx = dims_q.index(PCA_DIM)
bars2[chosen_idx].set_edgecolor("#ffd700")
bars2[chosen_idx].set_linewidth(2.5)

for bar, r, d in zip(bars2, retention, dims_q):
    color = "#ffd700" if d == PCA_DIM else "#e6edf3"
    fw = "bold" if d == PCA_DIM else "normal"
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{r:.0f}%", ha="center", va="bottom", fontsize=9, color=color, fontweight=fw)

ax2.set_ylabel("Retention vs baseline (%)")
ax2.set_title("Retention des inliers (vert = meilleur que baseline)")
ax2.legend(fontsize=8)
ax2.grid(True, axis="y")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_matching_quality.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("03_matching_quality.png saved")


# ──────────────────────────────────────────────────────────────────────────────
# 00 — Dashboard synthese (2x2)
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Synthese ACP SIFT-RANSAC — Dimension retenue : 32D",
             fontsize=15, fontweight="bold", y=1.01)
gs = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

# A — variance cumulee
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(comps, cumvar * 100, color="#58a6ff", linewidth=2)
ax_a.fill_between(comps, cumvar * 100, alpha=0.1, color="#58a6ff")
ax_a.axvline(PCA_DIM, linestyle="-", color="#ffd700", linewidth=2.5,
             label=f"32D : {cumvar[31]*100:.1f}% var.")
ax_a.set_xlabel("Composantes ACP")
ax_a.set_ylabel("Variance cumulee (%)")
ax_a.set_title("Variance expliquee cumulee")
ax_a.legend(fontsize=8)
ax_a.grid(True)
ax_a.set_xlim(1, 128)
ax_a.set_ylim(0, 101)

# B — tailles
ax_b = fig.add_subplot(gs[0, 1])
labels_s = ["128D\n(original)", "32D\n(retenu)", "48D", "64D"]
sizes_s  = [199.7, 146.2, 214.5, 283.0]
col_s    = ["#ff7b72", "#3fb950", "#8b949e", "#8b949e"]
bars_b = ax_b.bar(labels_s, sizes_s, color=col_s, alpha=0.82, width=0.55)
for bar, sz in zip(bars_b, sizes_s):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
              f"{sz:.1f} MB", ha="center", va="bottom", fontsize=9)
ax_b.set_ylabel("Taille compressee (MB)")
ax_b.set_title("Taille index compresse (npz)")
ax_b.grid(True, axis="y")
ax_b.set_ylim(0, 330)

# C — inliers
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(dims_q, inliers, color="#58a6ff", marker="o", markersize=7, zorder=3)
ax_c.axhline(BASELINE, linestyle="--", color="#ff7b72", linewidth=1.8,
             label=f"Baseline: {BASELINE:.1f}")
ax_c.axvline(PCA_DIM, linestyle="-", color="#ffd700", linewidth=2.5,
             label=f"{PCA_DIM}D retenu")
ax_c.fill_between(dims_q, inliers, BASELINE,
                  where=[v >= BASELINE for v in inliers],
                  alpha=0.15, color="#3fb950")
ax_c.fill_between(dims_q, inliers, BASELINE,
                  where=[v < BASELINE for v in inliers],
                  alpha=0.15, color="#ff7b72")
for x, y in zip(dims_q, inliers):
    ax_c.annotate(f"{y:.1f}", (x, y), xytext=(0, 8), textcoords="offset points",
                  ha="center", fontsize=8)
ax_c.set_xlabel("Dimension ACP")
ax_c.set_ylabel("Inliers RANSAC moyens")
ax_c.set_title("Qualite de matching (inliers RANSAC)")
ax_c.legend(fontsize=8)
ax_c.grid(True)
ax_c.set_xticks(dims_q)

# D — tableau avant/apres
ax_d = fig.add_subplot(gs[1, 1])
ax_d.axis("off")
rows_tbl = [
    ["128D (avant)", "uint8", "199.7 MB", f"{BASELINE:.1f}", "100%"],
    ["32D (apres)", "float16", "146.2 MB",
     f"{inliers[dims_q.index(32)]:.1f}", "+32%"],
]
col_labels = ["Dimension", "Dtype", "Taille", "Inliers moy.", "Retention"]
tbl = ax_d.table(cellText=rows_tbl, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2.4)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#21262d")
        cell.set_text_props(fontweight="bold")
    elif r == 2:
        cell.set_facecolor("#1a2e1a")
    else:
        cell.set_facecolor("#161b22")
    cell.set_edgecolor("#30363d")
    cell.set_text_props(color="#e6edf3")
ax_d.set_title("Comparaison avant / apres ACP", fontsize=11, pad=12)

plt.savefig(f"{RESULTS_DIR}/00_synthese.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("00_synthese.png saved")

print()
print(f"Tous les graphes generes dans {RESULTS_DIR}/")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".png"):
        size = os.path.getsize(f"{RESULTS_DIR}/{f}") / 1024
        print(f"  {f}  ({size:.0f} KB)")
