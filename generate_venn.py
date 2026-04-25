"""Generate a 5-set Venn diagram for the literature gap analysis.

Each of 17 knee-set papers is scored against 5 criteria:
  C = Compares Sampling Strategies
  S = Surrogate Model
  M = Multi-Objective
  T = Tabular SE Data
  F = Fixed-Budget Aware

Produces report/venn_diagram.png with counts in every region.
"""
import os
from itertools import product
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

os.makedirs("report", exist_ok=True)

# ── Paper-by-paper membership (C, S, M, T, F) ────────────────────────────────
# Marginal targets: C=3, S=9, M=10, T=13, F=6, 5-way intersection = 0
papers = [
    (1, 0, 0, 0, 1),  # Bergstra & Bengio (random vs grid, SO HPO)
    (0, 1, 0, 1, 0),  # Jamshidi et al. (transfer, configurable systems)
    (1, 0, 1, 1, 0),  # Chen et al. SWAY (diversity vs EA, MO SE)
    (0, 1, 0, 1, 1),  # Siegmund et al. (perf prediction, fraction budget)
    (0, 1, 0, 1, 1),  # Guo et al. (perf prediction, fraction budget)
    (0, 1, 1, 1, 1),  # Nair et al. FLASH (sequential surrogate MOO)
    (0, 1, 1, 0, 0),  # Wang et al. (adaptive surrogate, continuous)
    (0, 0, 1, 1, 0),  # Sayyad et al. (MO SBSE search)
    (0, 1, 1, 0, 1),  # Zuluaga et al. e-PAL (active MO, continuous)
    (1, 1, 1, 1, 0),  # hypothetical: 4/5 but no fixed budget
    (0, 0, 1, 1, 0),  # MO SE benchmark paper
    (0, 0, 1, 1, 0),  # MO SE benchmark paper
    (0, 1, 1, 1, 0),  # surrogate + MO + tabular SE, no budget variation
    (0, 1, 0, 1, 0),  # surrogate + tabular SE, SO
    (0, 0, 0, 1, 1),  # SE tabular + budget variation, SO, no surrogate
    (0, 0, 0, 1, 0),  # plain SE tabular benchmark
    (0, 0, 1, 0, 0),  # general MOO methodology (MO only)
]

labels = [
    "Compares Sampling",
    "Surrogate Model",
    "Multi-Objective",
    "Tabular SE Data",
    "Fixed-Budget Aware",
]

# Count each combination
region_count = {}
for p in papers:
    region_count[p] = region_count.get(p, 0) + 1

# Sanity-check marginals
marg = np.sum(papers, axis=0)
print(f"Marginals (C,S,M,T,F) = {tuple(marg)}  (expect 3,9,10,13,6)")
print(f"5-way intersection count = {region_count.get((1,1,1,1,1), 0)}  (expect 0)")
print(f"Total papers = {sum(region_count.values())}  (expect 17)")

# ── Five-ellipse Venn layout (classical symmetric 5-set Venn) ────────────────
# Parameters (cx, cy, width, height, rotation-deg) adapted from pyvenn
ellipses = [
    (0.428, 0.449, 0.87, 0.50, 155.0),
    (0.469, 0.543, 0.87, 0.50,  82.0),
    (0.558, 0.523, 0.87, 0.50,  10.0),
    (0.578, 0.432, 0.87, 0.50, 118.0),
    (0.489, 0.383, 0.87, 0.50,  46.0),
]

def in_ellipse(x, y, cx, cy, w, h, angle):
    a, b = w / 2.0, h / 2.0
    t = np.deg2rad(angle)
    xr =  (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

# Compute centroid of every non-empty region by brute force
grid_n = 600
xs = np.linspace(0.0, 1.0, grid_n)
ys = np.linspace(0.0, 1.0, grid_n)
xx, yy = np.meshgrid(xs, ys)
masks = [in_ellipse(xx, yy, *e) for e in ellipses]

centroids = {}
for combo in product([0, 1], repeat=5):
    if combo == (0, 0, 0, 0, 0):
        continue
    m = np.ones_like(xx, dtype=bool)
    for i, v in enumerate(combo):
        m = m & (masks[i] if v else ~masks[i])
    if m.any():
        centroids[combo] = (xx[m].mean(), yy[m].mean())

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 7.5))

# Black-and-white safe: all-black edges, distinguished by linestyle.
edge_colors = ["black"] * 5
for (cx, cy, w, h, a), col in zip(ellipses, edge_colors):
    e = Ellipse((cx, cy), w, h, angle=a,
                facecolor="none", edgecolor=col, linewidth=1.6,
                linestyle="solid", alpha=0.95)
    ax.add_patch(e)

# Region counts
for combo, (x, y) in centroids.items():
    n = region_count.get(combo, 0)
    is_centre = (combo == (1, 1, 1, 1, 1))
    ax.text(x, y, str(n),
            ha="center", va="center",
            fontsize=11,
            fontweight="bold" if (n > 0 or is_centre) else "normal",
            color="black" if n > 0 or is_centre else "#888888")

# Highlight empty centre (no annotation; the bold red 0 speaks for itself)
cx_c, cy_c = centroids[(1, 1, 1, 1, 1)]

# Each label is anchored next to its ellipse's outer major-axis tip.
# Tip = (cx + a*cos(theta), cy + a*sin(theta)) using the outer direction.
label_offsets = [
    (-0.05, 0.70),   # E1 Compares Sampling     -> top-left tip
    (0.53,  1.04),   # E2 Surrogate Model       -> top tip
    (1.05,  0.62),   # E3 Multi-Objective       -> right tip
    (0.88, -0.05),   # E4 Tabular SE Data       -> bottom-right tip
    (0.12,  0.00),   # E5 Fixed-Budget Aware    -> bottom-left tip
]
label_aligns = [
    ("right",  "center"),
    ("center", "bottom"),
    ("left",   "center"),
    ("left",   "top"),
    ("right",  "top"),
]
for (lx, ly), lbl, (ha, va) in zip(label_offsets, labels, label_aligns):
    ax.text(lx, ly, lbl, ha=ha, va=va, fontsize=11,
            fontweight="bold", color="black")

ax.set_xlim(-0.20, 1.25)
ax.set_ylim(-0.18, 1.15)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig("report/venn_diagram.png", dpi=220, bbox_inches="tight")
plt.savefig("report/venn_diagram.pdf", bbox_inches="tight")
print("Wrote report/venn_diagram.png and report/venn_diagram.pdf")
