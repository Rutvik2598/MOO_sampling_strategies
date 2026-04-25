"""Generate figures for the LaTeX paper."""
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("report", exist_ok=True)

# ── Load all results ─────────────────────────────────────────────────────────
files = glob.glob("results/*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

METHOD_ORDER = ["Clustering", "Diversity", "Stratified", "Random"]
BUDGET_ORDER = [50, 100, 200, 250]

# Nice B&W-safe styles: gray shades + distinct line styles + markers
COLOURS = {
    "Clustering": "0.00",   # black
    "Diversity":  "0.35",   # dark gray
    "Stratified": "0.60",   # medium gray
    "Random":     "0.80",   # light gray
}
LINESTYLES = {
    "Clustering": "-",
    "Diversity":  "--",
    "Stratified": "-.",
    "Random":     ":",
}
MARKERS = {
    "Clustering": "o",
    "Diversity":  "s",
    "Stratified": "^",
    "Random":     "D",
}

# ── Figure 1: Median Pareto Recall vs. budget (line chart) ───────────────────
med = (df.groupby(["method", "sample_size"])["recall"]
         .median()
         .reset_index()
         .rename(columns={"recall": "median_recall"}))

fig, ax = plt.subplots(figsize=(5.0, 3.8))

for method in METHOD_ORDER:
    sub = med[med["method"] == method].sort_values("sample_size")
    ax.plot(
        sub["sample_size"], sub["median_recall"],
        marker=MARKERS[method],
        color=COLOURS[method],
        linestyle=LINESTYLES[method],
        linewidth=1.8,
        markersize=6,
        label=method,
    )

ax.set_xlabel("Fixed sample budget $n$", fontsize=10)
ax.set_ylabel("Median Pareto Recall", fontsize=10)
ax.set_xticks([50, 100, 150, 200, 250])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.legend(framealpha=0.9, fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.12)
fig.savefig("report/fig_recall_vs_budget.pdf", bbox_inches="tight")
fig.savefig("report/fig_recall_vs_budget.png", dpi=200, bbox_inches="tight")
print("Saved fig_recall_vs_budget")

# ── Figure 2: Win counts grouped bar chart (B&W with hatching) ───────────────
# Compute win counts: for each (dataset, sample_size), best median recall wins
wins = (df.groupby(["dataset", "sample_size", "method"])["recall"]
          .median()
          .reset_index())
idx = wins.groupby(["dataset", "sample_size"])["recall"].idxmax()
winners = wins.loc[idx, ["sample_size", "method"]]
win_counts = (winners.groupby(["method", "sample_size"])
                     .size()
                     .reset_index(name="wins"))

# B&W-safe: gray shades only; bars identified by labels below
BW_SHADES  = {"Clustering": "0.20", "Diversity": "0.45",
              "Stratified": "0.65", "Random": "0.85"}
ABBREVS    = {"Clustering": "Clust.", "Diversity": "Divers.",
              "Stratified": "Strat.", "Random": "Rand."}

fig2, ax2 = plt.subplots(figsize=(5.0, 3.8))

x = np.arange(len(BUDGET_ORDER))
n_methods = len(METHOD_ORDER)
width = 0.18
offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

for i, method in enumerate(METHOD_ORDER):
    sub = win_counts[win_counts["method"] == method].set_index("sample_size")
    counts = [sub.loc[b, "wins"] if b in sub.index else 0 for b in BUDGET_ORDER]
    bars = ax2.bar(
        x + offsets[i],
        counts,
        width=width,
        color=BW_SHADES[method],
        edgecolor="black",
        linewidth=0.6,
    )
    # count label above bar
    for bar, val in zip(bars, counts):
        if val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(int(val)),
                ha="center", va="bottom", fontsize=7,
            )
    # strategy label below each bar (in axes-fraction y coords)
    for j, bar in enumerate(bars):
        ax2.annotate(
            ABBREVS[method],
            xy=(bar.get_x() + bar.get_width() / 2, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center", va="top", fontsize=6.5,
            rotation=45, clip_on=False,
        )

# hide default tick labels; add budget group labels via annotate at fixed depth
ax2.set_xticks(x)
ax2.set_xticklabels([""] * len(BUDGET_ORDER))
ax2.tick_params(axis="x", length=0)
for xi, b in zip(x, BUDGET_ORDER):
    ax2.annotate(
        f"$n={b}$",
        xy=(xi, 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -42),
        textcoords="offset points",
        ha="center", va="top", fontsize=9, clip_on=False,
    )
ax2.set_xlabel("Fixed sample budget", fontsize=10, labelpad=54)
ax2.set_ylabel("Win count (out of 50 datasets)", fontsize=10)
ax2.set_ylim(0, 31)
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.26)
fig2.savefig("report/fig_win_counts.pdf", bbox_inches="tight")
fig2.savefig("report/fig_win_counts.png", dpi=200, bbox_inches="tight")
print("Saved fig_win_counts")

# ── Figure 3: Secondary metrics (Precision, IGD, HV-diff) vs. budget ─────────
sec = (df.groupby(["method", "sample_size"])[["precision", "igd", "hv_diff"]]
         .median()
         .reset_index())

panels = [
    ("precision", "Precision $\\uparrow$"),
    ("igd",       "IGD $\\downarrow$"),
    ("hv_diff",   "HV-diff $\\downarrow$"),
]

fig3, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
for ax3, (metric, ylabel) in zip(axes, panels):
    for method in METHOD_ORDER:
        sub = sec[sec["method"] == method].sort_values("sample_size")
        ax3.plot(
            sub["sample_size"], sub[metric],
            marker=MARKERS[method],
            color=COLOURS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.5,
            markersize=5,
            label=method,
        )
    ax3.set_xlabel("$n$", fontsize=9)
    ax3.set_title(ylabel, fontsize=10)
    ax3.set_xticks([50, 150, 250])
    ax3.tick_params(axis="both", labelsize=8)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax3.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

# single legend below the row
handles, labels = axes[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=4,
            fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.28)
fig3.savefig("report/fig_secondary_metrics.pdf", bbox_inches="tight")
fig3.savefig("report/fig_secondary_metrics.png", dpi=220, bbox_inches="tight")
print("Saved fig_secondary_metrics")
