"""
Generate the research gap Venn diagram showing 5 thematic groups
and the unexplored intersection where our work sits.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def draw_venn_research_gap():
    """
    Draw a 3-circle Venn diagram for the core research gap:
      A = Surrogate/Model-based optimization
      B = Sampling / Design of Experiments (DoE)
      C = Software Engineering MOO

    Plus 2 smaller satellite circles:
      D = Active/Adaptive Learning
      E = MOO Algorithms/Metrics

    The gap is the center of A ∩ B ∩ C = 0 papers.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.0, 5.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- 3 main circles (Venn core) ----
    # A = Surrogate (top left)
    circle_A = plt.Circle((-1.2, 1.0), 2.2, fill=True,
                          facecolor='#3498db33', edgecolor='#2980b9',
                          linewidth=2.5, linestyle='-')
    ax.add_patch(circle_A)

    # B = Sampling/DoE (top right)
    circle_B = plt.Circle((1.2, 1.0), 2.2, fill=True,
                          facecolor='#e74c3c33', edgecolor='#c0392b',
                          linewidth=2.5, linestyle='-')
    ax.add_patch(circle_B)

    # C = Software Engineering MOO (bottom center)
    circle_C = plt.Circle((0.0, -1.0), 2.2, fill=True,
                          facecolor='#2ecc7133', edgecolor='#27ae60',
                          linewidth=2.5, linestyle='-')
    ax.add_patch(circle_C)

    # ---- Satellite circles (D and E) ----
    # D = Active Learning (top, smaller)
    circle_D = plt.Circle((-3.2, 3.5), 1.1, fill=True,
                          facecolor='#f39c1233', edgecolor='#e67e22',
                          linewidth=2.0, linestyle='--')
    ax.add_patch(circle_D)

    # E = MOO Algorithms/Metrics (right, smaller)
    circle_E = plt.Circle((3.5, 3.2), 1.1, fill=True,
                          facecolor='#9b59b633', edgecolor='#8e44ad',
                          linewidth=2.0, linestyle='--')
    ax.add_patch(circle_E)

    # ---- Labels for circles ----
    ax.text(-2.8, 2.2, "A. Surrogate/\nModel-based",
            fontsize=11, fontweight='bold', color='#2980b9',
            ha='center', va='center')
    ax.text(-2.5, 1.3, "DeepPerf\nFLASH\nGP-MOO",
            fontsize=8, color='#2471a3', ha='center', va='center',
            fontstyle='italic')

    ax.text(2.8, 2.2, "B. Sampling/\nDoE Strategy",
            fontsize=11, fontweight='bold', color='#c0392b',
            ha='center', va='center')
    ax.text(2.5, 1.3, "LHS, Sobol\nRandom Search\nBergstra '12",
            fontsize=8, color='#922b21', ha='center', va='center',
            fontstyle='italic')

    ax.text(0.0, -2.5, "C. Software Eng.\nMOO / Config",
            fontsize=11, fontweight='bold', color='#27ae60',
            ha='center', va='center')
    ax.text(0.0, -3.2, "SBSE, SPL tuning\nChen '19, Sayyad '13",
            fontsize=8, color='#1e8449', ha='center', va='center',
            fontstyle='italic')

    ax.text(-3.2, 4.2, "D. Active\nLearning",
            fontsize=10, fontweight='bold', color='#e67e22',
            ha='center', va='center')

    ax.text(3.5, 4.0, "E. MOO\nMetrics",
            fontsize=10, fontweight='bold', color='#8e44ad',
            ha='center', va='center')

    # ---- Pairwise intersection labels ----
    # A ∩ B (top center — surrogate + sampling in general domains)
    ax.text(0.0, 2.0, "A∩B\nSurrogate +\nSampling",
            fontsize=8, ha='center', va='center',
            color='#555', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1', alpha=0.7))

    # A ∩ C (left — surrogate for SE)
    ax.text(-1.2, -0.4, "A∩C\nSurrogate\nfor SE",
            fontsize=8, ha='center', va='center',
            color='#555', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1', alpha=0.7))

    # B ∩ C (right — sampling in SE)
    ax.text(1.2, -0.4, "B∩C\nSampling\nin SE",
            fontsize=8, ha='center', va='center',
            color='#555', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1', alpha=0.7))

    # ---- THE GAP: A ∩ B ∩ C = 0 papers ---- (center)
    # Highlight with a star/burst
    gap_x, gap_y = 0.0, 0.55
    star = plt.Circle((gap_x, gap_y), 0.55, fill=True,
                       facecolor='#e74c3c', edgecolor='#c0392b',
                       linewidth=3, alpha=0.85, zorder=10)
    ax.add_patch(star)
    ax.text(gap_x, gap_y + 0.08, "GAP",
            fontsize=14, fontweight='bold', color='white',
            ha='center', va='center', zorder=11)
    ax.text(gap_x, gap_y - 0.25, "Our Work",
            fontsize=10, fontweight='bold', color='white',
            ha='center', va='center', zorder=11)

    # Arrow and annotation box explaining the gap
    ax.annotate(
        "No papers systematically compare\n"
        "sampling strategies for training\n"
        "surrogates on discrete SE data\n\n"
        "→ Our work fills this gap:\n"
        "  7 strategies × 6 MOOT datasets\n"
        "  × 4 budgets × 20 repeats",
        xy=(gap_x + 0.5, gap_y),
        xytext=(3.0, -1.5),
        fontsize=9,
        ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffeaa7',
                  edgecolor='#f39c12', linewidth=1.5),
        arrowprops=dict(arrowstyle='->', color='#c0392b',
                        lw=2, connectionstyle='arc3,rad=0.3'),
        zorder=12
    )

    # ---- Dashed lines connecting D and E to main diagram ----
    ax.annotate('', xy=(-2.0, 2.5), xytext=(-2.8, 3.0),
                arrowprops=dict(arrowstyle='->', color='#e67e22',
                                lw=1.5, linestyle='--'))
    ax.annotate('', xy=(2.2, 2.5), xytext=(2.8, 2.8),
                arrowprops=dict(arrowstyle='->', color='#8e44ad',
                                lw=1.5, linestyle='--'))

    # ---- Title and subtitle ----
    ax.text(0.0, 4.8, "Research Gap: Budget-Aware Sampling for Surrogate MOO in SE",
            fontsize=15, fontweight='bold', ha='center', va='center',
            color='#2c3e50')
    ax.text(0.0, 4.35, "Intersection of A ∩ B ∩ C = 0 papers in literature (above-knee analysis of 60+ papers)",
            fontsize=10, ha='center', va='center', color='#7f8c8d',
            fontstyle='italic')

    # ---- Legend ----
    legend_items = [
        mpatches.Patch(facecolor='#3498db33', edgecolor='#2980b9', linewidth=2,
                       label='A. Surrogate/Model-based (5 papers)'),
        mpatches.Patch(facecolor='#e74c3c33', edgecolor='#c0392b', linewidth=2,
                       label='B. Sampling/DoE Strategy (4 papers)'),
        mpatches.Patch(facecolor='#2ecc7133', edgecolor='#27ae60', linewidth=2,
                       label='C. Software Config/SE (8 papers)'),
        mpatches.Patch(facecolor='#f39c1233', edgecolor='#e67e22', linewidth=2,
                       label='D. Active/Adaptive Learning (2 papers)'),
        mpatches.Patch(facecolor='#9b59b633', edgecolor='#8e44ad', linewidth=2,
                       label='E. MOO Algorithms/Metrics (2 papers)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2,
                       label='OUR WORK: A∩B∩C (unexplored)'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("venn_research_gap.png", dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig("venn_research_gap.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: venn_research_gap.png, venn_research_gap.pdf")
    plt.close()


if __name__ == "__main__":
    draw_venn_research_gap()
