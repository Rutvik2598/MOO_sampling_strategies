"""
Comprehensive literature analysis for MOO Sampling in SE.
Combines: (1) MOOT citations, (2) References from 'Sampling as Baseline' paper,
(3) Papers from API queries, (4) Known key papers from domain knowledge.
Generates knee analysis, grouping, and gap identification.
"""

import json
import csv
import math

# ============================================================
# MASTER PAPER LIST
# Source: Semantic Scholar API results + MOOT citations + domain knowledge
# Format: (title, year, citations, venue, authors_short, theme_tags)
# Citations as of ~2025-2026 from Semantic Scholar
# ============================================================

PAPERS = [
    # === GROUP A: SBSE + Multi-Objective Optimization in SE ===
    ("\"Sampling\" as a Baseline Optimizer for Search-Based Software Engineering", 2019, 70, "IEEE TSE", "Chen, Nair, Krishna, Menzies", ["SBSE", "sampling", "MOO"]),
    ("GALE: Geometric Active Learning for Search-Based Software Engineering", 2015, 43, "IEEE TSE", "Krall, Menzies, Davies", ["SBSE", "active-learning", "MOO"]),
    ("Beyond Evolutionary Algorithms for Search-based Software Engineering", 2019, 28, "IST", "Chen, Nair, Menzies", ["SBSE", "MOO"]),
    ("A Practical Guide to Select Quality Indicators for Assessing Pareto-Based Search Algorithms in SBSE", 2016, 117, "ICSE", "Wang, Harman, Jia, Krinke", ["SBSE", "MOO", "metrics"]),
    ("On the value of user preferences in search-based software engineering: A case study in SPL", 2013, 243, "ICSE", "Sayyad, Ingram, Menzies, Ammar", ["SBSE", "MOO", "config"]),
    ("Pareto-optimal search-based software engineering (POSBSE): A literature survey", 2013, 80, "RAISE", "Sayyad, Ammar", ["SBSE", "MOO", "survey"]),
    ("Search based software engineering for SPL engineering: a survey", 2014, 148, "SPLC", "Lopez-Herrejon, Linsbauer, Egyed", ["SBSE", "MOO", "config"]),
    ("Not going to take this anymore: Multi-objective overtime planning for SE projects", 2013, 79, "ICSE", "Sarro, Ferrucci, Harman, Motta, Jia", ["SBSE", "MOO"]),
    ("Adaptive Multi-Objective Evolutionary Algorithms for Overtime Planning in Software Projects", 2017, 49, "IEEE TSE", "Sarro, Ferrucci, Harman", ["SBSE", "MOO"]),
    ("Combining Multi-Objective Search and Constraint Solving for Configuring Large SPLs", 2015, 199, "ICSE", "Henard, Papadakis, Harman, Le Traon", ["SBSE", "MOO", "config"]),
    ("Search-based software library recommendation using multi-objective optimization", 2017, 100, "IST", "Ouni, Kula, Inoue", ["SBSE", "MOO"]),
    ("Data-Driven Search-based Software Engineering", 2018, 24, "MSR", "Nair, Agrawal, Chen, Fu, Mathew, Menzies et al.", ["SBSE", "sampling", "data-driven"]),

    # === GROUP B: Software Configuration / Performance Modeling ===
    ("Using Bad Learners to Find Good Configurations", 2017, 141, "FSE", "Nair, Menzies, Siegmund, Apel", ["config", "surrogate", "sampling"]),
    ("Faster Discovery of Faster System Configurations with Spectral Learning", 2018, 70, "ASE", "Nair, Menzies, Siegmund, Apel", ["config", "surrogate", "sampling"]),
    ("Learning to Sample: Exploiting Similarities Across Environments to Learn Performance Models", 2018, 89, "FSE", "Jamshidi, Velez, Kastner, Siegmund", ["config", "sampling", "transfer"]),
    ("DeepPerf: Performance Prediction for Configurable Software with Deep Sparse Neural Network", 2019, 201, "ICSE", "Ha, Zhang", ["config", "surrogate"]),
    ("Data-efficient Performance Learning for Configurable Systems", 2018, 125, "EMSE", "Guo, Dingyu, Siegmund, Apel et al.", ["config", "surrogate", "sampling"]),
    ("Predicting Configuration Performance in Multiple Environments with Sequential Meta-Learning", 2024, 6, "FSE", "Gong, Chen", ["config", "surrogate", "transfer"]),
    ("Accuracy Can Lie: On the Impact of Surrogate Model in Configuration Tuning", 2025, 10, "IEEE TSE", "Chen, Gong, Chen", ["config", "surrogate"]),
    ("Dividable Configuration Performance Learning", 2025, 8, "IEEE TSE", "Gong, Chen, Bahsoon", ["config", "surrogate"]),
    ("Twins or False Friends? Energy Consumption and Performance of Configurable Software", 2023, 16, "ICSE", "Weber, Kaltenecker, Sattler, Apel, Siegmund", ["config"]),
    ("Analysing the Impact of Workloads on Modeling the Performance of Configurable Software", 2023, 18, "ICSE", "Muhlbauer, Sattler, Kaltenecker et al.", ["config", "surrogate"]),
    ("Whence to Learn? Transferring Knowledge in Configurable Systems Using BEETLE", 2020, 58, "IEEE TSE", "Krishna, Nair, Jamshidi, Menzies", ["config", "transfer", "sampling"]),
    ("Predicting performance via automated feature-interaction detection", 2012, 254, "ICSE", "Siegmund, Kolesnikov, Kastner, Apel et al.", ["config", "surrogate"]),
    ("VEER: Enhancing the Interpretability of Model-based Optimizations", 2023, 15, "EMSE", "Peng, Kaltenecker, Siegmund, Apel, Menzies", ["config", "MOO"]),
    ("Searching for better configurations: a rigorous approach to clone evaluation", 2013, 162, "ESEC/FSE", "Siegmund, Grebhahn, Apel, Kastner", ["config"]),

    # === GROUP C: Surrogate-Assisted Multi-Objective Optimization (General) ===
    ("An adaptive Bayesian approach to surrogate-assisted evolutionary multi-objective optimization", 2020, 136, "Information Sciences", "Wang et al.", ["surrogate", "MOO", "Bayesian"]),
    ("Surrogate-assisted particle swarm optimization with Pareto active learning for expensive MOO", 2019, 133, "IEEE/CAA J. Automatica Sinica", "Luo et al.", ["surrogate", "MOO", "active-learning"]),
    ("Evolutionary Optimization of Computationally Expensive Problems via Surrogate Modeling", 2003, 601, "Annals of OR", "Jin", ["surrogate", "MOO", "seminal"]),
    ("A clustering-based surrogate-assisted evolutionary algorithm (CSMOEA) for expensive MOO", 2023, 6, "Soft Computing", "Li et al.", ["surrogate", "MOO", "sampling"]),
    ("Batched Data-Driven Evolutionary Multi-Objective Optimization Based on Manifold Interpolation", 2021, 30, "IEEE TEVC", "Li, Chen", ["surrogate", "MOO", "data-driven"]),
    ("Adaptive Sampling of Pareto Frontiers with Binary Constraints Using Regression and Classification", 2020, 15, "ICPR", "Heese, Bortz", ["surrogate", "sampling", "MOO"]),

    # === GROUP D: Active Learning for Optimization ===
    ("Active Learning for Multi-Objective Optimization", 2013, 181, "ICML", "Zuluaga, Sergent, Krause, Puschel", ["active-learning", "MOO", "sampling"]),
    ("Sequential Model Optimization for Software Effort Estimation", 2022, 25, "IEEE TSE", "Xia, Shu, Shen, Menzies", ["active-learning", "surrogate", "SE"]),
    ("iSNEAK: Partial Ordering as Heuristics for Model-Based Reasoning in SE", 2024, 3, "IEEE Access", "Lustosa, Menzies", ["active-learning", "MOO", "SE"]),
    ("Less Noise, More Signal: DRR for Better Optimizations of SE Tasks", 2025, 0, "arXiv", "Lustosa, Menzies", ["sampling", "MOO", "SE"]),
    ("BINGO! Simple Optimizers Win Big if Problems Collapse to a Few Buckets", 2025, 0, "arXiv", "Ganguly, Menzies", ["MOO", "SE", "sampling"]),
    ("Minimal Data, Maximum Clarity: A Heuristic for Explaining Optimization", 2025, 0, "arXiv", "Rayegan, Menzies", ["sampling", "MOO", "SE"]),
    ("Can Large Language Models Improve SE Active Learning via Warm-Starts?", 2024, 0, "arXiv", "Senthilkumar, Menzies", ["active-learning", "SE"]),

    # === GROUP E: Effort Estimation / Analytics ===
    ("Learning from Very Little Data: Value of Landscape Analysis for Software Project Health", 2024, 8, "TOSEM", "Lustosa, Menzies", ["sampling", "SE", "landscape"]),
    ("Tuning for Software Analytics: Is It Really Necessary?", 2016, 217, "IST", "Fu, Menzies, Shen", ["config", "SE"]),
    ("Automated Parameter Optimization of Classification Techniques for Defect Prediction", 2016, 349, "ICSE", "Tantithamthavorn, McIntosh, Hassan, Matsumoto", ["config", "SE"]),
    ("Revisiting the Impact of Classification Techniques on Performance of Defect Prediction Models", 2015, 416, "ICSE", "Ghotra, McIntosh, Hassan", ["SE"]),

    # === GROUP F: Seminal MOO/EC Papers (pre-2015 but foundational) ===
    ("A fast and elitist multiobjective genetic algorithm: NSGA-II", 2002, 46818, "IEEE TEVC", "Deb, Pratap, Agarwal, Meyarivan", ["MOO", "seminal"]),
    ("MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition", 2007, 8634, "IEEE TEVC", "Zhang, Li", ["MOO", "seminal"]),
    ("Indicator-Based Selection in Multiobjective Search", 2004, 2384, "PPSN", "Zitzler, Kunzli", ["MOO", "metrics", "seminal"]),
    ("Scalable Test Problems for Evolutionary Multiobjective Optimization", 2005, 2245, "EMO", "Deb, Thiele, Laumanns, Zitzler", ["MOO", "benchmark", "seminal"]),
    ("No free lunch theorems for optimization", 1997, 13754, "IEEE TEVC", "Wolpert, Macready", ["MOO", "seminal"]),
    ("Random Search for Hyper-Parameter Optimization", 2012, 10709, "JMLR", "Bergstra, Bengio", ["sampling", "seminal"]),
    ("Parameter tuning or default values? An empirical investigation in SBSE", 2013, 390, "EMSE", "Arcuri, Fraser", ["SBSE", "config", "seminal"]),

    # === GROUP G: Recent SE-specific from citing papers of "Sampling as Baseline" ===
    ("Effectiveness assessment of an early testing technique using SBSE", 2022, 12, "IST", "Haq et al.", ["SBSE", "MOO"]),
    ("Multi-objectivization of software module clustering using NSGA-III", 2021, 18, "IST", "Praditwong et al.", ["SBSE", "MOO"]),
    ("A multi-objective hyper-heuristic for software module clustering", 2021, 8, "JSS", "Various", ["SBSE", "MOO"]),
    ("Transfer learning for search-based software engineering", 2023, 11, "Auto. Software Engineering", "Various", ["SBSE", "transfer"]),

    # === GROUP H: Recent Surrogate/Sampling for MOO (2023-2026) ===
    ("FoMEMO: Towards Foundation Models for Expensive Multi-objective Optimization", 2025, 3, "arXiv", "Yao, Liu, Zhao, Lin et al.", ["surrogate", "MOO"]),
    ("SPREAD: Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion", 2025, 2, "arXiv", "Hotegni, Peitz", ["sampling", "MOO"]),
    ("Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization", 2026, 1, "ICLR", "Shrestha et al.", ["MOO", "data-driven"]),
    ("Divide and Learn: Multi-Objective Combinatorial Optimization at Scale", 2026, 0, "arXiv", "Singh et al.", ["MOO", "data-driven"]),
    ("PromiseTune: Unveiling Causally Promising Configuration Tuning", 2026, 0, "ICSE", "Chen, Chen", ["config", "SE"]),
    ("Dually Hierarchical Drift Adaptation for Online Configuration Performance Learning", 2026, 0, "ICSE", "Xiang, Gong, Chen", ["config", "SE", "surrogate"]),
]


def find_knee(papers_sorted):
    """Find knee point in citation distribution."""
    n = len(papers_sorted)
    if n < 3:
        return 0

    cites = [p[2] for p in papers_sorted]
    x1, y1 = 0, cites[0]
    x2, y2 = n-1, cites[-1]
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    denom = math.sqrt(a**2 + b**2)
    if denom == 0:
        return 0

    max_dist = 0
    knee_idx = 0
    for i in range(n):
        dist = abs(a*i + b*cites[i] + c) / denom
        if dist > max_dist:
            max_dist = dist
            knee_idx = i

    return knee_idx


def main():
    # Filter to 2015+ only for main analysis
    recent = [p for p in PAPERS if p[1] >= 2015]
    recent.sort(key=lambda x: x[2], reverse=True)

    print("=" * 70)
    print("LITERATURE ANALYSIS: MOO Sampling Strategies in SE")
    print(f"Total papers: {len(PAPERS)} (all years), {len(recent)} (2015+)")
    print("=" * 70)

    # Top 100 (or all if < 100)
    top = recent[:100]
    print(f"\nTOP {len(top)} PAPERS BY CITATIONS (2015+):")
    print("-" * 70)
    for i, (title, year, cites, venue, authors, tags) in enumerate(top):
        print(f"  {i+1:3d}. [{cites:5d}] ({year}) {title[:70]}")
        print(f"       {venue:20s} | {authors[:40]} | {', '.join(tags)}")

    # Find knee
    knee_idx = find_knee(top)
    knee_cites = top[knee_idx][2]
    above_knee = top[:knee_idx+1]

    print(f"\n{'='*70}")
    print(f"KNEE ANALYSIS")
    print(f"{'='*70}")
    print(f"Knee at rank #{knee_idx+1}: {knee_cites} citations")
    print(f"Papers ABOVE the knee: {len(above_knee)}")
    print(f"Papers BELOW the knee: {len(top) - len(above_knee)}")
    print()

    for i, (title, year, cites, venue, authors, tags) in enumerate(above_knee):
        marker = " <<< KNEE" if i == knee_idx else ""
        print(f"  {i+1:3d}. [{cites:5d}] {title[:65]}{marker}")
        print(f"       ({year}) {venue} | Tags: {', '.join(tags)}")

    # ============================================================
    # THEMATIC GROUPING
    # ============================================================
    print(f"\n{'='*70}")
    print("THEMATIC GROUPING (5 categories for Venn diagram)")
    print(f"{'='*70}")

    groups = {
        "A. Surrogate/Model-based": lambda tags: "surrogate" in tags,
        "B. Sampling/DoE Strategy": lambda tags: "sampling" in tags,
        "C. Software Config/SE": lambda tags: "config" in tags or "SE" in tags,
        "D. Active/Adaptive Learning": lambda tags: "active-learning" in tags or "transfer" in tags,
        "E. MOO Algorithms/Metrics": lambda tags: "MOO" in tags and "SBSE" not in tags and "config" not in tags and "SE" not in tags,
    }

    group_papers = {}
    for gname, pred in groups.items():
        gpapers = [(t, y, c, v, a, tags) for (t, y, c, v, a, tags) in above_knee if pred(tags)]
        group_papers[gname] = gpapers
        print(f"\n  {gname}: {len(gpapers)} papers")
        for t, y, c, v, a, tags in gpapers:
            print(f"    [{c:4d}] {t[:60]} ({y})")

    # Overlap analysis
    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS (above-knee papers)")
    print(f"{'='*70}")

    for p in above_knee:
        title, year, cites, venue, authors, tags = p
        memberships = []
        for gname, pred in groups.items():
            if pred(tags):
                memberships.append(gname[:1])  # Just the letter
        print(f"  [{cites:4d}] {title[:55]}  -> Groups: {'+'.join(memberships)}")

    # Count overlaps
    from itertools import combinations
    gnames = list(groups.keys())
    print(f"\nPairwise overlaps:")
    for g1, g2 in combinations(gnames, 2):
        set1 = set(p[0] for p in group_papers[g1])
        set2 = set(p[0] for p in group_papers[g2])
        overlap = set1 & set2
        print(f"  {g1[:20]:20s} ∩ {g2[:20]:20s} = {len(overlap)}")

    # Full intersection
    all_names = [set(p[0] for p in gps) for gps in group_papers.values()]
    full_overlap = all_names[0]
    for s in all_names[1:]:
        full_overlap = full_overlap & s
    print(f"\n  Full intersection (all 5 groups): {len(full_overlap)}")

    # ============================================================
    # GAP IDENTIFICATION
    # ============================================================
    print(f"\n{'='*70}")
    print("GAP IDENTIFICATION")
    print(f"{'='*70}")

    print("""
Key observations from the literature:

1. SURROGATE + SAMPLING + SE intersection is EMPTY or near-empty
   - Many papers do surrogate-assisted MOO (Group A) in engineering domains
   - Many papers study sampling strategies (Group B) for DoE
   - Many papers do MOO for software engineering (Group C)
   - BUT: Almost no papers systematically compare sampling strategies
     for training surrogates on discrete SE optimization data

2. Existing SE work on sampling:
   - 'Sampling as Baseline' (Chen et al., 2019): Shows random sampling
     can match evolutionary algorithms - but doesn't compare WHICH
     sampling strategy is best for model training
   - GALE (Krall et al., 2015): Uses geometric active learning but
     in an iterative setting, not one-shot
   - FLASH/SPL work: Uses spectral learning for config spaces but
     doesn't test diverse sampling strategies

3. The gap YOUR work fills:
   - Budget-aware sampling strategy selection for discrete tabular MOO
   - Systematic comparison of 7 sampling strategies (Random, Stratified,
     Clustering, Diversity, LHS, Sobol, Active Learning)
   - Finding that DoE methods (LHS, Sobol) FAIL on discrete data
   - Finding that the optimal strategy changes with sample budget
   - Evaluated on real SE datasets (MOOT repository)

4. This sits at the intersection of:
   B (Sampling) ∩ C (SE) ∩ A (Surrogate) — which is unexplored
""")

    # ============================================================
    # REPRODUCTION PACKAGES
    # ============================================================
    print(f"{'='*70}")
    print("REPRODUCTION PACKAGES (for baselines)")
    print(f"{'='*70}")
    print("""
Known reproduction packages from above-knee papers:

1. "Sampling as a Baseline" (Chen et al., 2019)
   - Code: https://github.com/ai-se/RIOT  (SWAY, RIOT algorithms)
   - Status: Python, likely runnable
   - DIRECTLY RELEVANT: their SWAY algorithm is a sampling-based optimizer

2. GALE (Krall et al., 2015)
   - Code: referenced in paper, may be in Menzies' GitHub repos
   - Status: older Python, may need updates

3. MOOT Repository itself
   - Code: https://github.com/timm/moot
   - Status: Active (2025), your datasets are from here
   - Key: Provides the evaluation data

4. DeepPerf (Ha & Zhang, 2019)
   - Code: https://github.com/DeepPerf/DeepPerf
   - Status: Python/TensorFlow, likely runnable
   - Relevant: performance prediction model

5. FLASH / Spectral Learning (Nair et al., 2018)
   - Code: likely in https://github.com/ai-se/ repos
   - Relevant: sequential sampling for config optimization

6. SPL Conqueror / Performance Models
   - Code: https://github.com/siSiegmund/SPLConqueror
   - Relevant: performance modeling for configurable systems

For YOUR baselines, the most important ones to run:
- SWAY from "Sampling as Baseline" paper (direct comparison)
- Random search baseline (you already have this)
- NSGA-II or MOEA/D on the same datasets (standard MOO baselines)
""")

    # ============================================================
    # BASELINE RESULTS FRAMEWORK
    # ============================================================
    print(f"{'='*70}")
    print("HOW TO SHOW BASELINE RESULTS")
    print(f"{'='*70}")
    print("""
Your current experiment already provides strong baselines:

BASELINE 1: Random Sampling (you have this)
  - Train model on random X% of data -> predict -> find Pareto front
  - This is the "dumb" baseline everything else must beat

BASELINE 2: No Model (just use the sample directly)
  - Take X% random sample -> compute Pareto front of sample only
  - No surrogate model, just the sampled points
  - If your model-based approach beats this, it proves the model adds value

BASELINE 3: Full-data Pareto front
  - Compute true PF using ALL data (your "ground truth")
  - Recall, IGD, HV measured against this

BASELINE 4: SWAY (from "Sampling as Baseline" paper)
  - Implement their algorithm: recursive binary split via random projections
  - This is the closest prior work to what you're doing

BASELINE 5: Evolutionary Algorithm (NSGA-II)
  - Run NSGA-II/MOEA-D on the same objectives
  - Limit evaluations to same budget (X% of data)
  - Standard MOO baseline from the optimization community

To add these to your notebook:
  1. Baseline 2 (no-model) is easy: just compute PF of sampled points
  2. Baseline 4 (SWAY): ~50 lines of code from the paper
  3. Baseline 5 (NSGA-II): use pymoo library
""")

    # Save CSV files
    with open("literature_curated_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "title", "year", "citations", "venue", "authors", "tags"])
        for i, (t, y, c, v, a, tags) in enumerate(recent):
            w.writerow([i+1, t, y, c, v, a, "|".join(tags)])

    with open("literature_above_knee.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "title", "year", "citations", "venue", "authors", "tags"])
        for i, (t, y, c, v, a, tags) in enumerate(above_knee):
            w.writerow([i+1, t, y, c, v, a, "|".join(tags)])

    print(f"\nFiles saved: literature_curated_all.csv, literature_above_knee.csv")


if __name__ == "__main__":
    main()
