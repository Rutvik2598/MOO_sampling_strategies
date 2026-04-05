# Budget-Aware Sampling Strategies for Multi-Objective Optimization in Software Engineering

## 1 Introduction

Modern software engineering is characterized by a plethora of choices. Given a goal, there are many ways to implement all the parts that lead to that goal. And across those choices there is a large space of configuration parameters that must be set.

While configuration flexibility is a strength, it comes at a steep cost. Configuration spaces are combinatorially large, often poorly documented, and rich in subtle interactions. Defaults are unreliable, manual tuning is unscalable, and human intuition is fallible. As systems and processes grow in complexity, the ability to manage these configurations becomes very difficult. Part of that difficulty arises from the necessity to trade off between competing constraints; e.g., how to deliver more code at less cost? Or how to answer database queries faster but use less energy? Hence, like other researchers, we warn:

> **Configurability is a liability without tool support.**

Accordingly, we explore *surrogate-assisted multi-objective optimization* for configuration support. In this paradigm, a machine learning model (the "surrogate") is trained on a small subset of evaluated configurations to predict the objectives for unevaluated ones. This approach is appealing because evaluating the true objectives can be very expensive—running a full test suite, compiling with different Makefile settings, or simulating a software process model all take significant time.

But this raises a fundamental question that has received almost no systematic study:

> *Given a fixed evaluation budget (say 5%, 10%, 20%, or 30% of all configurations), which sampling strategy should be used to select the training data for the surrogate model?*

Classical Design of Experiments (DoE) methods—Latin Hypercube Sampling (LHS), Sobol sequences—were designed for continuous engineering spaces. Software engineering data, by contrast, is typically discrete, mixed-type, and tabular. Do these classical methods still work? Are there better alternatives?

This question sits at the intersection of three research areas—*surrogate modeling*, *sampling/DoE*, and *multi-objective software engineering*—yet has received almost no prior investigation (Section 2).

**Contributions.** We present a systematic empirical study comparing 7 sampling strategies plus 3 baselines across 6 MOOT benchmark datasets at 4 budget levels (5%, 10%, 20%, 30%), with 20 repetitions per configuration (4,800 total experiments):

1. Objective-aware baselines (NSGA-II at 0.500 mean recall, SWAY at 0.421) outperform all surrogate methods—but use 3–100× more evaluations.
2. Among equal-budget surrogate methods, Diversity (MaxMin) is best overall (0.279 mean recall, 1.17–1.44× over random).
3. The optimal surrogate strategy is *budget-dependent*: Stratified wins at 5%, Diversity dominates at 10%+.
4. Classical DoE methods (LHS, Sobol) *fail* on discrete SE data.

The rest of the paper is organized as follows: Section 2 discusses related work and the gap in the literature, Section 3 details our methodology, and presents experimental results.

---

## 2 A Gap in the Literature

Having defined our research question, the literature was searched for work at this intersection. We analyzed 60+ papers from MOOT citations, the Semantic Scholar API, and domain literature. Following the approach of Senthilkumar and Menzies [1], we ranked retrieved papers by citation count and identified the "knee" of the citation curve—the point where citation accumulation significantly decreases. Papers above this threshold represent core contributions that have shaped the field. Snowballing was then used to find additional relevant work.

In all, we collected 17 papers from the knee of the citation curve and checked each against five criteria that, taken together, define our research contribution: (1) **Compares Sampling Strategies**, (2) **Surrogate Model**, (3) **Multi-Objective**, (4) **Tabular SE Data**, and (5) **Budget-Aware**. When expressed as a Venn diagram (Figure 1), the center intersection — where all five overlap — is empty. That is the gap this paper fills.

### Observations

- **Compares Sampling Strategies?** Only 3 of 17 papers compare sampling approaches at all. Bergstra & Bengio [3] compare random vs. grid search but for single-objective HPO on continuous spaces. Jamshidi et al. [12] explore transfer learning for sampling but not head-to-head strategy comparison. Chen et al. [15] compare SWAY to evolutionary algorithms but don't evaluate which *sampling strategy* produces the best surrogate model.

- **Surrogate Model?** 9 of 17 papers use surrogate models, but overwhelmingly for single-objective configuration tasks (Siegmund, Nair, Guo, Ha, Jamshidi) or on continuous engineering benchmarks (Jin, Wang, Zuluaga). None systematically study how to *select training data* for the surrogate.

- **Multi-Objective?** 10 of 17 papers handle MOO, but most either don't use surrogates (Sayyad, Henard, Krall, Sarro, Lustosa, Senthilkumar) or don't use real SE data (Jin, Zuluaga, Wang).

- **Tabular SE Data?** 13 of 17 papers use SE data, but only 3 of those also use surrogate models for multi-objective problems.

- **Budget-Aware?** Only 6 of 17 papers study varying evaluation budgets, and none of those compare multiple sampling strategies.

No paper even covers 4 of 5 criteria. The most isolating criterion is "Compares Sampling Strategies," which never co-occurs with "Budget-Aware" at all. Unlike prior work, this paper (a) systematically compares 7 sampling strategies and 3 baselines; (b) trains surrogate models (Random Forest) on the sampled data; (c) handles multi-objective optimization with 2–5 objectives; (d) evaluates on 6 real SE benchmark datasets from MOOT; and (e) studies 4 budget levels (5%, 10%, 20%, 30%), revealing that the optimal strategy changes with budget.

---

## 3 Baseline Results

Our experiment compared 10 methods (7 sampling strategies + 3 baselines) across 6 MOOT datasets at 4 budget levels (5%, 10%, 20%, 30%) with 20 repetitions each — 4,800 total experiments.

### Table 2. Best Method at Each Budget (Mean Pareto Recall)

| Budget | Best Overall | Recall | vs Random | Best Surrogate |
|--------|-------------|--------|-----------|----------------|
| 5% | SWAY | 0.251 | 1.98× | Stratified (0.148) |
| 10% | SWAY | 0.322 | 1.94× | Diversity (0.224) |
| 20% | NSGA-II | 0.611 | 2.60× | Diversity (0.339) |
| 30% | NSGA-II | 0.909 | 2.76× | Diversity (0.406) |

### Table 3. Overall Method Ranking (Mean Pareto Recall Across All Budgets)

| Rank | Method | Mean Recall | Type |
|------|--------|-------------|------|
| 1 | NSGA-II | 0.500 | Baseline (3× eval) |
| 2 | SWAY | 0.421 | Baseline (100% eval) |
| — | — | — | — |
| 3 | Diversity (MaxMin) | 0.279 | Surrogate |
| 4 | Clustering | 0.228 | Surrogate |
| 5 | Stratified | 0.223 | Surrogate |
| 6 | Active Learning | 0.217 | Surrogate |
| 7 | Random | 0.214 | Surrogate |
| — | — | — | — |
| 8 | NoModel | 0.158 | Baseline (X% eval) |
| 9 | LHS | 0.138 | Surrogate |
| 10 | Sobol | 0.127 | Surrogate |

### Table 4. Full Results — Mean Pareto Recall by Method and Budget

| Method | 5% | 10% | 20% | 30% |
|--------|-----|------|------|------|
| NSGA-II | 0.158 | 0.321 | 0.611 | 0.909 |
| SWAY | 0.251 | 0.322 | 0.451 | 0.660 |
| Diversity | 0.146 | 0.224 | 0.339 | 0.406 |
| Clustering | 0.137 | 0.175 | 0.307 | 0.292 |
| Stratified | 0.148 | 0.194 | 0.231 | 0.321 |
| Active Learning | 0.134 | 0.154 | 0.270 | 0.309 |
| Random | 0.127 | 0.166 | 0.235 | 0.329 |
| NoModel | 0.064 | 0.087 | 0.201 | 0.280 |
| LHS | 0.091 | 0.102 | 0.157 | 0.204 |
| Sobol | 0.085 | 0.092 | 0.141 | 0.191 |

### Key Findings

**Objective-aware baselines dominate raw recall.** NSGA-II achieves 0.909 recall at 30% budget, and SWAY leads at 5% (0.251) and 10% (0.322). However, NSGA-II evaluates 3× more rows' objectives (pool = 3 × budget), and SWAY uses all rows' objectives for recursive binary splitting—effectively a 100% evaluation budget.

**Among equal-budget methods, Diversity wins.** Restricting to methods that evaluate exactly X% of rows, Diversity (MaxMin) is best at 10%+ and Stratified at 5%. Diversity achieves 1.17–1.44× the recall of random sampling.

**DoE failure on discrete data.** LHS and Sobol perform worse than random (Table 3). These methods generate continuous points that map poorly to discrete data—multiple design points collapse to the same data row. This is an important practical finding: practitioners defaulting to standard DoE techniques for configurable software would underperform random sampling.

**Surrogate vs. more evaluations.** All surrogate methods outperform NoModel (same budget, no model), confirming surrogate value. But NSGA-II and SWAY show that directly seeing more objective values can outperform model-based generalization—a cost–accuracy trade-off for practitioners. When objective evaluations are cheap, use NSGA-II or SWAY. When evaluations are expensive, use Diversity + surrogate.

---

## References

- [1] L. Senthilkumar and T. Menzies, "Can Large Language Models Improve SE Active Learning via Warm-Starts?," *arXiv preprint*, 2025.
- [2] Y. Jin, "Evolutionary optimization of computationally expensive problems via surrogate modeling," *Annals of Operations Research*, vol. 186, pp. 285–312, 2003.
- [3] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *JMLR*, vol. 13, pp. 281–305, 2012.
- [4] A. Sayyad, J. Ingram, T. Menzies, and H. Ammar, "On the value of user preferences in search-based software engineering," in *Proc. ICSE*, 2013, pp. 492–501.
- [5] M. Zuluaga, G. Sergent, A. Krause, and M. Puschel, "Active learning for multi-objective optimization," in *Proc. ICML*, 2013, pp. 462–470.
- [6] N. Siegmund, A. Kolesnikov, C. Kästner, S. Apel, et al., "Predicting performance via automated feature-interaction detection," in *Proc. ICSE*, 2012, pp. 167–177.
- [7] C. Henard, M. Papadakis, M. Harman, and Y. Le Traon, "Combining multi-objective search and constraint solving for configuring large SPLs," in *Proc. ICSE*, 2015, pp. 517–528.
- [8] J. Krall, T. Menzies, and M. Davies, "GALE: Geometric active learning for search-based software engineering," *IEEE TSE*, vol. 41, no. 10, pp. 1001–1018, 2015.
- [9] V. Nair, T. Menzies, N. Siegmund, and S. Apel, "Using bad learners to find good configurations," in *Proc. FSE*, 2017, pp. 257–267.
- [10] F. Sarro, F. Ferrucci, M. Harman, A. Motta, and Y. Jia, "Adaptive multi-objective evolutionary algorithms for overtime planning in software projects," *IEEE TSE*, vol. 43, no. 10, pp. 898–917, 2017.
- [11] D. Guo et al., "Data-efficient performance learning for configurable systems," *EMSE*, vol. 23, no. 3, pp. 1826–1867, 2018.
- [12] P. Jamshidi, M. Velez, C. Kästner, and N. Siegmund, "Learning to sample: Exploiting similarities across environments to learn performance models," in *Proc. FSE*, 2018, pp. 71–82.
- [13] V. Nair, Z. Yu, T. Menzies, N. Siegmund, and S. Apel, "Finding faster configurations using FLASH," *IEEE TSE*, vol. 46, no. 7, pp. 794–811, 2018.
- [14] H. Ha and L. Zhang, "DeepPerf: Performance prediction for configurable software with deep sparse neural network," in *Proc. ICSE*, 2019, pp. 1095–1106.
- [15] J. Chen, V. Nair, R. Krishna, and T. Menzies, "'Sampling' as a baseline optimizer for search-based software engineering," *IEEE TSE*, vol. 45, no. 6, pp. 597–614, 2019.
- [16] H. Wang et al., "An adaptive Bayesian approach to surrogate-assisted evolutionary multi-objective optimization," *Information Sciences*, vol. 519, pp. 317–331, 2020.
- [17] A. Lustosa and T. Menzies, "iSNEAK: Partial ordering as heuristics for model-based reasoning in SE," *IEEE Access*, 2024.
