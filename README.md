# Budget-Aware Sampling Strategies for Surrogate-Assisted MOO

Empirical comparison of fixed-budget sampling strategies for surrogate-assisted
multi-objective optimization (MOO). Covers 50 tabular datasets, 4 strategies,
4 budget levels, and 10 repeats per configuration (7,640 total experiments).

## Research Summary

Given a fixed evaluation budget of *n* configurations, which sampling strategy
best recovers the Pareto front using a surrogate model?

**Key finding:** There is a sharp crossover at *n* ≈ 100.
- At *n = 50*: **Clustering** wins (28/50 datasets, median recall 0.093)
- At *n ≥ 100*: **Diversity (MaxMin)** wins, reaching median recall 0.275 at *n = 250*

## Project Structure

```
├── run_fixed_sample_experiments.py  # Main experiment runner
├── merge_results.py                 # Aggregate results from results/ CSVs
├── requirements.txt                 # Python dependencies
├── moot/                            # Benchmark datasets (MOOT repository)
│   └── optimize/
│       ├── process/                 # SE process models (pom3*, xomo*, nasa93dem, coc1000)
│       ├── config/                  # Configuration spaces (SS-* family)
│       ├── binary_config/           # Binary config spaces (Scrum, FM/FFM variants)
│       ├── misc/                    # Misc datasets (auto93, Car_price, Wine_quality)
│       ├── sales_data/              # Sales/marketing datasets
│       ├── financial_data/          # Financial datasets (BankChurners, Telco, home_data)
│       ├── behavior_data/           # HR/behavioral datasets
│       ├── health_data/             # Health datasets
│       └── hpo/                     # HPO project health datasets
├── results/                         # Per-dataset result CSVs (one file per dataset)
└── report/                          # LaTeX paper source
    ├── report.tex
    └── references.bib
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Running Experiments

1. **Select the dataset group** in `run_fixed_sample_experiments.py`:

   ```python
   DATASET_PATHS = DATASET_PATHS_SMALL   # 93–2,938 rows  (25 datasets)
   # DATASET_PATHS = DATASET_PATHS_LARGE # 3k–100k rows   (30 datasets)
   ```

2. **Run:**

   ```bash
   python run_fixed_sample_experiments.py
   ```

   Results are saved to `results/<dataset_name>.csv` as each dataset completes.
   Re-running overwrites existing files for the selected datasets only.

3. **Merge and re-analyze** all results at any time:

   ```bash
   python merge_results.py
   ```

## Sampling Strategies

| Strategy | Description |
|---|---|
| **Random** | Uniform sampling without replacement (baseline) |
| **Stratified** | Bins the first objective into √n equal-frequency bins, samples proportionally |
| **Clustering** | k-Means with k = n; selects the row closest to each centroid |
| **Diversity** | Greedy MaxMin: each point maximizes minimum distance to already-selected set |

## Configuration

Key parameters at the top of `run_fixed_sample_experiments.py`:

| Variable | Default | Description |
|---|---|---|
| `FIXED_SAMPLE_SIZES` | `[50, 100, 200, 250]` | Evaluation budget levels |
| `N_REPEATS` | `10` | Independent repeats per (dataset, method, budget) |
| `MIN_TEST_SIZE` | `10` | Minimum rows required in the held-out test set |

## Output Columns (per result CSV)

| Column | Description |
|---|---|
| `dataset` | Dataset name |
| `method` | Sampling strategy |
| `sample_size` | Fixed budget *n* |
| `recall` | Pareto Recall — fraction of true Pareto front recovered |
| `precision` | Pareto Precision — fraction of predicted front that is truly Pareto-optimal |
| `igd` | Inverted Generational Distance (lower is better) |
| `hv_diff` | Relative hypervolume difference between true and predicted fronts |
| `true_pf_size` | Size of the true Pareto front |
| `pred_pf_size` | Size of the predicted Pareto front |

## Datasets

Datasets are split into two groups by size:

- **`DATASET_PATHS_SMALL`** (25 datasets, 93–2,938 rows): SE process models, SS-* config spaces, real-world tabular (automotive, sales, financial, health)
- **`DATASET_PATHS_LARGE`** (30 datasets, 3k–100k rows): larger SS-* variants, XOMO simulations, FM/FFM binary configurations, HPO health datasets, large process models

All datasets use the MOOT column naming convention: columns ending in `+` are
maximization objectives, `-` are minimization objectives, and `X` are ignored metadata columns.
