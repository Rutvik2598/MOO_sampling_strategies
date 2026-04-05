"""
Baseline experiments: No-Model, SWAY, NSGA-II Pool-Selection
Compatible with the existing experiment pipeline from moo_sampling_experiment.ipynb.
Outputs results in the same CSV format for merging.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

# ============================================================
# 1. SHARED INFRASTRUCTURE (copied from notebook for standalone use)
# ============================================================

def parse_moot_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    feature_cols, obj_cols, obj_directions, ignore_cols = [], [], {}, []
    for col in df.columns:
        if col.endswith('+'):
            obj_cols.append(col)
            obj_directions[col] = 'max'
        elif col.endswith('-'):
            obj_cols.append(col)
            obj_directions[col] = 'min'
        elif col.endswith('X'):
            ignore_cols.append(col)
        else:
            feature_cols.append(col)
    return df, feature_cols, obj_cols, obj_directions, ignore_cols


def encode_features(df, feature_cols):
    df_encoded = df.copy()
    for col in feature_cols:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


def normalize_objectives(obj_df, obj_directions, obj_cols):
    normed = obj_df[obj_cols].copy().astype(float)
    for col in obj_cols:
        cmin, cmax = normed[col].min(), normed[col].max()
        if cmax - cmin > 1e-12:
            normed[col] = (normed[col] - cmin) / (cmax - cmin)
        else:
            normed[col] = 0.0
        if obj_directions[col] == 'max':
            normed[col] = 1.0 - normed[col]
    return normed


def is_dominated(a, b):
    return np.all(b <= a) and np.any(b < a)


def compute_pareto_front(obj_values_normed):
    n = len(obj_values_normed)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if is_dominated(obj_values_normed[i], obj_values_normed[j]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def pareto_recall(true_pf_indices, predicted_pf_indices):
    if len(true_pf_indices) == 0:
        return 0.0
    overlap = len(true_pf_indices & predicted_pf_indices)
    return overlap / len(true_pf_indices)


def pareto_precision(true_pf_indices, predicted_pf_indices):
    if len(predicted_pf_indices) == 0:
        return 0.0
    overlap = len(true_pf_indices & predicted_pf_indices)
    return overlap / len(predicted_pf_indices)


def igd(true_pf_values, predicted_pf_values):
    if len(predicted_pf_values) == 0 or len(true_pf_values) == 0:
        return float('inf')
    dists = cdist(true_pf_values, predicted_pf_values)
    return np.mean(np.min(dists, axis=1))


def hypervolume_2d(points, ref_point):
    if len(points) == 0:
        return 0.0
    sorted_pts = points[points[:, 0].argsort()]
    hv = 0.0
    prev_y = ref_point[1]
    for pt in sorted_pts:
        if pt[0] < ref_point[0] and pt[1] < prev_y:
            hv += (ref_point[0] - pt[0]) * (prev_y - pt[1])
            prev_y = pt[1]
    return hv


def hypervolume_approx(points, ref_point, n_mc=10000, rng=None):
    if len(points) == 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(42)
    dim = points.shape[1]
    ideal = points.min(axis=0)
    random_samples = rng.uniform(ideal, ref_point, size=(n_mc, dim))
    dominated_count = 0
    for sample in random_samples:
        for pt in points:
            if np.all(pt <= sample):
                dominated_count += 1
                break
    box_volume = np.prod(ref_point - ideal)
    return box_volume * (dominated_count / n_mc)


def compute_hypervolume(points, ref_point, rng=None):
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref_point)
    return hypervolume_approx(points, ref_point, rng=rng)


# ============================================================
# 2. LOAD DATASETS (same as notebook)
# ============================================================

DATASET_PATHS = {
    "auto93":       "moot/optimize/misc/auto93.csv",
    "pom3d":        "moot/optimize/process/pom3d.csv",
    "SS-A":         "moot/optimize/config/SS-A.csv",
    "SS-B":         "moot/optimize/config/SS-B.csv",
    "Wine_quality": "moot/optimize/misc/Wine_quality.csv",
    "coc1000":      "moot/optimize/process/coc1000.csv",
}


def load_all_datasets():
    datasets = {}
    for name, path in DATASET_PATHS.items():
        df, feat_cols, obj_cols, obj_dirs, ignore_cols = parse_moot_csv(path)
        df = encode_features(df, feat_cols)
        datasets[name] = {
            'df': df,
            'feature_cols': feat_cols,
            'obj_cols': obj_cols,
            'obj_directions': obj_dirs,
            'ignore_cols': ignore_cols,
        }
        print(f"  {name}: {len(df)} rows, {len(feat_cols)} features, {len(obj_cols)} objectives")
    return datasets


def compute_true_pareto_fronts(datasets):
    true_pfs = {}
    for name, data in datasets.items():
        df = data['df']
        normed = normalize_objectives(df[data['obj_cols']], data['obj_directions'], data['obj_cols'])
        pf_indices = compute_pareto_front(normed.values)
        true_pfs[name] = set(pf_indices.tolist())
        print(f"  {name}: {len(pf_indices)} Pareto-optimal rows out of {len(df)}")
    return true_pfs


# ============================================================
# 3. BASELINE 1: NO-MODEL
#    Just compute Pareto front of the sampled subset directly.
#    No surrogate model is trained. Tests whether the model adds value.
# ============================================================

def run_nomodel_experiment(dataset_name, data, true_pf, sample_pct, rng, sampling='random'):
    """No-Model baseline: sample rows, compute PF of sample only."""
    df = data['df']
    obj_cols = data['obj_cols']
    obj_dirs = data['obj_directions']

    n = len(df)
    n_samples = max(5, int(n * sample_pct / 100))

    # Random sample
    indices = rng.choice(n, size=n_samples, replace=False)

    # Compute PF of sampled subset only
    sampled_obj = df.iloc[indices][obj_cols]
    sampled_normed = normalize_objectives(sampled_obj, obj_dirs, obj_cols).values

    # Local PF indices (within the sample)
    local_pf = compute_pareto_front(sampled_normed)
    # Map back to global indices
    predicted_pf_indices = set(indices[local_pf].tolist())

    # Evaluate against true PF
    true_obj_normed = normalize_objectives(df[obj_cols], obj_dirs, obj_cols).values

    recall = pareto_recall(true_pf, predicted_pf_indices)
    precision = pareto_precision(true_pf, predicted_pf_indices)

    true_pf_vals = true_obj_normed[list(true_pf)]
    pred_pf_vals = true_obj_normed[list(predicted_pf_indices)] if len(predicted_pf_indices) > 0 else np.empty((0, len(obj_cols)))
    igd_val = igd(true_pf_vals, pred_pf_vals)

    ref_point = np.ones(len(obj_cols)) * 1.1
    hv_true = compute_hypervolume(true_pf_vals, ref_point, rng=rng)
    hv_pred = compute_hypervolume(pred_pf_vals, ref_point, rng=rng) if len(pred_pf_vals) > 0 else 0.0
    hv_diff = abs(hv_true - hv_pred) / max(hv_true, 1e-12)

    return {
        'dataset': dataset_name,
        'method': 'NoModel',
        'sample_pct': sample_pct,
        'n_samples': n_samples,
        'recall': recall,
        'precision': precision,
        'igd': igd_val,
        'hv_diff': hv_diff,
        'pred_pf_size': len(predicted_pf_indices),
        'true_pf_size': len(true_pf),
    }


# ============================================================
# 4. BASELINE 2: SWAY (from Chen et al. 2019 "Sampling as Baseline")
#    Binary recursive random projections to find promising region.
#    Algorithm: project data onto random line, split at median,
#    keep the "better" half, recurse until budget exhausted.
# ============================================================

def sway_select(X, Y_normed, n_samples, rng):
    """
    SWAY: Sampling Way for multi-objective optimization.
    Recursively splits data using random projections, keeping the
    half whose centroid has better (lower) aggregate objective value.

    Args:
        X: feature matrix (n, d)
        Y_normed: normalized objectives (n, m), all minimizing
        n_samples: target sample size
        rng: numpy random generator

    Returns:
        Array of selected row indices
    """
    indices = np.arange(len(X))
    return _sway_recurse(X, Y_normed, indices, n_samples, rng)


def _sway_recurse(X, Y_normed, indices, n_target, rng):
    """Recursive SWAY: split and keep better half."""
    if len(indices) <= n_target:
        return indices

    # Pick two random pivot points
    i1, i2 = rng.choice(len(indices), size=2, replace=False)
    pivot1, pivot2 = X[indices[i1]], X[indices[i2]]

    # Project all points onto the line between pivots
    direction = pivot2 - pivot1
    norm_sq = np.dot(direction, direction)
    if norm_sq < 1e-12:
        # Degenerate: random split
        half = len(indices) // 2
        return _sway_recurse(X, Y_normed, indices[:half], n_target, rng)

    projections = np.array([
        np.dot(X[idx] - pivot1, direction) / norm_sq
        for idx in indices
    ])

    # Split at median
    median_proj = np.median(projections)
    left_mask = projections <= median_proj
    right_mask = ~left_mask

    left_indices = indices[left_mask]
    right_indices = indices[right_mask]

    if len(left_indices) == 0 or len(right_indices) == 0:
        # Can't split: return what we have
        return indices[:n_target]

    # Determine which half is "better" using mean objective values
    left_mean = Y_normed[left_indices].mean(axis=0).sum()
    right_mean = Y_normed[right_indices].mean(axis=0).sum()

    # Keep the better half (lower sum = better since all minimizing)
    if left_mean <= right_mean:
        better_half = left_indices
    else:
        better_half = right_indices

    return _sway_recurse(X, Y_normed, better_half, n_target, rng)


def run_sway_experiment(dataset_name, data, true_pf, sample_pct, rng):
    """SWAY baseline: use SWAY to select promising samples, compute PF."""
    df = data['df']
    feat_cols = data['feature_cols']
    obj_cols = data['obj_cols']
    obj_dirs = data['obj_directions']

    X = df[feat_cols].values.astype(float)
    Y = df[obj_cols].values.astype(float)
    n = len(df)
    n_samples = max(5, int(n * sample_pct / 100))

    # Normalize features for distance computation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize objectives for SWAY decision-making
    Y_normed = normalize_objectives(df[obj_cols], obj_dirs, obj_cols).values

    # Run SWAY to select samples
    selected = sway_select(X_scaled, Y_normed, n_samples, rng)
    selected = selected[:n_samples]  # Ensure exact budget

    # Compute PF of selected subset (no surrogate model — SWAY IS the optimizer)
    sampled_normed = Y_normed[selected]
    local_pf = compute_pareto_front(sampled_normed)
    predicted_pf_indices = set(selected[local_pf].tolist())

    # Evaluate against true PF
    true_obj_normed = normalize_objectives(df[obj_cols], obj_dirs, obj_cols).values

    recall = pareto_recall(true_pf, predicted_pf_indices)
    precision = pareto_precision(true_pf, predicted_pf_indices)

    true_pf_vals = true_obj_normed[list(true_pf)]
    pred_pf_vals = true_obj_normed[list(predicted_pf_indices)] if len(predicted_pf_indices) > 0 else np.empty((0, len(obj_cols)))
    igd_val = igd(true_pf_vals, pred_pf_vals)

    ref_point = np.ones(len(obj_cols)) * 1.1
    hv_true = compute_hypervolume(true_pf_vals, ref_point, rng=rng)
    hv_pred = compute_hypervolume(pred_pf_vals, ref_point, rng=rng) if len(pred_pf_vals) > 0 else 0.0
    hv_diff = abs(hv_true - hv_pred) / max(hv_true, 1e-12)

    return {
        'dataset': dataset_name,
        'method': 'SWAY',
        'sample_pct': sample_pct,
        'n_samples': n_samples,
        'recall': recall,
        'precision': precision,
        'igd': igd_val,
        'hv_diff': hv_diff,
        'pred_pf_size': len(predicted_pf_indices),
        'true_pf_size': len(true_pf),
    }


# ============================================================
# 5. BASELINE 3: NSGA-II POOL SELECTION
#    Use NSGA-II (from pymoo) to select from the existing pool.
#    This is NOT running NSGA-II as a generative optimizer — it selects
#    the best subset of existing rows using non-dominated sorting
#    and crowding distance, limited to the same evaluation budget.
# ============================================================

def nsga2_pool_select(Y_normed, n_samples, rng):
    """
    NSGA-II-style pool selection: rank all rows by non-dominated sorting
    + crowding distance, select the top n_samples.

    This simulates what NSGA-II would pick if restricted to evaluating
    only n_samples points from the pool.
    """
    n = len(Y_normed)
    if n_samples >= n:
        return np.arange(n)

    # Step 1: Non-dominated sorting (assign fronts)
    fronts = _fast_non_dominated_sort(Y_normed)

    # Step 2: Fill selection front-by-front with crowding distance tie-breaking
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n_samples:
            selected.extend(front)
        else:
            # Need subset of this front — use crowding distance
            remaining = n_samples - len(selected)
            if remaining > 0:
                cd = _crowding_distance(Y_normed[front])
                # Sort by descending crowding distance (prefer diverse points)
                sorted_idx = np.argsort(-cd)
                selected.extend([front[i] for i in sorted_idx[:remaining]])
            break

    return np.array(selected)


def _fast_non_dominated_sort(obj_values):
    """Fast non-dominated sorting (NSGA-II style). Returns list of fronts."""
    n = len(obj_values)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(obj_values[i], obj_values[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(obj_values[j], obj_values[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


def _dominates(a, b):
    """Check if a dominates b (all objectives to minimize)."""
    return np.all(a <= b) and np.any(a < b)


def _crowding_distance(obj_values):
    """Compute crowding distance for a set of points."""
    n, m = obj_values.shape
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    for j in range(m):
        sorted_idx = np.argsort(obj_values[:, j])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        obj_range = obj_values[sorted_idx[-1], j] - obj_values[sorted_idx[0], j]
        if obj_range < 1e-12:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                obj_values[sorted_idx[k + 1], j] - obj_values[sorted_idx[k - 1], j]
            ) / obj_range

    return distances


def run_nsga2_experiment(dataset_name, data, true_pf, sample_pct, rng):
    """NSGA-II pool selection baseline."""
    df = data['df']
    obj_cols = data['obj_cols']
    obj_dirs = data['obj_directions']

    n = len(df)
    n_samples = max(5, int(n * sample_pct / 100))

    # Normalize objectives
    Y_normed = normalize_objectives(df[obj_cols], obj_dirs, obj_cols).values

    # NSGA-II pool selection: non-dominated sorting + crowding distance
    # Note: This uses ALL objective values (full info) — it's an oracle-like baseline
    # showing the ceiling of what evaluation-budget-limited selection can achieve.
    # To make it fair, we add noise: randomly subsample 3× budget, then apply NSGA-II.
    pool_size = min(n, n_samples * 3)
    pool_indices = rng.choice(n, size=pool_size, replace=False)
    pool_objs = Y_normed[pool_indices]

    selected_local = nsga2_pool_select(pool_objs, n_samples, rng)
    selected_global = pool_indices[selected_local]

    # Compute PF of selected subset
    selected_normed = Y_normed[selected_global]
    local_pf = compute_pareto_front(selected_normed)
    predicted_pf_indices = set(selected_global[local_pf].tolist())

    # Evaluate
    true_obj_normed = Y_normed
    recall = pareto_recall(true_pf, predicted_pf_indices)
    precision = pareto_precision(true_pf, predicted_pf_indices)

    true_pf_vals = true_obj_normed[list(true_pf)]
    pred_pf_vals = true_obj_normed[list(predicted_pf_indices)] if len(predicted_pf_indices) > 0 else np.empty((0, len(obj_cols)))
    igd_val = igd(true_pf_vals, pred_pf_vals)

    ref_point = np.ones(len(obj_cols)) * 1.1
    hv_true = compute_hypervolume(true_pf_vals, ref_point, rng=rng)
    hv_pred = compute_hypervolume(pred_pf_vals, ref_point, rng=rng) if len(pred_pf_vals) > 0 else 0.0
    hv_diff = abs(hv_true - hv_pred) / max(hv_true, 1e-12)

    return {
        'dataset': dataset_name,
        'method': 'NSGA-II',
        'sample_pct': sample_pct,
        'n_samples': n_samples,
        'recall': recall,
        'precision': precision,
        'igd': igd_val,
        'hv_diff': hv_diff,
        'pred_pf_size': len(predicted_pf_indices),
        'true_pf_size': len(true_pf),
    }


# ============================================================
# 6. MAIN: RUN ALL BASELINES
# ============================================================

SAMPLE_PERCENTAGES = [5, 10, 20, 30]
N_REPEATS = 20

BASELINE_RUNNERS = {
    'NoModel': run_nomodel_experiment,
    'SWAY':    run_sway_experiment,
    'NSGA-II': run_nsga2_experiment,
}


def main():
    print("=" * 70)
    print("BASELINE EXPERIMENTS: NoModel, SWAY, NSGA-II Pool Selection")
    print("=" * 70)

    print("\nLoading datasets...")
    datasets = load_all_datasets()

    print("\nComputing true Pareto fronts...")
    true_pfs = compute_true_pareto_fronts(datasets)

    all_results = []
    total_combos = len(datasets) * len(BASELINE_RUNNERS) * len(SAMPLE_PERCENTAGES)
    completed = 0
    start_time = time.time()

    for ds_name, data in datasets.items():
        for method_name, runner in BASELINE_RUNNERS.items():
            for pct in SAMPLE_PERCENTAGES:
                for rep in range(N_REPEATS):
                    rng = np.random.default_rng(
                        seed=rep * 1000 + hash(ds_name + method_name) % 10000 + pct
                    )
                    result = runner(ds_name, data, true_pfs[ds_name], pct, rng)
                    if result is not None:
                        result['repeat'] = rep
                        all_results.append(result)

                completed += 1
                if completed % 6 == 0 or completed == total_combos:
                    elapsed = time.time() - start_time
                    pct_done = 100 * completed / total_combos
                    print(f"  Progress: {completed}/{total_combos} ({pct_done:.0f}%) | "
                          f"{elapsed:.1f}s | {ds_name}/{method_name}/{pct}%")

    baseline_df = pd.DataFrame(all_results)
    elapsed_total = time.time() - start_time

    print(f"\nBaseline experiments done: {len(baseline_df)} results in {elapsed_total:.1f}s")

    # Save baseline results
    baseline_df.to_csv("baseline_raw_results.csv", index=False)
    print(f"Saved: baseline_raw_results.csv ({len(baseline_df)} rows)")

    # Merge with original results
    if os.path.exists("experiment_raw_results.csv"):
        original_df = pd.read_csv("experiment_raw_results.csv")
        merged_df = pd.concat([original_df, baseline_df], ignore_index=True)
        merged_df.to_csv("experiment_raw_results_with_baselines.csv", index=False)
        print(f"Saved merged: experiment_raw_results_with_baselines.csv ({len(merged_df)} rows)")

        # Recompute aggregated stats
        metrics = ['recall', 'precision', 'igd', 'hv_diff']
        group_cols = ['dataset', 'method', 'sample_pct']
        agg_funcs = {m: ['mean', 'median', 'std', 'min', 'max'] for m in metrics}
        agg_funcs['pred_pf_size'] = 'mean'
        agg_funcs['true_pf_size'] = 'first'

        stats_df = merged_df.groupby(group_cols).agg(agg_funcs).round(4)
        stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns]
        stats_df = stats_df.reset_index()
        stats_df.to_csv("experiment_aggregated_stats_with_baselines.csv", index=False)
        print(f"Saved: experiment_aggregated_stats_with_baselines.csv ({len(stats_df)} rows)")

        # Overall summary (method × sample_pct, averaged across datasets)
        overall = merged_df.groupby(['method', 'sample_pct']).agg({
            'recall': ['mean', 'std'],
            'precision': 'mean',
            'igd': 'mean',
            'hv_diff': 'mean',
        }).round(4)
        overall.columns = ['_'.join(col).strip('_') for col in overall.columns]
        overall = overall.reset_index()
        overall.to_csv("experiment_overall_summary_with_baselines.csv", index=False)
        print(f"Saved: experiment_overall_summary_with_baselines.csv")

        # Print summary table
        print("\n" + "=" * 90)
        print("OVERALL RESULTS (all methods, averaged across datasets)")
        print("=" * 90)
        for pct in SAMPLE_PERCENTAGES:
            print(f"\n--- At {pct}% sampling budget ---")
            pct_data = overall[overall['sample_pct'] == pct].sort_values('recall_mean', ascending=False)
            for _, row in pct_data.iterrows():
                print(f"  {row['method']:15s} | Recall={row['recall_mean']:.4f} ± {row['recall_std']:.4f} | "
                      f"Precision={row['precision_mean']:.4f} | IGD={row['igd_mean']:.4f} | HV_Diff={row['hv_diff_mean']:.4f}")

    else:
        print("Warning: experiment_raw_results.csv not found. Only baseline results saved.")

    # Print baseline-specific summary
    print("\n" + "=" * 70)
    print("BASELINE-ONLY SUMMARY")
    print("=" * 70)
    for method in BASELINE_RUNNERS:
        method_data = baseline_df[baseline_df['method'] == method]
        for pct in SAMPLE_PERCENTAGES:
            pct_data = method_data[method_data['sample_pct'] == pct]
            if len(pct_data) > 0:
                print(f"  {method:10s} @ {pct:2d}%: "
                      f"Recall={pct_data['recall'].mean():.4f} ± {pct_data['recall'].std():.4f}, "
                      f"IGD={pct_data['igd'].mean():.4f}, "
                      f"HV_Diff={pct_data['hv_diff'].mean():.4f}")


if __name__ == "__main__":
    main()
