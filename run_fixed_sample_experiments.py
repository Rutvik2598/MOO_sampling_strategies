#!/usr/bin/env python3
"""
Fixed Sample Size MOO Sampling Experiments

Compares 7 sampling strategies using fixed sample sizes (50, 100, 200, 250)
instead of percentage-based sampling.

Usage:
    python run_fixed_sample_experiments.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import product
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans

from scipy.stats import qmc
from scipy.spatial.distance import cdist

from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

FIXED_SAMPLE_SIZES = [50, 100, 200, 250]
N_REPEATS = 20
MIN_TEST_SIZE = 10  # Minimum test set size required

MOOT_ROOT = Path("moot/optimize")

# LARGE DATASETS - For scalability testing
DATASET_PATHS_LARGE = {
    "pom3a":        MOOT_ROOT / "process" / "pom3a.csv",         # 20,000 rows, 3 obj
    "xomo_flight":  MOOT_ROOT / "process" / "xomo_flight.csv",   # 10,000 rows, 4 obj
    "Scrum100k":    MOOT_ROOT / "binary_config" / "Scrum100k.csv", # 100,000 rows, 3 obj
    "SS-W":         MOOT_ROOT / "config" / "SS-W.csv",           # 65,536 rows, 2 obj
}

# ALL DATASETS - Combined collection (small-medium)
DATASET_PATHS_ALL = {
    # Original 6 datasets
    "auto93":       MOOT_ROOT / "misc" / "auto93.csv",           # 398 rows, 3 obj
    "pom3d":        MOOT_ROOT / "process" / "pom3d.csv",         # 500 rows, 3 obj
    "SS-A":         MOOT_ROOT / "config" / "SS-A.csv",           # 1343 rows, 2 obj
    "SS-B":         MOOT_ROOT / "config" / "SS-B.csv",           # 206 rows, 2 obj
    "Wine_quality": MOOT_ROOT / "misc" / "Wine_quality.csv",     # 1599 rows, 2 obj
    "coc1000":      MOOT_ROOT / "process" / "coc1000.csv",       # 1000 rows, 5 obj
    # New datasets (moderate size, diverse domains)
    "SS-M":         MOOT_ROOT / "config" / "SS-M.csv",           # 864 rows, 3 obj (database)
    "SS-C":         MOOT_ROOT / "config" / "SS-C.csv",           # 1512 rows, 2 obj (stream)
    "SS-I":         MOOT_ROOT / "config" / "SS-I.csv",           # 1080 rows, 2 obj (stream)
    "home_data":    MOOT_ROOT / "financial_data" / "home_data_for_ml_course.csv",  # 1459 rows, 4 obj
    "HR_Attrition": MOOT_ROOT / "behavior_data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",  # 1469 rows, 3 obj
    "accessories":  MOOT_ROOT / "sales_data" / "accessories.csv", # 1120 rows, 2 obj
}

# SELECT WHICH DATASET SET TO RUN
DATASET_PATHS = DATASET_PATHS_LARGE  # <- Change to DATASET_PATHS_ALL for small datasets

# ============================================================================
# DATASET LOADING
# ============================================================================

def parse_moot_csv(filepath):
    """Parse a MOOT CSV file and separate features, objectives, and metadata."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    feature_cols = []
    obj_cols = []
    obj_directions = {}
    ignore_cols = []
    
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
    """Encode categorical features as numeric."""
    df_encoded = df.copy()
    encoders = {}
    for col in feature_cols:
        # Try to convert to numeric, if it fails then encode
        try:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='raise')
            # Fill numeric NaN with median
            if df_encoded[col].isna().any():
                df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
        except (ValueError, TypeError):
            le = LabelEncoder()
            # Handle NaN values by converting to string first
            df_encoded[col] = df_encoded[col].fillna('_MISSING_').astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    return df_encoded, encoders


# ============================================================================
# PARETO FRONT COMPUTATION
# ============================================================================

def normalize_objectives(obj_values, obj_directions, obj_cols):
    """Normalize objective values to [0,1] and flip signs so all objectives become 'minimize'."""
    normed = obj_values.copy().astype(float)
    for col in obj_cols:
        col_min = normed[col].min()
        col_max = normed[col].max()
        if col_max - col_min > 1e-12:
            normed[col] = (normed[col] - col_min) / (col_max - col_min)
        else:
            normed[col] = 0.0
        if obj_directions[col] == 'max':
            normed[col] = 1.0 - normed[col]
    return normed


def is_dominated(row_a, row_b):
    """Check if row_a is dominated by row_b (all objectives minimized)."""
    return np.all(row_b <= row_a) and np.any(row_b < row_a)


def compute_pareto_front(obj_values_normed):
    """Compute indices of Pareto-optimal rows (non-dominated)."""
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


# ============================================================================
# SAMPLING STRATEGIES
# ============================================================================

def sample_random(X, n_samples, rng):
    """Simple random sampling without replacement."""
    indices = rng.choice(len(X), size=n_samples, replace=False)
    return np.sort(indices)


def sample_stratified(X, y_obj, n_samples, rng):
    """Stratified sampling: bin objective space, sample proportionally from each bin."""
    n_bins = max(2, int(np.sqrt(n_samples)))
    y_first = y_obj[:, 0]
    bin_edges = np.percentile(y_first, np.linspace(0, 100, n_bins + 1))
    bin_labels = np.digitize(y_first, bin_edges[:-1]) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)
    
    unique_bins = np.unique(bin_labels)
    selected = []
    samples_per_bin = max(1, n_samples // len(unique_bins))
    
    for b in unique_bins:
        bin_indices = np.where(bin_labels == b)[0]
        n_take = min(samples_per_bin, len(bin_indices))
        chosen = rng.choice(bin_indices, size=n_take, replace=False)
        selected.extend(chosen)
    
    selected = list(set(selected))
    if len(selected) < n_samples:
        remaining = list(set(range(len(X))) - set(selected))
        extra = rng.choice(remaining, size=min(n_samples - len(selected), len(remaining)), replace=False)
        selected.extend(extra)
    
    return np.sort(np.array(selected[:n_samples]))


def sample_clustering(X, n_samples, rng):
    """K-Means clustering: cluster into n_samples groups, pick closest to centroid."""
    n_clusters = min(n_samples, len(X))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng.integers(0, 10000), n_init=5)
    kmeans.fit(X_scaled)
    
    selected = []
    for c in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == c)[0]
        if len(cluster_indices) == 0:
            continue
        dists = cdist(X_scaled[cluster_indices], [kmeans.cluster_centers_[c]])
        best = cluster_indices[np.argmin(dists)]
        selected.append(best)
    
    selected = list(set(selected))
    if len(selected) < n_samples:
        remaining = list(set(range(len(X))) - set(selected))
        extra = rng.choice(remaining, size=min(n_samples - len(selected), len(remaining)), replace=False)
        selected.extend(extra)
    
    return np.sort(np.array(selected[:n_samples]))


def sample_diversity_maxmin(X, n_samples, rng):
    """MaxMin diversity sampling: iteratively pick the point farthest from selected set."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(X)
    
    selected = [rng.integers(0, n)]
    min_dists = cdist(X_scaled, X_scaled[selected]).flatten()
    
    for _ in range(n_samples - 1):
        candidate = np.argmax(min_dists)
        selected.append(candidate)
        new_dists = cdist(X_scaled, X_scaled[[candidate]]).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected] = -1
    
    return np.sort(np.array(selected))


def sample_lhs(X, n_samples, rng):
    """Latin Hypercube Sampling: space-filling design, map to nearest data points."""
    n_features = X.shape[1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    sampler = qmc.LatinHypercube(d=n_features, seed=rng.integers(0, 10000))
    lhs_points = sampler.random(n=n_samples)
    
    dists = cdist(lhs_points, X_scaled)
    selected = []
    available = set(range(len(X)))
    
    for i in range(n_samples):
        sorted_candidates = np.argsort(dists[i])
        for candidate in sorted_candidates:
            if candidate in available:
                selected.append(candidate)
                available.discard(candidate)
                break
    
    return np.sort(np.array(selected))


def sample_sobol(X, n_samples, rng):
    """Sobol quasi-random sequence: low-discrepancy sampling mapped to nearest data points."""
    n_features = X.shape[1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    sampler = qmc.Sobol(d=n_features, scramble=True, seed=rng.integers(0, 10000))
    n_pow2 = max(4, 2 ** int(np.ceil(np.log2(n_samples))))
    sobol_points = sampler.random(n=n_pow2)[:n_samples]
    
    dists = cdist(sobol_points, X_scaled)
    selected = []
    available = set(range(len(X)))
    
    for i in range(n_samples):
        sorted_candidates = np.argsort(dists[i])
        for candidate in sorted_candidates:
            if candidate in available:
                selected.append(candidate)
                available.discard(candidate)
                break
    
    return np.sort(np.array(selected))


def sample_active_learning(X, y_obj, n_samples, rng):
    """Uncertainty-based active learning: start small, iteratively add most uncertain points."""
    n = len(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    seed_size = max(5, n_samples // 5)
    seed_indices = list(rng.choice(n, size=seed_size, replace=False))
    remaining_budget = n_samples - seed_size
    
    selected = list(seed_indices)
    pool = list(set(range(n)) - set(selected))
    
    while remaining_budget > 0 and len(pool) > 0:
        rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        rf.fit(X_scaled[selected], y_obj[selected])
        
        pool_arr = np.array(pool)
        predictions = np.array([tree.predict(X_scaled[pool_arr]) for tree in rf.estimators_])
        uncertainty = predictions.std(axis=0).mean(axis=1)
        
        batch_size = max(1, min(remaining_budget, max(3, remaining_budget // 3)))
        top_uncertain_idx = np.argsort(uncertainty)[-batch_size:]
        new_points = pool_arr[top_uncertain_idx].tolist()
        
        selected.extend(new_points)
        pool = [p for p in pool if p not in set(new_points)]
        remaining_budget -= len(new_points)
    
    return np.sort(np.array(selected[:n_samples]))


SAMPLING_METHODS = {
    'Random':          lambda X, y, n, rng: sample_random(X, n, rng),
    'Stratified':      lambda X, y, n, rng: sample_stratified(X, y, n, rng),
    'Clustering':      lambda X, y, n, rng: sample_clustering(X, n, rng),
    'Diversity':       lambda X, y, n, rng: sample_diversity_maxmin(X, n, rng),
    'LHS':             lambda X, y, n, rng: sample_lhs(X, n, rng),
    'Sobol':           lambda X, y, n, rng: sample_sobol(X, n, rng),
    'ActiveLearning':  lambda X, y, n, rng: sample_active_learning(X, y, n, rng),
}


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def pareto_recall(true_pf_indices, predicted_pf_indices):
    """% of true Pareto front rows found in predicted front."""
    if len(true_pf_indices) == 0:
        return 0.0
    overlap = len(true_pf_indices & predicted_pf_indices)
    return overlap / len(true_pf_indices)


def pareto_precision(true_pf_indices, predicted_pf_indices):
    """% of predicted Pareto front rows that are truly Pareto-optimal."""
    if len(predicted_pf_indices) == 0:
        return 0.0
    overlap = len(true_pf_indices & predicted_pf_indices)
    return overlap / len(predicted_pf_indices)


def igd(true_pf_values, predicted_pf_values):
    """Inverted Generational Distance: avg min-distance from each true PF point to predicted PF."""
    if len(predicted_pf_values) == 0 or len(true_pf_values) == 0:
        return float('inf')
    dists = cdist(true_pf_values, predicted_pf_values)
    return np.mean(np.min(dists, axis=1))


def hypervolume_2d(points, ref_point):
    """Compute hypervolume for 2D case."""
    if len(points) == 0:
        return 0.0
    sorted_pts = points[points[:, 0].argsort()]
    hv = 0.0
    prev_y = ref_point[1]
    for pt in sorted_pts:
        if pt[0] < ref_point[0] and pt[1] < ref_point[1]:
            hv += (ref_point[0] - pt[0]) * (prev_y - pt[1])
            prev_y = pt[1]
    return hv


def hypervolume_approx(points, ref_point, n_mc=10000, rng=None):
    """Monte Carlo approximation of hypervolume for >2D case."""
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
    """Dispatch to exact 2D or approximate MC hypervolume."""
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref_point)
    return hypervolume_approx(points, ref_point, rng=rng)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(dataset_name, data, true_pf, sampling_method_name, n_samples, rng):
    """Run one experiment: sample → train → predict → evaluate."""
    df = data['df']
    feat_cols = data['feature_cols']
    obj_cols = data['obj_cols']
    obj_dirs = data['obj_directions']
    
    X = df[feat_cols].values.astype(float)
    Y = df[obj_cols].values.astype(float)
    n = len(df)
    
    # Skip if not enough data
    if n_samples >= n - MIN_TEST_SIZE:
        return None
    
    # --- Step 1: Sample ---
    sampler = SAMPLING_METHODS[sampling_method_name]
    train_idx = sampler(X, Y, n_samples, rng)
    test_idx = np.array([i for i in range(n) if i not in set(train_idx)])
    
    if len(test_idx) < MIN_TEST_SIZE:
        return None
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # --- Step 2: Train model ---
    model = RandomForestRegressor(n_estimators=100, random_state=rng.integers(0, 100000), n_jobs=-1)
    model.fit(X_train, Y_train)
    
    # --- Step 3: Predict on unseen data ---
    Y_pred = model.predict(X_test)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)
    
    # --- Step 4: Compute predicted Pareto front ---
    Y_full_pred = np.zeros_like(Y)
    Y_full_pred[train_idx] = Y_train
    Y_full_pred[test_idx] = Y_pred
    
    pred_df = pd.DataFrame(Y_full_pred, columns=obj_cols)
    pred_normed = normalize_objectives(pred_df, obj_dirs, obj_cols).values
    predicted_pf_indices = set(compute_pareto_front(pred_normed))
    
    # --- Step 5: Evaluation metrics ---
    true_obj_normed = normalize_objectives(df[obj_cols], obj_dirs, obj_cols).values
    
    recall = pareto_recall(true_pf, predicted_pf_indices)
    precision = pareto_precision(true_pf, predicted_pf_indices)
    
    true_pf_vals = true_obj_normed[list(true_pf)]
    pred_pf_vals = pred_normed[list(predicted_pf_indices)] if len(predicted_pf_indices) > 0 else np.empty((0, len(obj_cols)))
    igd_val = igd(true_pf_vals, pred_pf_vals)
    
    ref_point = np.ones(len(obj_cols)) * 1.1
    hv_true = compute_hypervolume(true_pf_vals, ref_point, rng=rng)
    hv_pred = compute_hypervolume(pred_pf_vals, ref_point, rng=rng) if len(pred_pf_vals) > 0 else 0.0
    hv_diff = abs(hv_true - hv_pred) / max(hv_true, 1e-12)
    
    return {
        'dataset': dataset_name,
        'method': sampling_method_name,
        'sample_size': n_samples,
        'sample_pct': round(100 * n_samples / n, 2),
        'dataset_size': n,
        'recall': recall,
        'precision': precision,
        'igd': igd_val,
        'hv_diff': hv_diff,
        'pred_pf_size': len(predicted_pf_indices),
        'true_pf_size': len(true_pf),
    }


def main():
    print("=" * 70)
    print("  FIXED SAMPLE SIZE MOO SAMPLING EXPERIMENTS")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sample sizes: {FIXED_SAMPLE_SIZES}")
    print(f"Repeats per config: {N_REPEATS}")
    print(f"Sampling methods: {list(SAMPLING_METHODS.keys())}")
    print()
    
    # --- Load datasets ---
    print("Loading datasets...")
    datasets = {}
    for name, path in DATASET_PATHS.items():
        df, feat_cols, obj_cols, obj_dirs, ignore_cols = parse_moot_csv(path)
        df_enc, encoders = encode_features(df, feat_cols)
        datasets[name] = {
            'df': df_enc,
            'feature_cols': feat_cols,
            'obj_cols': obj_cols,
            'obj_directions': obj_dirs,
            'ignore_cols': ignore_cols,
        }
        print(f"  ✓ {name:15s} | rows={len(df):5d} | features={len(feat_cols):2d} | objectives={len(obj_cols)}")
    
    # --- Compute true Pareto fronts ---
    print("\nComputing true Pareto fronts...")
    true_pareto_fronts = {}
    for name, data in datasets.items():
        obj_vals = data['df'][data['obj_cols']]
        obj_normed = normalize_objectives(obj_vals, data['obj_directions'], data['obj_cols'])
        pf_indices = compute_pareto_front(obj_normed.values)
        true_pareto_fronts[name] = set(pf_indices)
        print(f"  ✓ {name:15s} | Pareto front: {len(pf_indices):4d} / {len(data['df'])} ({100*len(pf_indices)/len(data['df']):.1f}%)")
    
    # --- Determine valid experiment configurations ---
    print("\nDetermining valid experiment configurations...")
    experiments = []
    for ds_name, data in datasets.items():
        ds_size = len(data['df'])
        for sample_size in FIXED_SAMPLE_SIZES:
            if sample_size >= ds_size - MIN_TEST_SIZE:
                print(f"  ⚠ Skipping {ds_name} (n={ds_size}) for sample_size={sample_size} (insufficient test data)")
                continue
            for method_name in SAMPLING_METHODS:
                for rep in range(N_REPEATS):
                    experiments.append({
                        'dataset': ds_name,
                        'method': method_name,
                        'sample_size': sample_size,
                        'repeat': rep,
                    })
    
    total_experiments = len(experiments)
    print(f"\nTotal experiments to run: {total_experiments}")
    print()
    
    # --- Run experiments with progress bar ---
    print("=" * 70)
    print("  RUNNING EXPERIMENTS")
    print("=" * 70)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # Process each dataset separately and save to its own file
    for ds_name in tqdm(list(datasets.keys()), desc="Datasets", unit="ds"):
        data = datasets[ds_name]
        true_pf = true_pareto_fronts[ds_name]
        ds_size = len(data['df'])
        
        dataset_results = []
        
        # Build configs for this dataset
        configs = []
        for sample_size in FIXED_SAMPLE_SIZES:
            if sample_size >= ds_size - MIN_TEST_SIZE:
                continue
            for method_name in SAMPLING_METHODS:
                configs.append((method_name, sample_size))
        
        for method_name, sample_size in tqdm(configs, desc=f"  {ds_name}", leave=False, unit="cfg"):
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(seed=rep * 1000 + hash(ds_name + method_name) % 10000 + sample_size)
                result = run_single_experiment(ds_name, data, true_pf, method_name, sample_size, rng)
                if result is not None:
                    result['repeat'] = rep
                    dataset_results.append(result)
        
        # Save this dataset's results (overwrites if exists)
        if dataset_results:
            ds_df = pd.DataFrame(dataset_results)
            ds_df.to_csv(results_dir / f"{ds_name}.csv", index=False)
    
    elapsed_total = time.time() - start_time
    
    # --- Load all results from results/ folder ---
    all_csvs = list(results_dir.glob("*.csv"))
    results_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    
    print()
    print("=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Datasets processed: {len(all_csvs)}")
    print(f"Total results collected: {len(results_df)}")
    print(f"Total runtime: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"Results saved to: results/*.csv")
    print()
    
    # --- Compute aggregated statistics ---
    metrics = ['recall', 'precision', 'igd', 'hv_diff']
    group_cols = ['dataset', 'method', 'sample_size']
    
    agg_funcs = {m: ['mean', 'median', 'std', 'min', 'max'] for m in metrics}
    agg_funcs['sample_pct'] = 'first'
    agg_funcs['dataset_size'] = 'first'
    agg_funcs['pred_pf_size'] = 'mean'
    agg_funcs['true_pf_size'] = 'first'
    
    stats_df = results_df.groupby(group_cols).agg(agg_funcs).round(4)
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns]
    stats_df = stats_df.reset_index()
    
    # --- Compute overall summary (averaged across datasets) ---
    overall_summary = results_df.groupby(['method', 'sample_size']).agg({
        'recall': ['mean', 'median', 'std'],
        'precision': ['mean', 'median', 'std'],
        'igd': ['mean', 'median', 'std'],
        'hv_diff': ['mean', 'median', 'std'],
    }).round(4)
    overall_summary.columns = ['_'.join(col) for col in overall_summary.columns]
    overall_summary = overall_summary.reset_index()
    
    # --- Compute WIN COUNTS per dataset (more robust than mean) ---
    print()
    print("=" * 70)
    print("  WIN COUNTS: Best Method per Dataset & Sample Size")
    print("=" * 70)
    
    win_counts = defaultdict(lambda: defaultdict(int))
    for ds_name in results_df['dataset'].unique():
        for ss in FIXED_SAMPLE_SIZES:
            subset = results_df[(results_df['dataset'] == ds_name) & (results_df['sample_size'] == ss)]
            if len(subset) == 0:
                continue
            # Find best method by median recall for this dataset/size combo
            method_medians = subset.groupby('method')['recall'].median()
            best_method = method_medians.idxmax()
            win_counts[ss][best_method] += 1
    
    # Print win count table
    print(f"\n{'Method':<15}", end="")
    for ss in FIXED_SAMPLE_SIZES:
        print(f"{ss:>8}", end="")
    print("   Total")
    print("-" * 60)
    
    for method in SAMPLING_METHODS.keys():
        total_wins = sum(win_counts[ss][method] for ss in FIXED_SAMPLE_SIZES)
        print(f"{method:<15}", end="")
        for ss in FIXED_SAMPLE_SIZES:
            print(f"{win_counts[ss][method]:>8}", end="")
        print(f"   {total_wins:>5}")
    
    # --- Print MEDIAN Recall (more robust) ---
    print()
    print("=" * 70)
    print("  MEDIAN Pareto Recall by Method & Sample Size")
    print("=" * 70)
    
    pivot_median = results_df.pivot_table(
        values='recall', 
        index='method', 
        columns='sample_size', 
        aggfunc='median'
    ).round(3)
    print(pivot_median.to_string())
    
    print()
    print("=" * 70)
    print("  BEST METHOD PER SAMPLE SIZE (by Median Recall)")
    print("=" * 70)
    
    for ss in FIXED_SAMPLE_SIZES:
        ss_data = results_df[results_df['sample_size'] == ss]
        if len(ss_data) == 0:
            continue
        best = ss_data.groupby('method')['recall'].median().idxmax()
        best_val = ss_data.groupby('method')['recall'].median().max()
        print(f"  Sample size {ss:3d}: {best:15s} (median recall={best_val:.3f})")
    
    # --- Supplementary: Mean Recall ---
    print()
    print("=" * 70)
    print("  SUPPLEMENTARY: Mean Pareto Recall by Method & Sample Size")
    print("=" * 70)
    
    pivot_mean = results_df.pivot_table(
        values='recall', 
        index='method', 
        columns='sample_size', 
        aggfunc='mean'
    ).round(3)
    print(pivot_mean.to_string())
    
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Done!")
    

if __name__ == "__main__":
    main()
