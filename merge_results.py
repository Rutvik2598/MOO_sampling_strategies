#!/usr/bin/env python3
"""Merge and analyze experiment results."""

import pandas as pd
from collections import defaultdict

# Load merged data
df = pd.read_csv('all_experiments_raw.csv')
print(f"Loaded {len(df)} rows from {df['dataset'].nunique()} datasets")
print(f"Datasets: {sorted(df['dataset'].unique())}")

# Aggregated stats per (dataset, method, sample_size)
metrics = ['recall', 'precision', 'igd', 'hv_diff']
group_cols = ['dataset', 'method', 'sample_size']

agg_funcs = {m: ['mean', 'median', 'std', 'min', 'max'] for m in metrics}
agg_funcs['sample_pct'] = 'first'
agg_funcs['dataset_size'] = 'first'
agg_funcs['pred_pf_size'] = 'mean'
agg_funcs['true_pf_size'] = 'first'

stats_df = df.groupby(group_cols).agg(agg_funcs).round(4)
stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns]
stats_df = stats_df.reset_index()
stats_df.to_csv('all_experiments_stats.csv', index=False)
print(f"✓ Saved: all_experiments_stats.csv ({len(stats_df)} rows)")

# Overall summary
overall = df.groupby(['method', 'sample_size']).agg({
    'recall': ['mean', 'median', 'std'],
    'precision': ['mean', 'median', 'std'],
    'igd': ['mean', 'median', 'std'],
    'hv_diff': ['mean', 'median', 'std'],
}).round(4)
overall.columns = ['_'.join(col) for col in overall.columns]
overall = overall.reset_index()
overall.to_csv('all_experiments_summary.csv', index=False)
print(f"✓ Saved: all_experiments_summary.csv ({len(overall)} rows)")

# WIN COUNTS
FIXED_SAMPLE_SIZES = [50, 100, 200, 250]
methods = sorted(df['method'].unique())

win_counts = defaultdict(lambda: defaultdict(int))
for ds_name in df['dataset'].unique():
    for ss in FIXED_SAMPLE_SIZES:
        subset = df[(df['dataset'] == ds_name) & (df['sample_size'] == ss)]
        if len(subset) == 0:
            continue
        method_medians = subset.groupby('method')['recall'].median()
        best_method = method_medians.idxmax()
        win_counts[ss][best_method] += 1

print("\n" + "=" * 60)
print("  WIN COUNTS: Best Method per Dataset & Sample Size")
print("=" * 60)
print(f"\n{'Method':<15}", end="")
for ss in FIXED_SAMPLE_SIZES:
    print(f"{ss:>8}", end="")
print("   Total")
print("-" * 60)

for method in methods:
    total_wins = sum(win_counts[ss][method] for ss in FIXED_SAMPLE_SIZES)
    print(f"{method:<15}", end="")
    for ss in FIXED_SAMPLE_SIZES:
        print(f"{win_counts[ss][method]:>8}", end="")
    print(f"   {total_wins:>5}")

# MEDIAN recall table
print("\n" + "=" * 60)
print("  MEDIAN Pareto Recall by Method & Sample Size")
print("=" * 60)
pivot = df.pivot_table(values='recall', index='method', columns='sample_size', aggfunc='median').round(3)
print(pivot.to_string())

print("\n" + "=" * 60)
print("  BEST METHOD PER SAMPLE SIZE (by Median Recall)")
print("=" * 60)
for ss in FIXED_SAMPLE_SIZES:
    ss_data = df[df['sample_size'] == ss]
    best = ss_data.groupby('method')['recall'].median().idxmax()
    best_val = ss_data.groupby('method')['recall'].median().max()
    print(f"  Sample size {ss:3d}: {best:15s} (median={best_val:.3f})")

print("\nDone!")
