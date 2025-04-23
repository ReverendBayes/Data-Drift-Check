#!/usr/bin/env python3
"""
driftcheck.py: Tabular Data Drift Detector & Reporter

Usage:
    python driftcheck.py --baseline base.csv --current new.csv --report drift_report.md

Features:
    - Auto-detects numeric vs. categorical columns
    - Computes KS-test (numeric) and Jensen-Shannon divergence (categorical)
    - Flags features with JS divergence > 0.1 by default
    - Generates Matplotlib histograms and bar charts, saved under ./charts/
    - Outputs a Markdown report with embedded charts
Dependencies:
    pandas, numpy, scipy, matplotlib
"""
import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_numeric_drift(baseline, current, col):
    a = baseline[col].dropna().astype(float)
    b = current[col].dropna().astype(float)
    if a.empty or b.empty:
        return None, None
    stat = ks_2samp(a, b).statistic
    return stat, (a, b)

def compute_categorical_drift(baseline, current, col):
    a_counts = baseline[col].fillna('nan').value_counts(normalize=True)
    b_counts = current[col].fillna('nan').value_counts(normalize=True)
    cats = sorted(set(a_counts.index) | set(b_counts.index))
    p = np.array([a_counts.get(c, 0) for c in cats])
    q = np.array([b_counts.get(c, 0) for c in cats])
    # jensenshannon returns sqrt(JS divergence)
    js = jensenshannon(p, q)
    return js, (cats, p, q)

def plot_numeric(a, b, col, charts_dir):
    plt.figure()
    plt.hist(a, bins=20, alpha=0.5, label='baseline')
    plt.hist(b, bins=20, alpha=0.5, label='current')
    plt.title(f'{col} distribution')
    plt.legend()
    path = os.path.join(charts_dir, f'{col}_hist.png')
    plt.savefig(path)
    plt.close()
    return path

def plot_categorical(cats, p, q, col, charts_dir):
    x = np.arange(len(cats))
    width = 0.35
    plt.figure(figsize=(max(6, len(cats)*0.5), 4))
    plt.bar(x - width/2, p, width, label='baseline')
    plt.bar(x + width/2, q, width, label='current')
    plt.xticks(x, cats, rotation=45, ha='right')
    plt.title(f'{col} category distribution')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(charts_dir, f'{col}_bar.png')
    plt.savefig(path)
    plt.close()
    return path

def main():
    parser = argparse.ArgumentParser(description='Tabular Data Drift Detector & Reporter')
    parser.add_argument('--baseline', required=True, help='Baseline CSV file path')
    parser.add_argument('--current', required=True, help='Current CSV file path')
    parser.add_argument('--report', required=True, help='Markdown report output path')
    parser.add_argument('--threshold', type=float, default=0.1, help='JS divergence flag threshold')
    args = parser.parse_args()

    # Load data
    baseline = pd.read_csv(args.baseline)
    current = pd.read_csv(args.current)

    # Prepare output
    report_path = args.report
    charts_dir = os.path.join(os.path.dirname(report_path) or '.', 'charts')
    ensure_dir(charts_dir)

    # Identify common columns
    cols = sorted(set(baseline.columns) & set(current.columns))

    numeric_results = []
    categorical_results = []

    # Compute drift
    for col in cols:
        if pd.api.types.is_numeric_dtype(baseline[col]):
            stat, data = compute_numeric_drift(baseline, current, col)
            if stat is not None:
                numeric_results.append((col, stat, data))
        else:
            js, data = compute_categorical_drift(baseline, current, col)
            categorical_results.append((col, js, data))

    # Write report
    with open(report_path, 'w') as f:
        f.write(f'# Drift Report

Generated on {datetime.now().isoformat()}

')

        # Numeric drift
        f.write('## Numeric Features Drift (KS Statistic)

')
        for col, stat, data in numeric_results:
            f.write(f'- **{col}**: KS statistic = {stat:.4f}
')
        f.write('
### Histograms

')
        for col, stat, (a, b) in numeric_results:
            img = plot_numeric(a, b, col, charts_dir)
            f.write(f'![{col} distribution]({os.path.relpath(img)})

')

        # Categorical drift
        f.write('## Categorical Features Drift (Jensen-Shannon)

')
        for col, js, data in categorical_results:
            flag = ' ⚠️' if js > args.threshold else ''
            f.write(f'- **{col}**: JS divergence = {js:.4f}{flag}
')
        f.write('
### Category Distribution

')
        for col, js, (cats, p, q) in categorical_results:
            img = plot_categorical(cats, p, q, col, charts_dir)
            f.write(f'![{col} categories]({os.path.relpath(img)})

')

    print(f'Drift report generated: {report_path}')
    print(f'Charts saved under: {charts_dir}')

if __name__ == '__main__':
    main()
