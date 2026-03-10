"""
Run DROS + SVM experiment on ALL available datasets.
Prints a summary table comparable to Table 2 of the paper.
"""
import sys
import time
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src import load_dataset, list_datasets, run_experiment
from src.data_loader import DATASET_REGISTRY

# Paper's reported DROS g-mean values (Table 2, SVM classifier)
PAPER_GMEAN = {
    'haberman':       0.5224,
    'wpbc':           0.5750,
    'diabetes':       0.7046,
    'hepatitis':      0.8099,
    'housing':        0.8660,
    'spectf':         0.7792,
    'iris':           1.0000,
    'abalone_5_6':    0.6457,
    'abalone_4_11':   0.7975,
    'ecoli_4_2':      0.9180,
    'ecoli_5_1':      0.9757,
    'glass_7_2':      0.9404,
    'glass_5_1':      0.7659,
    'pageblocks_3_1': 0.9433,
    'pageblocks_5_2': 0.8889,
    'yeast_5_3':      0.9512,
    'yeast_9_4':      0.9685,
}

all_results = []

for name in DATASET_REGISTRY:
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        X, y, info = load_dataset(name)
        print(f"  Loaded: {info['n_samples']} samples, {info['n_features']} features")
        print(f"  Minority: {info['n_minority']}, Majority: {info['n_majority']}, IR: {info['ir']}")

        results, summary = run_experiment(X, y, verbose=True)
        elapsed = time.time() - start

        gmean_mean = summary['g_mean']['mean']
        gmean_std = summary['g_mean']['std']
        paper_val = PAPER_GMEAN.get(name, None)
        diff = gmean_mean - paper_val if paper_val else None

        row = {
            'Dataset': info['description'],
            'Key': name,
            'Min:Maj': f"{info['n_minority']}:{info['n_majority']}",
            'IR': info['ir'],
            'Precision': f"{summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}",
            'Recall': f"{summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}",
            'F-measure': f"{summary['f_measure']['mean']:.4f} ± {summary['f_measure']['std']:.4f}",
            'G-mean': f"{gmean_mean:.4f} ± {gmean_std:.4f}",
            'AUC': f"{summary['auc']['mean']:.4f} ± {summary['auc']['std']:.4f}",
            'Paper G-mean': f"{paper_val:.4f}" if paper_val else "N/A",
            'Diff': f"{diff:+.4f}" if diff is not None else "N/A",
            'Time(s)': f"{elapsed:.1f}",
            'Status': 'OK',
        }
        all_results.append(row)

        print(f"\n  Results (50 folds):")
        for metric in ['precision', 'recall', 'f_measure', 'g_mean', 'auc']:
            m, s = summary[metric]['mean'], summary[metric]['std']
            print(f"    {metric:12s}: {m:.4f} ± {s:.4f}")
        if paper_val:
            print(f"    Paper g-mean: {paper_val:.4f}  |  Diff: {diff:+.4f}")
        print(f"  Time: {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        traceback.print_exc()
        all_results.append({
            'Dataset': DATASET_REGISTRY[name]['description'],
            'Key': name,
            'Min:Maj': '?',
            'IR': '?',
            'Precision': '-', 'Recall': '-', 'F-measure': '-',
            'G-mean': '-', 'AUC': '-',
            'Paper G-mean': f"{PAPER_GMEAN.get(name, 'N/A')}",
            'Diff': '-',
            'Time(s)': f"{elapsed:.1f}",
            'Status': f'FAILED: {e}',
        })

# Print final summary table
print(f"\n\n{'='*80}")
print("FINAL SUMMARY — DROS + SVM (Gaussian kernel)")
print(f"{'='*80}")

df = pd.DataFrame(all_results)
print(df[['Dataset', 'Min:Maj', 'IR', 'G-mean', 'Paper G-mean', 'Diff', 'Time(s)', 'Status']].to_string(index=False))

# Count successes
ok = df[df['Status'] == 'OK']
print(f"\nSuccessful: {len(ok)}/{len(df)} datasets")
