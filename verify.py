"""
verify.py
=========
Cross-checks this repository's actual code and data against the exact
numbers reported in the manuscript. Run this any time you want proof
the repo is still reproducing correctly - no need to read raw training
logs yourself, this prints a clear PASS/FAIL for each claim.

Run from the repository root:
    python verify.py
"""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f"  -> {detail}" if detail else ""))


def check_file_exists(label, path):
    exists = os.path.exists(os.path.join(BASE_DIR, path))
    check(f"File exists: {path}", exists, "" if exists else "NOT FOUND")


def close(a, b, tol):
    return abs(a - b) <= tol


print("=" * 70)
print("PART 1: Required files present")
print("=" * 70)
for path in [
    "data/shared/features.csv",
    "data/shared/skp2_features.csv",
    "data/shared/other_ligase_substrates.csv",
    "data/shared/fbox_family_proteins.csv",
    "data/btrc/pathway_genes.csv",
    "data/btrc/pathway_genes_features.csv",
    "models/btrc/lr_model.pkl",
    "models/btrc/xgb_model.pkl",
    "models/btrc/scaler.pkl",
    "models/btrc/feature_columns.txt",
    "models/btrc/optimal_threshold.txt",
    "prediction/genome_wide_predictions.csv",
]:
    check_file_exists(path, path)

print()
print("=" * 70)
print("PART 2: BTRC final model - reproduces the manuscript exactly")
print("=" * 70)

import train_btrc_model as btrc

df = btrc.load_training_data()
n_pos = int((df["label"] == 1).sum())
n_neg = int((df["label"] == 0).sum())
check("BTRC positives == 68", n_pos == 68, f"got {n_pos}")
check("BTRC negatives == 131", n_neg == 131, f"got {n_neg}")

oof_proba, fold_aucs = btrc.cross_validate(df)
import numpy as np
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)
check("Mean ROC-AUC ~= 0.755 (+/- 0.02 tolerance)", close(mean_auc, 0.755, 0.02), f"got {mean_auc:.3f}")
check("Std ROC-AUC ~= 0.117 (+/- 0.02 tolerance)", close(std_auc, 0.117, 0.02), f"got {std_auc:.3f}")

threshold, f1 = btrc.optimize_threshold(df["label"].values, oof_proba)
check("Optimal threshold == 0.28", threshold == 0.28, f"got {threshold}")
check("F1 at optimal threshold ~= 0.620 (+/- 0.02 tolerance)", close(f1, 0.620, 0.02), f"got {f1:.3f}")

with open(os.path.join(BASE_DIR, "models/btrc/feature_columns.txt")) as f:
    saved_features = f.read().splitlines()
check(
    "Saved feature_columns.txt matches the 4 expected features",
    saved_features == btrc.FEATURE_COLS,
    f"got {saved_features}",
)

with open(os.path.join(BASE_DIR, "models/btrc/optimal_threshold.txt")) as f:
    saved_threshold = float(f.read().strip())
check("Saved optimal_threshold.txt == 0.28", saved_threshold == 0.28, f"got {saved_threshold}")

print()
print("=" * 70)
print("PART 3: SKP2 pilot - matches the real original scripts' sample size")
print("=" * 70)

import train_skp2_pilot as skp2

skp2_df = skp2.load_training_data()
skp2_pos = int((skp2_df["label"] == 1).sum())
skp2_neg = int((skp2_df["label"] == 0).sum())
check("SKP2 positives == 22", skp2_pos == 22, f"got {skp2_pos}")
check("SKP2 negatives == 284", skp2_neg == 284, f"got {skp2_neg}")
check(
    "SKP2 feature set matches real scripts (has_confirmed_phospho_cdk, confirmed_phospho_cdk_count)",
    skp2.FEATURE_COLS == ["has_confirmed_phospho_cdk", "confirmed_phospho_cdk_count"],
    f"got {skp2.FEATURE_COLS}",
)
check("SKP2 pilot uses 5-fold CV (matches real scripts)", skp2.N_FOLDS == 5, f"got {skp2.N_FOLDS}")

print()
print("=" * 70)
print("PART 4: Genome-wide screening data integrity")
print("=" * 70)

import pandas as pd

genome_preds = pd.read_csv(os.path.join(BASE_DIR, "prediction/genome_wide_predictions.csv"))
check("Genome-wide screen covers 859 proteins", len(genome_preds) == 859, f"got {len(genome_preds)}")

pathway_genes = pd.read_csv(os.path.join(BASE_DIR, "data/btrc/pathway_genes.csv"))
check("MSigDB gene list has 874 genes", len(pathway_genes) == 874, f"got {len(pathway_genes)}")

print()
print("=" * 70)
n_pass = sum(1 for s, _, _ in results if s == PASS)
n_total = len(results)
print(f"SUMMARY: {n_pass}/{n_total} checks passed")
if n_pass < n_total:
    print("\nFailed checks:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"  - {label}: {detail}")
print("=" * 70)
