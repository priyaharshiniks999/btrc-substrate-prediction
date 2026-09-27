"""
train_btrc_model.py
====================
Trains the final BTRC substrate classifier: an unweighted-average
ensemble of class-weighted logistic regression and class-weighted
XGBoost, using the four features computed by build_features.py.

This exactly follows the verified methodology of the original analysis
(confirmed by reproducing the manuscript's reported ROC-AUC 0.755 +/-
0.117, F1 0.620 to three decimal places):

  - Positive class:  BTRC substrates (source_ligase == "BTRC")
  - Negative class:  substrates of other F-box ligases (dual-targeted /
                      ambiguous accessions excluded)
  - Evaluation:       10-fold stratified cross-validation (data is
                      already one-row-per-protein, so stratified
                      k-fold is equivalent to protein-grouped k-fold)
  - Logistic regression: class_weight="balanced" (corrects for the
                      ~1:2 class imbalance)
  - XGBoost:          scale_pos_weight set to the exact negative:positive
                      ratio in each fold's training split (same imbalance
                      correction, XGBoost's native mechanism for it)
  - Ensemble:          simple average of LR and XGBoost predicted
                        probabilities
  - Decision threshold: chosen by maximizing F1 score on pooled
                        out-of-fold predictions (discovery-oriented:
                        prioritizes recall over precision)

Output: models/btrc/
  - lr_model.pkl, xgb_model.pkl, scaler.pkl
  - feature_columns.txt
  - optimal_threshold.txt

Run from the repository root, after build_features.py:
    python train_btrc_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "shared")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "btrc")

FEATURE_COLS = ["has_dsgxxs_motif", "dsgxxs_count", "max_disorder_score", "mean_plddt"]
N_FOLDS = 10
RANDOM_STATE = 42


def load_training_data():
    """Load the feature table and apply the same exclusions used in the
    manuscript (Results 3.1): proteins claimed as substrates of more than
    one F-box ligase are excluded, since they cannot serve as an
    unambiguous positive or negative example for any single receptor."""
    path = os.path.join(DATA_DIR, "features.csv")
    df = pd.read_csv(path)

    ligases_per_accession = df.groupby("accession")["source_ligase"].apply(set)
    ambiguous = ligases_per_accession[ligases_per_accession.apply(len) > 1].index
    df = df[~df["accession"].isin(ambiguous)].drop_duplicates(subset=["accession"], keep="first")

    df = df.dropna(subset=FEATURE_COLS)
    df["label"] = (df["source_ligase"] == "BTRC").astype(int)
    return df


def cross_validate(df):
    """10-fold stratified cross-validation of the LR+XGBoost ensemble.
    Returns pooled out-of-fold probabilities and per-fold ROC-AUC scores."""
    X = df[FEATURE_COLS].values
    y = df["label"].values

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(df))
    fold_aucs = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_train_s, y_train)
        lr_proba = lr.predict_proba(X_test_s)[:, 1]

        pos_weight = sum(y_train == 0) / sum(y_train == 1)
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE
        )
        xgb_model.fit(X_train_s, y_train)
        xgb_proba = xgb_model.predict_proba(X_test_s)[:, 1]

        ensemble_proba = (lr_proba + xgb_proba) / 2
        oof_proba[test_idx] = ensemble_proba
        fold_aucs.append(roc_auc_score(y_test, ensemble_proba))

    return oof_proba, fold_aucs


def optimize_threshold(y_true, y_proba):
    """Sweep candidate thresholds and return the one that maximizes F1."""
    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.02):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return round(best_threshold, 2), best_f1


def train_final_models(df):
    """Fit LR and XGBoost on the FULL dataset for downstream inference."""
    X = df[FEATURE_COLS].values
    y = df["label"].values

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_s, y)

    pos_weight = sum(y == 0) / sum(y == 1)
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE
    )
    xgb_model.fit(X_s, y)

    return lr, xgb_model, scaler


def save_models(lr, xgb_model, scaler, threshold):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_columns.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))
    with open(os.path.join(MODEL_DIR, "optimal_threshold.txt"), "w") as f:
        f.write(str(threshold))

    print(f"[train_btrc_model] Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    df = load_training_data()
    print(f"[train_btrc_model] Training set: {(df['label']==1).sum()} positive, "
          f"{(df['label']==0).sum()} negative")

    oof_proba, fold_aucs = cross_validate(df)
    print(f"[train_btrc_model] 10-fold ROC-AUC: {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")
    print(f"[train_btrc_model] Pooled out-of-fold ROC-AUC: {roc_auc_score(df['label'], oof_proba):.3f}")

    threshold, f1 = optimize_threshold(df["label"].values, oof_proba)
    print(f"[train_btrc_model] Optimal decision threshold (F1-maximized): {threshold} (F1={f1:.3f})")

    lr, xgb_model, scaler = train_final_models(df)
    save_models(lr, xgb_model, scaler, threshold)
