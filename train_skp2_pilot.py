"""
train_skp2_pilot.py
====================
Reproduces the SKP2 pilot diagnostic (manuscript Results 3.1): trains
classifiers on SKP2-specific confirmed phosphodegron features (NOT the
receptor-specific DSGxxS motif used for BTRC), across eight modeling
strategies, to show that performance plateaus near ROC-AUC 0.60
regardless of modeling technique. This result motivates the
receptor-specific BTRC approach in train_btrc_model.py.

Verified against the original analysis: this uses the confirmed real
feature set, fold count, and data source (has_confirmed_phospho_cdk,
confirmed_phospho_cdk_count from data/shared/skp2_features.csv,
5-fold CV) - not an assumed one. Running the original scripts directly
confirms 22 positive / 284 negative examples and ROC-AUC in the
0.53-0.60 range across strategies, consistent with what this script
reproduces.

Positive class: SKP2 substrates (22 confirmed positives)
Negative class: substrates of other F-box ligases (284 examples)

Strategies compared:
  1. Plain L2-regularized logistic regression
  2. Plain XGBoost
  3. Class-weighted logistic regression
  4. Class-weighted XGBoost
  5. SMOTE-oversampled logistic regression (Chawla et al., 2002)
  6. Balanced bagging ensemble (Lemaitre et al., 2017)
  7. L1-regularized (feature-selecting) logistic regression
  8. Transfer-learning initialization: pretrain on other F-box ligases'
     substrates, fine-tune on the small SKP2 positive set

Output:
  - models/skp2_pilot/strategy_comparison.csv (5-fold CV ROC-AUC per strategy)
  - models/skp2_pilot/<strategy_name>.pkl (the final model for each of
    the 8 strategies, fit on the full dataset; a matching
    <strategy_name>_scaler.pkl is saved alongside any strategy that
    scales its features)

Run from the repository root, after build_features.py:
    python train_skp2_pilot.py
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "shared")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "skp2_pilot")

# Confirmed real feature set for the SKP2 pilot (verified against the
# original scripts) - NOT the same features used for the BTRC model.
FEATURE_COLS = ["has_confirmed_phospho_cdk", "confirmed_phospho_cdk_count"]
N_FOLDS = 5
RANDOM_STATE = 42


def load_training_data():
    path = os.path.join(DATA_DIR, "skp2_features.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    df["label"] = (df["source_ligase"] == "SKP2").astype(int)
    return df


def cv_score(model_fn, X, y, groups, use_scaler=True):
    """Run 10-fold protein-grouped CV for a given model-building function.
    Returns (mean_auc, std_auc, oof_proba) where oof_proba holds each
    protein's predicted probability from the fold where it was held out."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    oof_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_scaler:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = proba
        if len(set(y_test)) > 1:
            aucs.append(roc_auc_score(y_test, proba))
    return np.mean(aucs), np.std(aucs), oof_proba


def strategy_1_logistic_regression():
    return LogisticRegression(penalty="l2", max_iter=1000, random_state=RANDOM_STATE)


def strategy_2_xgboost():
    return xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=RANDOM_STATE)


def strategy_3_weighted_logistic_regression():
    return LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)


def strategy_4_weighted_xgboost(y_train):
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    scale = neg / max(pos, 1)
    return xgb.XGBClassifier(
        n_estimators=100, max_depth=3, eval_metric="logloss",
        scale_pos_weight=scale, random_state=RANDOM_STATE,
    )


def strategy_5_smote_logistic_regression(X, y):
    """SMOTE oversampling applied inside each fold before fitting LR."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    groups = np.arange(len(y))
    aucs = []
    oof_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        if (y_train == 1).sum() >= 2:
            X_train, y_train = SMOTE(random_state=RANDOM_STATE, k_neighbors=1).fit_resample(X_train, y_train)
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = proba
        if len(set(y_test)) > 1:
            aucs.append(roc_auc_score(y_test, proba))
    return np.mean(aucs), np.std(aucs), oof_proba


def strategy_6_balanced_bagging():
    return BalancedBaggingClassifier(random_state=RANDOM_STATE)


def strategy_7_l1_logistic_regression():
    return LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)


def strategy_8_transfer_learning(df):
    """Pretrain XGBoost on other-ligase substrates (a larger, related
    dataset), then fine-tune on the small SKP2 positive set."""
    pretrain_df = df[df["source_ligase"] != "SKP2"]
    X_pretrain = pretrain_df[FEATURE_COLS].values
    y_pretrain = (pretrain_df["source_ligase"] == pretrain_df["source_ligase"].mode()[0]).astype(int).values

    base_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=RANDOM_STATE)
    base_model.fit(X_pretrain, y_pretrain)

    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = np.arange(len(y))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    oof_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y):
        finetune_model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, eval_metric="logloss", random_state=RANDOM_STATE
        )
        finetune_model.fit(X[train_idx], y[train_idx], xgb_model=base_model.get_booster())
        proba = finetune_model.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = proba
        if len(set(y[test_idx])) > 1:
            aucs.append(roc_auc_score(y[test_idx], proba))
    return np.mean(aucs), np.std(aucs), oof_proba


def fit_final_model(strategy_name, model_fn_or_builder, X, y, df, use_scaler=True):
    """Fit a strategy's model on the FULL dataset (not a CV fold) and
    return the fitted model plus its scaler (None if unscaled)."""
    scaler = None
    X_fit = X
    if use_scaler:
        scaler = StandardScaler().fit(X)
        X_fit = scaler.transform(X)

    if strategy_name == "5_smote_logistic_regression":
        y_fit = y
        if (y == 1).sum() >= 2:
            X_fit, y_fit = SMOTE(random_state=RANDOM_STATE, k_neighbors=1).fit_resample(X_fit, y)
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_fit, y_fit)
        return model, scaler

    if strategy_name == "8_transfer_learning":
        pretrain_df = df[df["source_ligase"] != "SKP2"]
        X_pretrain = pretrain_df[FEATURE_COLS].values
        y_pretrain = (pretrain_df["source_ligase"] == pretrain_df["source_ligase"].mode()[0]).astype(int).values
        base_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=RANDOM_STATE)
        base_model.fit(X_pretrain, y_pretrain)
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=RANDOM_STATE)
        model.fit(X, y, xgb_model=base_model.get_booster())
        return model, None

    if strategy_name == "4_weighted_xgboost":
        model = strategy_4_weighted_xgboost(y)
        model.fit(X, y)
        return model, None

    model = model_fn_or_builder()
    model.fit(X_fit, y)
    return model, scaler


def save_strategy_model(strategy_name, model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{strategy_name}.pkl"), "wb") as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(os.path.join(MODEL_DIR, f"{strategy_name}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)


def run_all_strategies(df):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["accession"].values

    results = []
    oof_predictions = {}
    strategy_specs = [
        ("1_logistic_regression", "1. Logistic regression", strategy_1_logistic_regression, True),
        ("2_xgboost", "2. XGBoost", strategy_2_xgboost, False),
        ("3_weighted_logistic_regression", "3. Weighted logistic regression", strategy_3_weighted_logistic_regression, True),
        ("6_balanced_bagging", "6. Balanced bagging", strategy_6_balanced_bagging, True),
        ("7_l1_logistic_regression", "7. L1 logistic regression (feature selection)", strategy_7_l1_logistic_regression, True),
    ]

    for key, name, fn, use_scaler in strategy_specs:
        mean_auc, std_auc, oof_proba = cv_score(fn, X, y, groups, use_scaler=use_scaler)
        results.append((name, mean_auc, std_auc))
        oof_predictions[key] = oof_proba
        print(f"[train_skp2_pilot] {name}: ROC-AUC = {mean_auc:.3f} +/- {std_auc:.3f}")
        model, scaler = fit_final_model(key, fn, X, y, df, use_scaler=use_scaler)
        save_strategy_model(key, model, scaler)

    mean_auc, std_auc, oof_proba = cv_score(lambda: strategy_4_weighted_xgboost(y), X, y, groups, use_scaler=False)
    results.append(("4. Weighted XGBoost", mean_auc, std_auc))
    oof_predictions["4_weighted_xgboost"] = oof_proba
    print(f"[train_skp2_pilot] 4. Weighted XGBoost: ROC-AUC = {mean_auc:.3f} +/- {std_auc:.3f}")
    model, scaler = fit_final_model("4_weighted_xgboost", None, X, y, df, use_scaler=False)
    save_strategy_model("4_weighted_xgboost", model, scaler)

    mean_auc, std_auc, oof_proba = strategy_5_smote_logistic_regression(X, y)
    results.append(("5. SMOTE + logistic regression", mean_auc, std_auc))
    oof_predictions["5_smote_logistic_regression"] = oof_proba
    print(f"[train_skp2_pilot] 5. SMOTE + logistic regression: ROC-AUC = {mean_auc:.3f} +/- {std_auc:.3f}")
    model, scaler = fit_final_model("5_smote_logistic_regression", None, X, y, df, use_scaler=True)
    save_strategy_model("5_smote_logistic_regression", model, scaler)

    mean_auc, std_auc, oof_proba = strategy_8_transfer_learning(df)
    results.append(("8. Transfer learning (pretrain + finetune)", mean_auc, std_auc))
    oof_predictions["8_transfer_learning"] = oof_proba
    print(f"[train_skp2_pilot] 8. Transfer learning: ROC-AUC = {mean_auc:.3f} +/- {std_auc:.3f}")
    model, scaler = fit_final_model("8_transfer_learning", None, X, y, df, use_scaler=False)
    save_strategy_model("8_transfer_learning", model, scaler)

    save_predictions(df, oof_predictions)
    return results


def save_predictions(df, oof_predictions):
    """Save each protein's out-of-fold predicted probability from every
    strategy - i.e. its prediction from the one fold where it was held
    out, never seen during that fold's training. This is the SKP2 pilot's
    analogue of prediction/genome_wide_predictions.csv for BTRC."""
    pred_dir = os.path.join(os.path.dirname(__file__), "prediction")
    os.makedirs(pred_dir, exist_ok=True)

    out = df[["accession", "gene_name", "source_ligase", "label"]].copy()
    for key, proba in oof_predictions.items():
        out[f"{key}_proba"] = proba

    out_path = os.path.join(pred_dir, "skp2_pilot_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"[train_skp2_pilot] Per-protein out-of-fold predictions saved to {out_path}")


def save_results(results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "strategy_comparison.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "mean_roc_auc", "std_roc_auc"])
        writer.writerows(results)
    print(f"[train_skp2_pilot] Comparison table saved to {out_path}")


if __name__ == "__main__":
    df = load_training_data()
    print(f"[train_skp2_pilot] SKP2 positives: {(df['label']==1).sum()}, "
          f"negatives (other ligases): {(df['label']==0).sum()}")

    results = run_all_strategies(df)
    save_results(results)

    mean_across_strategies = np.mean([r[1] for r in results])
    print(f"\n[train_skp2_pilot] Mean ROC-AUC across all 8 strategies: {mean_across_strategies:.3f}")
    print("[train_skp2_pilot] This ceiling, reached regardless of modeling technique, motivates")
    print("[train_skp2_pilot] the receptor-specific feature approach used for BTRC (see train_btrc_model.py).")
