import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

print("=" * 80)
print("V7 EXPERIMENTS: RECENCY WEIGHTING, ALTERNATIVE MODELS, & HP TUNING")
print("=" * 80)

# Load data
train_df = pd.read_parquet('v6/train_features_v6.parquet')

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v6_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
c_idx = [v6_features.index(c) for c in cat_names if c in v6_features]

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

# Baseline CatBoost params
base_hp = {
    'depth': 6,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0
}

# ===========================================================================
# EXPERIMENT A: RECENCY-WEIGHTED CATBOOST
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT A: RECENCY-WEIGHTED CATBOOST")
print("=" * 60)

date_to_month = {
    '2026-01-31': 1,
    '2026-02-28': 2,
    '2026-03-31': 3,
    '2026-04-30': 4
}
months_series = train_df['snapshot_date'].astype(str).map(date_to_month)

weight_schemes = {
    "Uniform (Baseline)": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    "Range 0.8 -> 1.2":   {1: 0.80, 2: 0.9333, 3: 1.0667, 4: 1.20},
    "Range 0.7 -> 1.3":   {1: 0.70, 2: 0.90, 3: 1.10, 4: 1.30},
    "Exponential (Half-life 60d)": {1: 0.75, 2: 0.85, 3: 1.05, 4: 1.35}
}

exp_a_results = {}

for name, w_map in weight_schemes.items():
    weights_col = months_series.map(w_map).values
    fold_aucs = []
    
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        tr_w = weights_col[tr_mask]
        
        m = CatBoostClassifier(
            iterations=1500, random_seed=42,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **base_hp
        )
        m.fit(
            tr_sub[v6_features], tr_sub['churn'],
            sample_weight=tr_w,
            eval_set=(vl_sub[v6_features], vl_sub['churn'])
        )
        preds = m.predict_proba(vl_sub[v6_features])[:, 1]
        auc = roc_auc_score(vl_sub['churn'], preds)
        fold_aucs.append(auc)
        
    mean_auc = np.mean(fold_aucs)
    delta = mean_auc - v6_res['v6_cb_5seed_mean_cv'] if '5seed' in name else mean_auc - 0.86465
    exp_a_results[name] = (mean_auc, fold_aucs, delta)
    print(f"  {name:30s} -> Mean: {mean_auc:.5f} (Δ: {delta:+.5f}) | F1: {fold_aucs[0]:.5f}, F2: {fold_aucs[1]:.5f}, F3: {fold_aucs[2]:.5f}")

# ===========================================================================
# EXPERIMENT B: DIFFERENT MODEL FAMILY
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT B: DIFFERENT MODEL FAMILIES (HISTGRADIENTBOOSTING & LOGISTIC)")
print("=" * 60)

# 1. HistGradientBoostingClassifier
print("\n1. Training HistGradientBoostingClassifier...")
# Encode categoricals as integer codes for HGB
hgb_df = train_df[v6_features].copy()
cat_col_indices = []
for idx, c in enumerate(v6_features):
    if c in cat_names:
        hgb_df[c] = hgb_df[c].astype('category').cat.codes
        cat_col_indices.append(idx)

hgb_params = {
    'max_iter': 1000,
    'learning_rate': 0.04,
    'max_leaf_nodes': 31,
    'min_samples_leaf': 30,
    'l2_regularization': 1.0,
    'random_state': 42,
    'early_stopping': True,
    'n_iter_no_change': 50
}

hgb_folds = []
hgb_preds_all = []
cb_preds_all = []
y_true_all = []

# Get CB baseline preds for correlation and blending
for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    y_vl = vl_sub['churn']
    
    # CB baseline
    m_cb = CatBoostClassifier(
        iterations=1500, random_seed=42,
        cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
        **base_hp
    )
    m_cb.fit(tr_sub[v6_features], tr_sub['churn'], eval_set=(vl_sub[v6_features], y_vl))
    cb_p = m_cb.predict_proba(vl_sub[v6_features])[:, 1]
    
    # HGB
    m_hgb = HistGradientBoostingClassifier(
        categorical_features=cat_col_indices,
        **hgb_params
    )
    m_hgb.fit(hgb_df[tr_mask], tr_sub['churn'])
    hgb_p = m_hgb.predict_proba(hgb_df[vl_mask])[:, 1]
    
    hgb_auc = roc_auc_score(y_vl, hgb_p)
    hgb_folds.append(hgb_auc)
    hgb_preds_all.append(hgb_p)
    cb_preds_all.append(cb_p)
    y_true_all.append(y_vl.values)
    
    corr = np.corrcoef(cb_p, hgb_p)[0, 1]
    print(f"  {fold_name}: HGB AUC = {hgb_auc:.5f} | CB AUC = {roc_auc_score(y_vl, cb_p):.5f} | Prediction Corr = {corr:.4f}")

mean_hgb_auc = np.mean(hgb_folds)
print(f"  HistGradientBoosting Mean CV: {mean_hgb_auc:.5f}")

# 2. Regularized Logistic Regression
print("\n2. Training Regularized Logistic Regression Pipeline...")
num_cols = [c for c in v6_features if c not in cat_names]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
    ]
)

log_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])

lr_folds = []
lr_preds_all = []
for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    y_vl = vl_sub['churn']
    
    log_pipeline.fit(tr_sub[v6_features], tr_sub['churn'])
    lr_p = log_pipeline.predict_proba(vl_sub[v6_features])[:, 1]
    lr_auc = roc_auc_score(y_vl, lr_p)
    lr_folds.append(lr_auc)
    lr_preds_all.append(lr_p)
    
    corr_lr = np.corrcoef(cb_preds_all[fold_idx], lr_p)[0, 1]
    print(f"  {fold_name}: Logistic AUC = {lr_auc:.5f} | Prediction Corr with CB = {corr_lr:.4f}")

mean_lr_auc = np.mean(lr_folds)
print(f"  Logistic Regression Mean CV: {mean_lr_auc:.5f}")

# 3. Blending Tests
print("\n3. Testing Blends with CatBoost (90/10, 80/20, 70/30)...")
blend_weights = [0.10, 0.20, 0.30]

print("\n--- CatBoost + HistGradientBoosting Blends ---")
for w_alt in blend_weights:
    w_cb = 1.0 - w_alt
    b_folds = []
    for f in range(3):
        bp = w_cb * cb_preds_all[f] + w_alt * hgb_preds_all[f]
        b_folds.append(roc_auc_score(y_true_all[f], bp))
    print(f"  Blend {int(w_cb*100)}/{int(w_alt*100)} (CB/HGB) -> Mean: {np.mean(b_folds):.5f} | F1: {b_folds[0]:.5f}, F2: {b_folds[1]:.5f}, F3: {b_folds[2]:.5f} (Δ: {np.mean(b_folds) - 0.86465:+.5f})")

print("\n--- CatBoost + Logistic Regression Blends ---")
for w_alt in [0.03, 0.05, 0.10, 0.20]:
    w_cb = 1.0 - w_alt
    b_folds = []
    for f in range(3):
        bp = w_cb * cb_preds_all[f] + w_alt * lr_preds_all[f]
        b_folds.append(roc_auc_score(y_true_all[f], bp))
    print(f"  Blend {int(w_cb*100)}/{int(w_alt*100)} (CB/LR)  -> Mean: {np.mean(b_folds):.5f} | F1: {b_folds[0]:.5f}, F2: {b_folds[1]:.5f}, F3: {b_folds[2]:.5f} (Δ: {np.mean(b_folds) - 0.86465:+.5f})")

# ===========================================================================
# EXPERIMENT C: CATBOOST DEPTH & LEARNING RATE TUNING
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT C: CATBOOST DEPTH & LEARNING RATE SWEEP")
print("=" * 60)

hp_candidates = [
    {"depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 1},  # current baseline
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 1},
    {"depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 1},
    {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 1},
    {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 1},
    {"depth": 7, "learning_rate": 0.02, "l2_leaf_reg": 1},
    {"depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 1},
]

exp_c_results = []
for idx, hp in enumerate(hp_candidates):
    full_cfg = base_hp.copy()
    full_cfg.update(hp)
    
    f_res = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        
        m = CatBoostClassifier(
            iterations=1500, random_seed=42,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **full_cfg
        )
        m.fit(tr_sub[v6_features], tr_sub['churn'], eval_set=(vl_sub[v6_features], vl_sub['churn']))
        preds = m.predict_proba(vl_sub[v6_features])[:, 1]
        f_res.append(roc_auc_score(vl_sub['churn'], preds))
        
    m_res = np.mean(f_res)
    delta = m_res - 0.86465
    exp_c_results.append((hp, m_res, delta, f_res))
    print(f"  Config {idx:d}: depth={hp['depth']}, lr={hp['learning_rate']} -> Mean: {m_res:.5f} (Δ: {delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}")

# Save all results to v7/v7_results.json
import os
os.makedirs('v7', exist_ok=True)
with open('v7/v7_results.json', 'w') as f:
    json.dump({
        "v6_baseline_single_cv": 0.86465,
        "v6_baseline_5seed_cv": v6_res['v6_cb_5seed_mean_cv'],
        "exp_a_recency_weights": {k: {"mean": v[0], "delta": v[2], "folds": v[1]} for k, v in exp_a_results.items()},
        "exp_b_models": {
            "hgb": {"mean": mean_hgb_auc, "folds": hgb_folds},
            "logistic": {"mean": mean_lr_auc, "folds": lr_folds}
        },
        "exp_c_hpo": [{"hp": r[0], "mean": r[1], "delta": r[2], "folds": r[3]} for r in exp_c_results]
    }, f, indent=2)

print("\nAll V7 experimental results saved to v7/v7_results.json")
