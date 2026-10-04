import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import json
import time
import os

print("--- Step 1: Loading candidate datasets ---")
train_df = pd.read_parquet('scratch_train_v4_candidates.parquet')
test_df = pd.read_parquet('scratch_test_v4_candidates.parquet')

with open('v3/best_config.json') as f:
    v3_config = json.load(f)

v3_features = v3_config['features']
best_hp = v3_config['best_hp']
cat_names = v3_config['cat_features']

val_date = train_df['snapshot_date'].max()
val_mask = train_df['snapshot_date'] == val_date
tr_df = train_df[~val_mask]
vl_df = train_df[val_mask]
y_val = vl_df['churn']

# Positive features identified
positive_candidates = [
    'fraction_inactive_lifetime',
    'pf_delta_7_30',
    'support_delta_7_30',
    'sess_dur_mean_7d',
    'dur_ratio_60_90'
]

print("\n--- Step 2: Testing feature combinations ---")
base_features = v3_features.copy()

current_features = base_features.copy()
cat_idx = [current_features.index(c) for c in cat_names if c in current_features]
cb = CatBoostClassifier(
    iterations=2000, eval_metric="AUC", random_seed=42, verbose=0,
    cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
    **best_hp
)
cb.fit(tr_df[current_features], tr_df['churn'], eval_set=(vl_df[current_features], vl_df['churn']))
best_auc = roc_auc_score(y_val, cb.predict_proba(vl_df[current_features])[:, 1])
print(f"Base v3 features (count: {len(current_features)}) -> AUC: {best_auc:.5f}")

selected_new = []
for feat in positive_candidates:
    test_feats = current_features + [feat]
    cat_idx = [test_feats.index(c) for c in cat_names if c in test_feats]
    cb = CatBoostClassifier(
        iterations=2000, eval_metric="AUC", random_seed=42, verbose=0,
        cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
        **best_hp
    )
    cb.fit(tr_df[test_feats], tr_df['churn'], eval_set=(vl_df[test_feats], vl_df['churn']))
    score = roc_auc_score(y_val, cb.predict_proba(vl_df[test_feats])[:, 1])
    print(f"Adding {feat:28s} -> AUC: {score:.5f} (Δ: {score - best_auc:+.5f})")
    if score > best_auc:
        best_auc = score
        current_features = test_feats
        selected_new.append(feat)

print(f"\nFinal v4 selected features count: {len(current_features)}")
print(f"New features added: {selected_new}")
print(f"v4 Single CatBoost AUC: {best_auc:.5f}")

# Train final v4 CatBoost
v4_features = current_features
cat_idx_v4 = [v4_features.index(c) for c in cat_names if c in v4_features]
v4_cb = CatBoostClassifier(
    iterations=2000, eval_metric="AUC", random_seed=42, verbose=0,
    cat_features=cat_idx_v4, early_stopping_rounds=150, use_best_model=True,
    **best_hp
)
v4_cb.fit(tr_df[v4_features], tr_df['churn'], eval_set=(vl_df[v4_features], vl_df['churn']))
v4_single_cb_pred = v4_cb.predict_proba(vl_df[v4_features])[:, 1]
v4_single_cb_auc = roc_auc_score(y_val, v4_single_cb_pred)

# Train v4 CatBoost 5-seed ensemble
print("\n--- Step 3: Training v4 CatBoost 5-seed ensemble ---")
seeds = [42, 123, 314, 555, 777]
cb_seed_preds = []
for s in seeds:
    m = CatBoostClassifier(
        iterations=2000, eval_metric="AUC", random_seed=s, verbose=0,
        cat_features=cat_idx_v4, early_stopping_rounds=150, use_best_model=True,
        **best_hp
    )
    m.fit(tr_df[v4_features], tr_df['churn'], eval_set=(vl_df[v4_features], vl_df['churn']))
    sp = m.predict_proba(vl_df[v4_features])[:, 1]
    s_auc = roc_auc_score(y_val, sp)
    print(f"  CatBoost Seed {s}: AUC = {s_auc:.5f} (iter: {m.get_best_iteration()})")
    cb_seed_preds.append(sp)

v4_cb_ens_pred = np.mean(cb_seed_preds, axis=0)
v4_cb_ens_auc = roc_auc_score(y_val, v4_cb_ens_pred)
print(f"v4 CatBoost 5-seed ensemble AUC: {v4_cb_ens_auc:.5f}")

# Train LightGBM model on v4 features
print("\n--- Step 4: Training LightGBM on v4 features ---")
X_tr_enc = tr_df[v4_features].copy()
X_vl_enc = vl_df[v4_features].copy()
for c in cat_names:
    if c in X_tr_enc.columns:
        X_tr_enc[c] = X_tr_enc[c].cat.codes
        X_vl_enc[c] = X_vl_enc[c].cat.codes

lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.04, "num_leaves": 63,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

lgb_seed_preds = []
for s in [42, 123, 314]:
    p = lgb_params.copy()
    p["random_state"] = s
    lm = lgb.LGBMClassifier(**p)
    lm.fit(X_tr_enc, tr_df['churn'], eval_set=[(X_vl_enc, y_val)], eval_metric="auc",
           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    lp = lm.predict_proba(X_vl_enc)[:, 1]
    print(f"  LightGBM Seed {s}: AUC = {roc_auc_score(y_val, lp):.5f} (iter: {lm.best_iteration_})")
    lgb_seed_preds.append(lp)

v4_lgb_pred = np.mean(lgb_seed_preds, axis=0)
v4_lgb_auc = roc_auc_score(y_val, v4_lgb_pred)
print(f"v4 LightGBM ensemble AUC: {v4_lgb_auc:.5f}")

# Blend models
print("\n--- Step 5: Blending CatBoost and LightGBM ---")
best_blend_auc = 0
best_blend_weights = None
best_blend_pred = None

# Grid search weights
for w in np.arange(0.5, 0.95, 0.02):
    bp = w * v4_cb_ens_pred + (1 - w) * v4_lgb_pred
    ba = roc_auc_score(y_val, bp)
    if ba > best_blend_auc:
        best_blend_auc = ba
        best_blend_weights = (w, 1 - w)
        best_blend_pred = bp

print(f"Best blend: {best_blend_weights[0]:.2f}*CatBoost_ens + {best_blend_weights[1]:.2f}*LightGBM_ens -> AUC: {best_blend_auc:.5f}")

# Try 3-way blend: v4_single_cb, v4_cb_ens, v4_lgb
for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.2, 0.6, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05: continue
        bp = w1 * v4_single_cb_pred + w2 * v4_cb_ens_pred + w3 * v4_lgb_pred
        ba = roc_auc_score(y_val, bp)
        if ba > best_blend_auc:
            best_blend_auc = ba
            best_blend_weights = (w1, w2, w3)
            best_blend_pred = bp
            print(f"New best 3-way blend: {w1:.2f}*CB_single + {w2:.2f}*CB_ens + {w3:.2f}*LGB -> AUC: {best_blend_auc:.5f}")

os.makedirs('v4', exist_ok=True)
train_df[v4_features + ['churn', 'snapshot_date', 'customer_id']].to_parquet('v4/train_features_v4.parquet')
test_df[v4_features + ['snapshot_date', 'customer_id']].to_parquet('v4/test_features_v4.parquet')
v4_cb.save_model('v4/catboost_v4.cbm')

with open('v4/v4_results.json', 'w') as f:
    json.dump({
        "v3_baseline_single_auc": 0.86601,
        "v3_baseline_blend_auc": 0.86651,
        "v4_single_catboost_auc": v4_single_cb_auc,
        "v4_catboost_5seed_auc": v4_cb_ens_auc,
        "v4_lightgbm_auc": v4_lgb_auc,
        "v4_best_blend_auc": best_blend_auc,
        "v4_features": v4_features,
        "added_features": selected_new
    }, f, indent=2)

print("\nSaved v4 feature matrices, model, and results to v4/")
