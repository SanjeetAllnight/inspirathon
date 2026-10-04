import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

print("=" * 80)
print("TESTING ENSEMBLE DIVERSITY: CATBOOST 5-SEED + LIGHTGBM BLEND")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v5/train_features_v5.parquet')
test_df = pd.read_parquet('v5/test_features_v5.parquet')

with open('v5/v5_results.json') as f:
    v5_res = json.load(f)

v5_features = v5_res['v5_features']
catboost_hp = v5_res['best_v5_hp']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Load scratch parquets
df_b1 = pd.read_parquet('scratch_b1.parquet')
df_b2 = pd.read_parquet('scratch_b2.parquet')

train_df = train_df.merge(df_b1, on='customer_id', how='left')
train_df = train_df.merge(df_b2, on='customer_id', how='left')
test_df = test_df.merge(df_b1, on='customer_id', how='left')
test_df = test_df.merge(df_b2, on='customer_id', how='left')

act_feats = [f'active_days_w{w}' for w in range(1, 9)]
v6_features = v5_features + act_feats + ['session_slope_8w']
print(f"Total V6 features: {len(v6_features)}")

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

seeds_cb = [42, 123, 314, 555, 777]
c_idx = [v6_features.index(c) for c in cat_names if c in v6_features]

lgb_encoded = train_df[v6_features].copy()
for c in cat_names:
    if c in lgb_encoded.columns:
        lgb_encoded[c] = lgb_encoded[c].cat.codes

lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.04, "num_leaves": 63,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

fold_cb_ens = []
fold_lgb = []
fold_blends = {w: [] for w in [0.03, 0.05, 0.10, 0.15]}

for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    y_vl = vl_sub['churn']
    
    # 1. CB 5 seeds
    cb_preds = []
    for s in seeds_cb:
        m = CatBoostClassifier(
            iterations=1500, eval_metric="AUC", random_seed=s, verbose=0,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **catboost_hp
        )
        m.fit(tr_sub[v6_features], tr_sub['churn'], eval_set=(vl_sub[v6_features], y_vl))
        cb_preds.append(m.predict_proba(vl_sub[v6_features])[:, 1])
    
    cb_ens_p = np.mean(cb_preds, axis=0)
    cb_auc = roc_auc_score(y_vl, cb_ens_p)
    fold_cb_ens.append(cb_auc)
    
    # 2. LGB
    lm = lgb.LGBMClassifier(**lgb_params)
    lm.fit(
        lgb_encoded[tr_mask], tr_sub['churn'],
        eval_set=[(lgb_encoded[vl_mask], y_vl)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    lgb_p = lm.predict_proba(lgb_encoded[vl_mask])[:, 1]
    lgb_auc = roc_auc_score(y_vl, lgb_p)
    fold_lgb.append(lgb_auc)
    
    print(f"\n{fold_name}:")
    print(f"  CB 5-Seed AUC : {cb_auc:.5f}")
    print(f"  LGB AUC       : {lgb_auc:.5f}")
    
    for w in fold_blends.keys():
        b_p = (1.0 - w) * cb_ens_p + w * lgb_p
        b_auc = roc_auc_score(y_vl, b_p)
        fold_blends[w].append(b_auc)
        print(f"  Blend ({1.0-w:.2f} CB + {w:.2f} LGB): {b_auc:.5f} (Δ: {b_auc - cb_auc:+.5f})")

print("\n" + "=" * 80)
print("OVERALL SUMMARY ACROSS 3 FOLDS")
print("=" * 80)
print(f"V5 5-Seed Ensemble Mean CV: {v5_res['v5_5seed_mean_cv']:.5f}")
print(f"V6 5-Seed CatBoost Mean CV: {np.mean(fold_cb_ens):.5f} (Δ: {np.mean(fold_cb_ens) - v5_res['v5_5seed_mean_cv']:+.5f})")
print(f"V6 LightGBM Single Mean CV: {np.mean(fold_lgb):.5f}")

for w, b_aucs in fold_blends.items():
    print(f"V6 Blend ({1.0-w:.2f} CB + {w:.2f} LGB) Mean CV: {np.mean(b_aucs):.5f} (Δ vs CB: {np.mean(b_aucs) - np.mean(fold_cb_ens):+.5f})")

# Feature importances of V6 model
print("\n--- Feature Importance Analysis of V6 ---")
full_m = CatBoostClassifier(
    iterations=500, eval_metric="AUC", random_seed=42, verbose=0,
    cat_features=c_idx, **catboost_hp
)
full_m.fit(train_df[v6_features], train_df['churn'])
fi = pd.Series(full_m.get_feature_importance(), index=v6_features).sort_values(ascending=False)

print("\nTop 25 Most Important Features in V6:")
for i, (name, imp) in enumerate(fi.head(25).items(), 1):
    new_marker = " *** NEW V6 ***" if name in act_feats + ['session_slope_8w'] else ""
    print(f"  {i:2d}. {name:30s}: {imp:.3f}%{new_marker}")

print("\nRankings of All 9 New V6 Features:")
for name in ['session_slope_8w'] + act_feats:
    rank = list(fi.index).index(name) + 1
    imp = fi[name]
    print(f"  Rank {rank:2d}/{len(v6_features)}: {name:25s} (Importance: {imp:.3f}%)")

os.makedirs('v6', exist_ok=True)
with open('v6/v6_final_results.json', 'w') as f:
    json.dump({
        "v5_baseline_mean_cv": v5_res['v5_5seed_mean_cv'],
        "v6_cb_5seed_mean_cv": float(np.mean(fold_cb_ens)),
        "v6_cb_5seed_folds": [float(x) for x in fold_cb_ens],
        "v6_lgb_mean_cv": float(np.mean(fold_lgb)),
        "v6_lgb_folds": [float(x) for x in fold_lgb],
        "v6_features": v6_features,
        "n_features": len(v6_features),
        "new_features": act_feats + ['session_slope_8w'],
        "top_features": fi.head(20).to_dict()
    }, f, indent=2)

print("\nSaved V6 final results to v6/v6_final_results.json")
