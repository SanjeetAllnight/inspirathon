import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import os

print("=" * 70)
print("V5: OPTIMIZING WINNING FEATURE SET & CATBOOST HYPERPARAMETERS")
print("=" * 70)

# Load base data
train_df = pd.read_parquet('v4/train_features_v4.parquet')
test_df = pd.read_parquet('v4/test_features_v4.parquet')

with open('v4/v4_results.json') as f:
    v4_results = json.load(f)

v4_features = v4_results['v4_features']

with open('v3/best_config.json') as f:
    v3_config = json.load(f)

best_hp = v3_config['best_hp']
cat_names = v3_config['cat_features']

# Load candidate B and C features
exp_b = pd.read_parquet('scratch_exp_b.parquet')
exp_c = pd.read_parquet('scratch_exp_c.parquet')

train_df = train_df.merge(exp_b, on='customer_id', how='left')
train_df = train_df.merge(exp_c, on='customer_id', how='left')
test_df = test_df.merge(exp_b, on='customer_id', how='left')
test_df = test_df.merge(exp_c, on='customer_id', how='left')

# Impute missing kth-last sessions with account_age_days
for k in [2, 3, 5, 10]:
    col = f'days_since_{k}th_last_session'
    train_df[col] = train_df[col].fillna(train_df['account_age_days'])
    test_df[col] = test_df[col].fillna(test_df['account_age_days'])

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

def eval_cv(feature_list, hp, seed=42):
    c_idx = [feature_list.index(c) for c in cat_names if c in feature_list]
    fold_aucs = []
    
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        
        m = CatBoostClassifier(
            iterations=1500, eval_metric="AUC", random_seed=seed, verbose=0,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **hp
        )
        m.fit(tr_sub[feature_list], tr_sub['churn'], eval_set=(vl_sub[feature_list], vl_sub['churn']))
        preds = m.predict_proba(vl_sub[feature_list])[:, 1]
        fold_aucs.append(roc_auc_score(vl_sub['churn'], preds))
        
    return fold_aucs, np.mean(fold_aucs)

# Step 1: Compare B1 combinations
print("\n[Step 1] Fine-tuning B1 kth-last session features...")
b1_all = [f'days_since_{k}th_last_session' for k in [2, 3, 5, 10]]
b1_top = [f'days_since_{k}th_last_session' for k in [2, 3, 5]]

f_all, m_all = eval_cv(v4_features + b1_all, best_hp)
print(f"  v4 + all 4 kth sessions (2,3,5,10) -> F1: {f_all[0]:.5f} | F2: {f_all[1]:.5f} | F3: {f_all[2]:.5f} | Mean: {m_all:.5f}")

f_top, m_top = eval_cv(v4_features + b1_top, best_hp)
print(f"  v4 + top 3 kth sessions (2,3,5)    -> F1: {f_top[0]:.5f} | F2: {f_top[1]:.5f} | F3: {f_top[2]:.5f} | Mean: {m_top:.5f}")

# Select the better of the two
winning_kth = b1_all if m_all >= m_top else b1_top
v5_base_feats = v4_features + winning_kth
print(f"  Selected kth-session set: {winning_kth}")

# Check if adding overdue_ratio_median or sessions_since_last_support helps on top
f_overdue, m_overdue = eval_cv(v5_base_feats + ['overdue_ratio_median'], best_hp)
print(f"  + overdue_ratio_median            -> Mean: {m_overdue:.5f} (Δ: {m_overdue - max(m_all, m_top):+.5f})")

f_support, m_support = eval_cv(v5_base_feats + ['sessions_since_last_support'], best_hp)
print(f"  + sessions_since_last_support     -> Mean: {m_support:.5f} (Δ: {m_support - max(m_all, m_top):+.5f})")

v5_features = v5_base_feats
if m_overdue > max(m_all, m_top) + 0.0001:
    v5_features = v5_features + ['overdue_ratio_median']
if m_support > max(m_all, m_top) + 0.0001:
    v5_features = v5_features + ['sessions_since_last_support']

print(f"\nFinal V5 feature count: {len(v5_features)} (added {len(v5_features) - len(v4_features)} features)")

# Step 2: Small hyperparameter tuning sweep around best config
print("\n[Step 2] CatBoost Hyperparameter Tuning Sweep...")
hp_candidates = [
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1, "random_strength": 1, "bagging_temperature": 1},  # current best
    {"depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 1, "random_strength": 1, "bagging_temperature": 1},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 2, "random_strength": 1, "bagging_temperature": 1},
    {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 1, "random_strength": 1, "bagging_temperature": 1},
    {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 2, "random_strength": 1, "bagging_temperature": 1},
    {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 1, "random_strength": 0.5, "bagging_temperature": 0.8},
]

best_v5_hp = None
best_v5_mean = 0
best_v5_folds = None

for idx, hp in enumerate(hp_candidates):
    f_res, m_res = eval_cv(v5_features, hp)
    print(f"  Config {idx}: Mean={m_res:.5f} | F1={f_res[0]:.5f}, F2={f_res[1]:.5f}, F3={f_res[2]:.5f} | {hp}")
    if m_res > best_v5_mean:
        best_v5_mean = m_res
        best_v5_hp = hp
        best_v5_folds = f_res

print(f"\nBest V5 CatBoost config: {best_v5_hp}")
print(f"Best V5 Single CatBoost CV Mean: {best_v5_mean:.5f}")

# Step 3: Evaluate 5-seed ensemble across all 3 folds with winning V5 model
print("\n[Step 3] Evaluating V5 5-Seed Ensemble across all 3 temporal folds...")
seeds = [42, 123, 314, 555, 777]
c_idx = [v5_features.index(c) for c in cat_names if c in v5_features]

fold_ens_aucs = []
for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    
    seed_preds = []
    for s in seeds:
        m = CatBoostClassifier(
            iterations=1500, eval_metric="AUC", random_seed=s, verbose=0,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **best_v5_hp
        )
        m.fit(tr_sub[v5_features], tr_sub['churn'], eval_set=(vl_sub[v5_features], vl_sub['churn']))
        seed_preds.append(m.predict_proba(vl_sub[v5_features])[:, 1])
        
    ens_pred = np.mean(seed_preds, axis=0)
    ens_auc = roc_auc_score(vl_sub['churn'], ens_pred)
    fold_ens_aucs.append(ens_auc)
    print(f"  {fold_name}: 5-Seed Ensemble AUC = {ens_auc:.5f}")

mean_ens_auc = np.mean(fold_ens_aucs)
print(f"\nV5 5-Seed Ensemble Mean Temporal CV AUC: {mean_ens_auc:.5f}")

# Save V5 feature matrices and metadata
os.makedirs('v5', exist_ok=True)
train_df[v5_features + ['churn', 'snapshot_date', 'customer_id']].to_parquet('v5/train_features_v5.parquet')
test_df[v5_features + ['snapshot_date', 'customer_id']].to_parquet('v5/test_features_v5.parquet')

with open('v5/v5_results.json', 'w') as f:
    json.dump({
        "v4_mean_temporal_cv": 0.86375,
        "v4_single_cb_mean_cv": 0.86294,
        "v4_cb_5seed_mean_cv": 0.86384,
        "v5_single_cb_mean_cv": best_v5_mean,
        "v5_single_cb_folds": best_v5_folds,
        "v5_5seed_mean_cv": mean_ens_auc,
        "v5_5seed_folds": fold_ens_aucs,
        "improvement_over_v4_blend": mean_ens_auc - 0.86375,
        "improvement_over_v4_cb_ens": mean_ens_auc - 0.86384,
        "best_v5_hp": best_v5_hp,
        "v5_features": v5_features,
        "retained_block": winning_kth
    }, f, indent=2)

print("\nSaved V5 artifacts to v5/")
