import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import json
import time

print("Loading candidate datasets...")
train_df = pd.read_parquet('scratch_train_v4_candidates.parquet')

with open('v3/best_config.json') as f:
    v3_config = json.load(f)

v3_features = v3_config['features']
best_hp = v3_config['best_hp']
cat_names = v3_config['cat_features']

val_date = train_df['snapshot_date'].max()
val_mask = train_df['snapshot_date'] == val_date
tr_df = train_df[~val_mask]
vl_df = train_df[val_mask]

def eval_features(feat_list, name=""):
    cat_idx = [feat_list.index(c) for c in cat_names if c in feat_list]
    cb = CatBoostClassifier(
        iterations=2000, eval_metric="AUC", random_seed=42, verbose=0,
        cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
        **best_hp
    )
    t0 = time.time()
    cb.fit(tr_df[feat_list], tr_df['churn'], eval_set=(vl_df[feat_list], vl_df['churn']))
    preds = cb.predict_proba(vl_df[feat_list])[:, 1]
    auc = roc_auc_score(vl_df['churn'], preds)
    dt = time.time() - t0
    print(f"[{name}] AUC: {auc:.5f} (Δ: {auc - 0.86601:+.5f}) in {dt:.1f}s | iter: {cb.get_best_iteration()} | feats: {len(feat_list)}")
    return auc, preds, cb

print("\n--- Baseline v3 Evaluation ---")
base_auc, base_preds, base_cb = eval_features(v3_features, "v3 baseline")

# Group 1: Inactivity
g1 = ['longest_session_gap', 'n_session_gaps_gt_7', 'n_session_gaps_gt_14', 'n_session_gaps_gt_30', 'fraction_inactive_lifetime']
auc_g1, _, _ = eval_features(v3_features + g1, "+ Group 1 (Inactivity)")

# Group 2: Session duration trends
g2 = ['sess_dur_mean_7d', 'dur_ratio_7_30', 'dur_ratio_14_30', 'dur_ratio_60_90', 'dur_ratio_30_90', 'dur_diff_30_hist', 'dur_ratio_30_hist', 'dur_diff_7_30']
auc_g2, _, _ = eval_features(v3_features + g2, "+ Group 2 (Duration Trends)")

# Group 3: Event activity trends
g3 = [
    'support_7d', 'support_30d', 'support_ratio_7_30', 'support_ratio_30_60', 'support_delta_7_30', 'support_delta_30_60',
    'pf_7d', 'pf_30d', 'pf_60d', 'pf_ratio_7_30', 'pf_ratio_30_60', 'pf_delta_7_30', 'pf_delta_30_60',
    'plan_change_7d', 'plan_change_30d', 'plan_change_60d', 'plan_change_ratio_30_60', 'plan_change_delta_30_60'
]
auc_g3, _, _ = eval_features(v3_features + g3, "+ Group 3 (Event Trends)")

# Group 4: Targeted Interactions
g4 = [
    'tenure_x_plan_premium', 'recency_x_billing', 'recency_event_x_billing',
    'pf_recency_x_billing', 'pf_60d_x_billing', 'support_60d_x_plan_basic', 'support_count_x_plan_basic'
]
auc_g4, _, _ = eval_features(v3_features + g4, "+ Group 4 (Interactions)")

# Combine all groups
all_candidates = g1 + g2 + g3 + g4
auc_all, _, _ = eval_features(v3_features + all_candidates, "+ All Candidate Groups")
