import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import json
import time

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

candidates = [
    'longest_session_gap', 'n_session_gaps_gt_7', 'n_session_gaps_gt_14', 'n_session_gaps_gt_30', 'fraction_inactive_lifetime',
    'sess_dur_mean_7d', 'dur_ratio_7_30', 'dur_ratio_14_30', 'dur_ratio_60_90', 'dur_ratio_30_90', 'dur_diff_30_hist', 'dur_ratio_30_hist', 'dur_diff_7_30',
    'support_7d', 'support_30d', 'support_ratio_7_30', 'support_ratio_30_60', 'support_delta_7_30', 'support_delta_30_60',
    'pf_7d', 'pf_30d', 'pf_60d', 'pf_ratio_7_30', 'pf_ratio_30_60', 'pf_delta_7_30', 'pf_delta_30_60',
    'plan_change_7d', 'plan_change_30d', 'plan_change_60d', 'plan_change_ratio_30_60', 'plan_change_delta_30_60',
    'tenure_x_plan_premium', 'recency_x_billing', 'recency_event_x_billing',
    'pf_recency_x_billing', 'pf_60d_x_billing', 'support_60d_x_plan_basic', 'support_count_x_plan_basic'
]

print(f"Total candidate features: {len(candidates)}")

# Fit joint model to inspect feature importance of new candidates
all_feats = v3_features + candidates
cat_idx = [all_feats.index(c) for c in cat_names if c in all_feats]

cb_joint = CatBoostClassifier(
    iterations=1500, eval_metric="AUC", random_seed=42, verbose=0,
    cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
    **best_hp
)
cb_joint.fit(tr_df[all_feats], tr_df['churn'], eval_set=(vl_df[all_feats], vl_df['churn']))
imp_df = pd.DataFrame({'feature': all_feats, 'importance': cb_joint.get_feature_importance()})
cand_imp = imp_df[imp_df['feature'].isin(candidates)].sort_values('importance', ascending=False)
print("\nCandidate Feature Importances:")
print(cand_imp.to_string(index=False))

# Now test adding candidates individually
base_auc = 0.86601
results = []
print("\nTesting individual candidate additions to v3 baseline...")
for feat in cand_imp['feature'].tolist():
    trial_feats = v3_features + [feat]
    c_idx = [trial_feats.index(c) for c in cat_names if c in trial_feats]
    cb = CatBoostClassifier(
        iterations=1500, eval_metric="AUC", random_seed=42, verbose=0,
        cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
        **best_hp
    )
    cb.fit(tr_df[trial_feats], tr_df['churn'], eval_set=(vl_df[trial_feats], vl_df['churn']))
    preds = cb.predict_proba(vl_df[trial_feats])[:, 1]
    auc = roc_auc_score(vl_df['churn'], preds)
    delta = auc - base_auc
    results.append({'feature': feat, 'auc': auc, 'delta': delta, 'importance': cand_imp.loc[cand_imp['feature']==feat, 'importance'].values[0]})
    sign = "+" if delta >= 0 else ""
    print(f"  {feat:30s} -> AUC: {auc:.5f} ({sign}{delta:+.5f})")

res_df = pd.DataFrame(results).sort_values('delta', ascending=False)
print("\nSummary of Individual Feature Impact:")
print(res_df.to_string(index=False))
res_df.to_csv('candidate_impact_results.csv', index=False)
