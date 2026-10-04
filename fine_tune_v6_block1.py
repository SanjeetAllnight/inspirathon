import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("DISSECTING BLOCK 1: WEEKLY ACTIVITY PROFILE ABLATION & EXPANSION")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v5/train_features_v5.parquet')

with open('v5/v5_results.json') as f:
    v5_res = json.load(f)

v5_features = v5_res['v5_features']
catboost_hp = v5_res['best_v5_hp']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Load Block 1 and Block 2
df_b1 = pd.read_parquet('scratch_b1.parquet')
df_b2 = pd.read_parquet('scratch_b2.parquet')

train_df = train_df.merge(df_b1, on='customer_id', how='left')
train_df = train_df.merge(df_b2, on='customer_id', how='left')

# Split definitions
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
        auc = roc_auc_score(vl_sub['churn'], preds)
        fold_aucs.append(auc)
        
    return fold_aucs, np.mean(fold_aucs)

base_mean = v5_res['v5_single_cb_mean_cv']
print(f"V5 Baseline CV Mean: {base_mean:.5f}")

# Sub-groups of Block 1
metric_types = ['sess_count', 'active_days', 'sess_dur', 'support', 'pf', 'pc']
sub_groups = {}
for m in metric_types:
    sub_groups[m] = [f'{m}_w{w}' for w in range(1, 9)]

print("\n--- Evaluating Individual Metric Types (8 features each) ---")
metric_results = {}
for m in metric_types:
    feats = sub_groups[m]
    f_res, m_res = eval_cv(v5_features + feats, catboost_hp)
    delta = m_res - base_mean
    metric_results[m] = (m_res, delta, f_res)
    print(f"  + {m:12s} ({len(feats)} feats) -> Mean: {m_res:.5f} (Δ: {delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}")

# Cumulative forward addition of positive metric types
print("\n--- Testing Progressive Combinations ---")
# Sort metrics by delta
sorted_metrics = sorted(metric_types, key=lambda m: metric_results[m][1], reverse=True)
print("Ranking of metrics:", [(m, f"{metric_results[m][1]:+.5f}") for m in sorted_metrics])

current_feats = list(v5_features)
current_mean = base_mean
best_combination = list(v5_features)

for m in sorted_metrics:
    cand_feats = current_feats + sub_groups[m]
    f_res, m_res = eval_cv(cand_feats, catboost_hp)
    delta = m_res - base_mean
    step_delta = m_res - current_mean
    print(f"  + Adding {m:12s} -> Mean: {m_res:.5f} (Δ base: {delta:+.5f}, Δ step: {step_delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}")
    if step_delta > 0:
        current_feats = cand_feats
        current_mean = m_res
        best_combination = cand_feats

print(f"\nBest Weekly Profile combination: {len(best_combination)} features with Mean CV: {current_mean:.5f} (Δ: {current_mean - base_mean:+.5f})")

# Now check if any specific trajectory feature adds on top of this best combination
print("\n--- Testing Individual Trajectory Features on top of Best Combination ---")
b2_features = [c for c in df_b2.columns if c != 'customer_id']
traj_winners = []
for tf in b2_features:
    f_res, m_res = eval_cv(best_combination + [tf], catboost_hp)
    delta = m_res - current_mean
    if delta >= 0.0001:
        print(f"  + {tf:28s} -> Mean: {m_res:.5f} (Δ: {delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f} *WINNER*")
        traj_winners.append(tf)
    else:
        print(f"  + {tf:28s} -> Mean: {m_res:.5f} (Δ: {delta:+.5f})")

if traj_winners:
    f_all_traj, m_all_traj = eval_cv(best_combination + traj_winners, catboost_hp)
    print(f"\nBest Combination + Trajectory Winners ({len(traj_winners)} feats) -> Mean: {m_all_traj:.5f} (Δ: {m_all_traj - base_mean:+.5f})")
    if m_all_traj > current_mean:
        best_combination = best_combination + traj_winners
        current_mean = m_all_traj

# Save final ablation results to json
with open('v6_ablation_results.json', 'w') as f:
    json.dump({
        "v5_baseline_mean": base_mean,
        "best_v6_mean": current_mean,
        "improvement": current_mean - base_mean,
        "best_v6_features": best_combination,
        "n_best_features": len(best_combination)
    }, f, indent=2)

print("\nAblation complete and saved to v6_ablation_results.json")
