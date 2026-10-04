import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 70)
print("V5 EXPERIMENTAL EVALUATION ACROSS 3-FOLD FORWARD TEMPORAL CV")
print("=" * 70)

# Load V4 train features and results
train_df = pd.read_parquet('v4/train_features_v4.parquet')
with open('v4/v4_results.json') as f:
    v4_results = json.load(f)

v4_features = v4_results['v4_features']

with open('v3/best_config.json') as f:
    v3_config = json.load(f)

best_hp = v3_config['best_hp']
cat_names = v3_config['cat_features']

# Load candidate feature sets
exp_a = pd.read_parquet('scratch_exp_a.parquet')
exp_b = pd.read_parquet('scratch_exp_b.parquet')
exp_c = pd.read_parquet('scratch_exp_c.parquet')

# Merge into train_df
train_df = train_df.merge(exp_a, on='customer_id', how='left')
train_df = train_df.merge(exp_b, on='customer_id', how='left')
train_df = train_df.merge(exp_c, on='customer_id', how='left')

# Impute any missing in kth-last sessions with account_age_days
for k in [2, 3, 5, 10]:
    col = f'days_since_{k}th_last_session'
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df['account_age_days'])

# Fill remaining numeric NaNs in new candidate features with 0
new_cols = [c for c in list(exp_a.columns) + list(exp_b.columns) + list(exp_c.columns) if c != 'customer_id']
for col in new_cols:
    if col in train_df.columns and train_df[col].isna().sum() > 0:
        train_df[col] = train_df[col].fillna(0)

# Temporal CV Folds
cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

def eval_feature_set(feature_list, name=""):
    c_idx = [feature_list.index(c) for c in cat_names if c in feature_list]
    fold_aucs = []
    
    t_start = time.time()
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        
        m = CatBoostClassifier(
            iterations=1500, eval_metric="AUC", random_seed=42, verbose=0,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **best_hp
        )
        m.fit(tr_sub[feature_list], tr_sub['churn'], eval_set=(vl_sub[feature_list], vl_sub['churn']))
        preds = m.predict_proba(vl_sub[feature_list])[:, 1]
        score = roc_auc_score(vl_sub['churn'], preds)
        fold_aucs.append(score)
        
    mean_auc = np.mean(fold_aucs)
    elapsed = time.time() - t_start
    return fold_aucs, mean_auc, elapsed

print("\n--- Evaluating V4 Baseline on 3-Fold Temporal CV ---")
base_folds, base_mean, base_time = eval_feature_set(v4_features, "V4 Baseline")
print(f"V4 Baseline: Fold1={base_folds[0]:.5f}, Fold2={base_folds[1]:.5f}, Fold3={base_folds[2]:.5f} | Mean={base_mean:.5f} ({base_time:.1f}s)")

results_summary = []

def test_addition(new_feats, label):
    test_features = v4_features + new_feats
    folds, mean_auc, dt = eval_feature_set(test_features, label)
    delta = mean_auc - base_mean
    sign = "+" if delta >= 0 else ""
    passed = (delta >= 0.00030) and (folds[0] >= base_folds[0] - 0.0005) and (folds[1] >= base_folds[1] - 0.0005) and (folds[2] >= base_folds[2] - 0.0005)
    flag = "ACCEPT" if passed else "REJECT"
    print(f"  {label:35s} | F1: {folds[0]:.5f} | F2: {folds[1]:.5f} | F3: {folds[2]:.5f} | Mean: {mean_auc:.5f} ({sign}{delta:+.5f}) [{flag}]")
    results_summary.append({
        "label": label, "features": new_feats,
        "fold1": folds[0], "fold2": folds[1], "fold3": folds[2],
        "mean_auc": mean_auc, "delta": delta, "passed": passed
    })
    return passed, delta

# ---------------------------------------------------------------------------
# EXPERIMENT A: Billing-Cycle Phase
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT A — BILLING-CYCLE PHASE")
print("=" * 70)

exp_a_features = [
    'annual_days_to_anniversary',
    'annual_near_renewal',
    'annual_cycle_position',
    'monthly_cycle_position',
    'near_first_monthly_renewal',
    'near_first_annual_renewal'
]

# Test individually
for f in exp_a_features:
    test_addition([f], f"Exp A: {f}")

# Test entire Exp A block
test_addition(exp_a_features, "Exp A: Full Block")

# ---------------------------------------------------------------------------
# EXPERIMENT B: Sparse-Aware Activity
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT B — SPARSE-AWARE ACTIVITY")
print("=" * 70)

b1_kth = [f'days_since_{k}th_last_session' for k in [2, 3, 5, 10]]
b2_exposure = ['events_per_exposure_day', 'sessions_per_exposure_day', 'dur_per_exposure_day']
b3_ewma = ['session_ewma_7', 'session_ewma_21', 'session_ewma_60', 'event_ewma_7', 'event_ewma_21', 'event_ewma_60']
b4_overdue = ['overdue_ratio_mean', 'overdue_ratio_median']

print("\n--- Sub-block B1: kth-last sessions ---")
for f in b1_kth:
    test_addition([f], f"Exp B1: {f}")
test_addition(b1_kth, "Exp B1: Full Sub-block")

print("\n--- Sub-block B2: Exposure-adjusted activity ---")
for f in b2_exposure:
    test_addition([f], f"Exp B2: {f}")
test_addition(b2_exposure, "Exp B2: Full Sub-block")

print("\n--- Sub-block B3: EWMA Decayed Activity ---")
for f in b3_ewma:
    test_addition([f], f"Exp B3: {f}")
test_addition(b3_ewma, "Exp B3: Full Sub-block")

print("\n--- Sub-block B4: Overdue Ratios ---")
for f in b4_overdue:
    test_addition([f], f"Exp B4: {f}")
test_addition(b4_overdue, "Exp B4: Full Sub-block")

# ---------------------------------------------------------------------------
# EXPERIMENT C: Event State / Order
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT C — EVENT STATE / ORDER")
print("=" * 70)

exp_c_features = [
    'sessions_since_last_pf',
    'sessions_since_last_support',
    'last_plan_change_direction',
    'support_followed_by_inactivity',
    'pf_followed_by_inactivity'
]

for f in exp_c_features:
    test_addition([f], f"Exp C: {f}")

test_addition(exp_c_features, "Exp C: Full Block")

# Save detailed results
res_df = pd.DataFrame(results_summary)
res_df.to_csv('v5_experimental_results.csv', index=False)
print("\nSaved all experimental results to v5_experimental_results.csv")
