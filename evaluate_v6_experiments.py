import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("EVALUATING V6 CANDIDATE FEATURE BLOCKS ON 3 FORWARD TEMPORAL FOLDS")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v5/train_features_v5.parquet')

with open('v5/v5_results.json') as f:
    v5_res = json.load(f)

v5_features = v5_res['v5_features']
catboost_hp = v5_res['best_v5_hp']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Load candidate parquets
df_b4 = pd.read_parquet('scratch_b4.parquet')
df_b3 = pd.read_parquet('scratch_b3.parquet')
df_b1 = pd.read_parquet('scratch_b1.parquet')
df_b2 = pd.read_parquet('scratch_b2.parquet')

train_df = train_df.merge(df_b4, on='customer_id', how='left')
train_df = train_df.merge(df_b3, on='customer_id', how='left')
train_df = train_df.merge(df_b1, on='customer_id', how='left')
train_df = train_df.merge(df_b2, on='customer_id', how='left')

# Impute missing kth-last sessions with account_age_days
for k in [15, 20]:
    col = f'days_since_{k}th_last_session'
    train_df[col] = train_df[col].fillna(train_df['account_age_days'])

b4_features = [f'days_since_{k}th_last_session' for k in [15, 20]]
b3_features = [c for c in df_b3.columns if c != 'customer_id']
b1_features = [c for c in df_b1.columns if c != 'customer_id']
b2_features = [c for c in df_b2.columns if c != 'customer_id']

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

# Baseline V5 evaluation
print("\n[Baseline] Running V5 Baseline...")
base_folds, base_mean = eval_cv(v5_features, catboost_hp)
print(f"V5 Baseline -> Mean: {base_mean:.5f} | F1: {base_folds[0]:.5f}, F2: {base_folds[1]:.5f}, F3: {base_folds[2]:.5f}")

# 1. Test Block 4 (Extended Recency Progression)
print("\n[Block 4] Evaluating Extended Recency Progression (k=15, 20)...")
b4_folds, b4_mean = eval_cv(v5_features + b4_features, catboost_hp)
delta_b4 = b4_mean - base_mean
print(f"V5 + Block 4 ({len(b4_features)} feats) -> Mean: {b4_mean:.5f} (Δ: {delta_b4:+.5f}) | F1: {b4_folds[0]:.5f}, F2: {b4_folds[1]:.5f}, F3: {b4_folds[2]:.5f}")

# Also test k=15 only
b4_15_folds, b4_15_mean = eval_cv(v5_features + ['days_since_15th_last_session'], catboost_hp)
delta_b4_15 = b4_15_mean - base_mean
print(f"V5 + k=15 only     -> Mean: {b4_15_mean:.5f} (Δ: {delta_b4_15:+.5f}) | F1: {b4_15_folds[0]:.5f}, F2: {b4_15_folds[1]:.5f}, F3: {b4_15_folds[2]:.5f}")

# 2. Test Block 3 (Event Composition)
print("\n[Block 3] Evaluating Event Composition (30, 60, 90d)...")
b3_folds, b3_mean = eval_cv(v5_features + b3_features, catboost_hp)
delta_b3 = b3_mean - base_mean
print(f"V5 + Block 3 ({len(b3_features)} feats) -> Mean: {b3_mean:.5f} (Δ: {delta_b3:+.5f}) | F1: {b3_folds[0]:.5f}, F2: {b3_folds[1]:.5f}, F3: {b3_folds[2]:.5f}")

# Test sub-groups of Block 3: fractions vs cross-ratios
b3_fracs = [c for c in b3_features if 'frac' in c]
b3_ratios = [c for c in b3_features if 'to' in c]
b3_f_folds, b3_f_mean = eval_cv(v5_features + b3_fracs, catboost_hp)
print(f"V5 + B3 Fractions ({len(b3_fracs)} feats) -> Mean: {b3_f_mean:.5f} (Δ: {b3_f_mean - base_mean:+.5f}) | F1: {b3_f_folds[0]:.5f}, F2: {b3_f_folds[1]:.5f}, F3: {b3_f_folds[2]:.5f}")

# 3. Test Block 2 (Trajectory Features)
print("\n[Block 2] Evaluating Trajectory Features (14 feats)...")
b2_folds, b2_mean = eval_cv(v5_features + b2_features, catboost_hp)
delta_b2 = b2_mean - base_mean
print(f"V5 + Block 2 ({len(b2_features)} feats) -> Mean: {b2_mean:.5f} (Δ: {delta_b2:+.5f}) | F1: {b2_folds[0]:.5f}, F2: {b2_folds[1]:.5f}, F3: {b2_folds[2]:.5f}")

# Test key sub-groups of Block 2
b2_slopes = [c for c in b2_features if 'slope' in c]
b2_slopes_folds, b2_slopes_mean = eval_cv(v5_features + b2_slopes, catboost_hp)
print(f"V5 + B2 Slopes ({len(b2_slopes)} feats)    -> Mean: {b2_slopes_mean:.5f} (Δ: {b2_slopes_mean - base_mean:+.5f}) | F1: {b2_slopes_folds[0]:.5f}, F2: {b2_slopes_folds[1]:.5f}, F3: {b2_slopes_folds[2]:.5f}")

b2_non_slopes = [c for c in b2_features if 'slope' not in c]
b2_ns_folds, b2_ns_mean = eval_cv(v5_features + b2_non_slopes, catboost_hp)
print(f"V5 + B2 Non-slopes ({len(b2_non_slopes)} feats)-> Mean: {b2_ns_mean:.5f} (Δ: {b2_ns_mean - base_mean:+.5f}) | F1: {b2_ns_folds[0]:.5f}, F2: {b2_ns_folds[1]:.5f}, F3: {b2_ns_folds[2]:.5f}")

# 4. Test Block 1 (Weekly Raw Profile)
print("\n[Block 1] Evaluating Weekly Activity Profile (48 feats)...")
# Test sessions only first
b1_sess = [f'sess_count_w{w}' for w in range(1, 9)]
b1_s_folds, b1_s_mean = eval_cv(v5_features + b1_sess, catboost_hp)
print(f"V5 + B1 Weekly Sessions (8 feats) -> Mean: {b1_s_mean:.5f} (Δ: {b1_s_mean - base_mean:+.5f}) | F1: {b1_s_folds[0]:.5f}, F2: {b1_s_folds[1]:.5f}, F3: {b1_s_folds[2]:.5f}")

b1_folds, b1_mean = eval_cv(v5_features + b1_features, catboost_hp)
delta_b1 = b1_mean - base_mean
print(f"V5 + Block 1 All ({len(b1_features)} feats) -> Mean: {b1_mean:.5f} (Δ: {delta_b1:+.5f}) | F1: {b1_folds[0]:.5f}, F2: {b1_folds[1]:.5f}, F3: {b1_folds[2]:.5f}")

print("\n" + "=" * 80)
print("BLOCK SUMMARY RESULTS")
print("=" * 80)
results = [
    ("V5 Baseline", len(v5_features), base_mean, 0.0, base_folds),
    ("Block 4 (Ext Recency)", len(b4_features), b4_mean, delta_b4, b4_folds),
    ("Block 4 (k=15 only)", 1, b4_15_mean, delta_b4_15, b4_15_folds),
    ("Block 3 (Event Composition)", len(b3_features), b3_mean, delta_b3, b3_folds),
    ("Block 3 (Fractions only)", len(b3_fracs), b3_f_mean, b3_f_mean - base_mean, b3_f_folds),
    ("Block 2 (Trajectories)", len(b2_features), b2_mean, delta_b2, b2_folds),
    ("Block 2 (Slopes only)", len(b2_slopes), b2_slopes_mean, b2_slopes_mean - base_mean, b2_slopes_folds),
    ("Block 2 (Non-slopes)", len(b2_non_slopes), b2_ns_mean, b2_ns_mean - base_mean, b2_ns_folds),
    ("Block 1 (Weekly Sessions)", len(b1_sess), b1_s_mean, b1_s_mean - base_mean, b1_s_folds),
    ("Block 1 (All Weekly Profile)", len(b1_features), b1_mean, delta_b1, b1_folds),
]

summary_df = pd.DataFrame([{
    'Block': r[0],
    'N_Feats': r[1],
    'Mean_CV': r[2],
    'Delta_CV': r[3],
    'Fold_1': r[4][0],
    'Fold_2': r[4][1],
    'Fold_3': r[4][2]
} for r in results])
print(summary_df.to_string(index=False))
summary_df.to_csv('v6_block_eval_results.csv', index=False)
