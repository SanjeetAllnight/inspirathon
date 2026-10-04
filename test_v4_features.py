import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import json
import os

print("--- Step 1: Loading v3 feature matrices and events ---")
t0 = time.time()
train_df = pd.read_parquet('v3/train_features_v3.parquet')
test_df = pd.read_parquet('v3/test_features_v3.parquet')

with open('v3/best_config.json') as f:
    v3_config = json.load(f)

v3_features = v3_config['features']
best_hp = v3_config['best_hp']
cat_names = v3_config['cat_features']

print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
print(f"v3 baseline features count: {len(v3_features)}")

# Temporal split
val_date = train_df['snapshot_date'].max()
val_mask = train_df['snapshot_date'] == val_date
tr_df = train_df[~val_mask]
vl_df = train_df[val_mask]

# Baseline CatBoost evaluation
cat_idx_v3 = [v3_features.index(c) for c in cat_names if c in v3_features]
cb_base = CatBoostClassifier(
    iterations=2000, eval_metric="AUC", random_seed=42, verbose=0,
    cat_features=cat_idx_v3, early_stopping_rounds=150, use_best_model=True,
    **best_hp
)
cb_base.fit(tr_df[v3_features], tr_df['churn'], eval_set=(vl_df[v3_features], vl_df['churn']))
base_auc = roc_auc_score(vl_df['churn'], cb_base.predict_proba(vl_df[v3_features])[:, 1])
print(f"Verified v3 CatBoost baseline AUC: {base_auc:.5f} (best iter: {cb_base.get_best_iteration()})")

# Step 2: Compute session gaps from events.csv
print("\n--- Step 2: Computing session gap features from events.csv ---")
events = pd.read_csv('events.csv', parse_dates=['timestamp'])
snapshots = pd.concat([
    train_df[['customer_id', 'snapshot_date']],
    test_df[['customer_id', 'snapshot_date']],
]).drop_duplicates(subset=['customer_id'])

events = events.merge(snapshots, on='customer_id', how='inner')
events = events[events['timestamp'] <= events['snapshot_date']].copy()

sess = events[events['event_type'] == 'session'][['customer_id', 'timestamp']].sort_values(['customer_id', 'timestamp'])
sess['prev_time'] = sess.groupby('customer_id')['timestamp'].shift(1)
sess['gap_days'] = (sess['timestamp'] - sess['prev_time']).dt.total_seconds() / 86400.0

sess['is_gt_7'] = (sess['gap_days'] > 7).astype(np.int16)
sess['is_gt_14'] = (sess['gap_days'] > 14).astype(np.int16)
sess['is_gt_30'] = (sess['gap_days'] > 30).astype(np.int16)

gap_stats = sess.groupby('customer_id').agg(
    longest_session_gap=('gap_days', 'max'),
    n_session_gaps_gt_7=('is_gt_7', 'sum'),
    n_session_gaps_gt_14=('is_gt_14', 'sum'),
    n_session_gaps_gt_30=('is_gt_30', 'sum')
).reset_index()

# Merge into train_df and test_df
train_df = train_df.merge(gap_stats, on='customer_id', how='left')
test_df = test_df.merge(gap_stats, on='customer_id', how='left')

# For customers with < 2 sessions, gap_days is NaN
# Fill longest_session_gap with account_age_days
for df in [train_df, test_df]:
    df['longest_session_gap'] = df['longest_session_gap'].fillna(df['account_age_days'])
    df['n_session_gaps_gt_7'] = df['n_session_gaps_gt_7'].fillna(0)
    df['n_session_gaps_gt_14'] = df['n_session_gaps_gt_14'].fillna(0)
    df['n_session_gaps_gt_30'] = df['n_session_gaps_gt_30'].fillna(0)

print("Session gap features computed successfully.")

# Step 3: Compute candidate targeted features
print("\n--- Step 3: Computing new targeted features ---")
eps = 1e-6

for df in [train_df, test_df]:
    # 1. Recency features already exist:
    # days_since_last_session, days_since_last_support_ticket, days_since_last_payment_failed, days_since_last_plan_change
    
    # 2. Session duration trends
    # sess_dur_mean_7d, 14d, 30d, 60d, 90d
    df['dur_ratio_7_30'] = df['sess_dur_mean_7d'] / (df['sess_dur_mean_30d'] + eps)
    df['dur_ratio_14_30'] = df['sess_dur_mean_14d'] / (df['sess_dur_mean_30d'] + eps)
    df['dur_ratio_60_90'] = df['sess_dur_mean_60d'] / (df['sess_dur_mean_90d'] + eps)
    df['dur_ratio_30_90'] = df['sess_dur_mean_30d'] / (df['sess_dur_mean_90d'] + eps)
    df['dur_diff_30_hist'] = df['sess_dur_mean_30d'] - df['sess_dur_mean']
    df['dur_ratio_30_hist'] = df['sess_dur_mean_30d'] / (df['sess_dur_mean'] + eps)
    df['dur_diff_7_30'] = df['sess_dur_mean_7d'] - df['sess_dur_mean_30d']

    # 3. Per-event-type activity trends
    # plan changes in windows (upgrade + downgrade)
    df['plan_change_7d'] = df['upgrade_7d'] + df['downgrade_7d']
    df['plan_change_30d'] = df['upgrade_30d'] + df['downgrade_30d']
    df['plan_change_60d'] = df['upgrade_60d'] + df['downgrade_60d']
    
    df['support_ratio_7_30'] = df['support_7d'] / (df['support_30d'] + eps)
    df['support_ratio_30_60'] = df['support_30d'] / (df['support_60d'] + eps)
    df['support_delta_7_30'] = df['support_30d'] - df['support_7d']
    df['support_delta_30_60'] = df['support_60d'] - df['support_30d']
    
    df['pf_ratio_7_30'] = df['pf_7d'] / (df['pf_30d'] + eps)
    df['pf_ratio_30_60'] = df['pf_30d'] / (df['pf_60d'] + eps)
    df['pf_delta_7_30'] = df['pf_30d'] - df['pf_7d']
    df['pf_delta_30_60'] = df['pf_60d'] - df['pf_30d']
    
    df['plan_change_ratio_30_60'] = df['plan_change_30d'] / (df['plan_change_60d'] + eps)
    df['plan_change_delta_30_60'] = df['plan_change_60d'] - df['plan_change_30d']

    # 4. Inactivity behavior
    df['fraction_inactive_lifetime'] = (df['account_age_days'] - df['n_distinct_dates']).clip(lower=0) / df['account_age_days'].clip(lower=1)

    # 5. Targeted interactions
    is_monthly = (df['billing_cycle'] == 'monthly').astype(int)
    is_basic = (df['plan'] == 'basic').astype(int)
    is_premium = (df['plan'] == 'premium').astype(int)
    
    df['tenure_x_plan_premium'] = df['account_age_days'] * is_premium
    df['recency_x_billing'] = df['days_since_last_session'] * is_monthly
    df['recency_event_x_billing'] = df['days_since_last_event'] * is_monthly
    df['pf_recency_x_billing'] = df['days_since_last_payment_failed'] * is_monthly
    df['pf_60d_x_billing'] = df['pf_60d'] * is_monthly
    df['support_60d_x_plan_basic'] = df['support_60d'] * is_basic
    df['support_count_x_plan_basic'] = df['support_count'] * is_basic

# Save candidate full data to scratch for quick iterations
train_df.to_parquet('scratch_train_v4_candidates.parquet')
test_df.to_parquet('scratch_test_v4_candidates.parquet')
print("Saved candidate data.")
