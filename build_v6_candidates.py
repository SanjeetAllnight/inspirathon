import pandas as pd
import numpy as np
import json
import time
import os
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 75)
print("BUILDING AND EVALUATING V6 CANDIDATE FEATURE BLOCKS")
print("=" * 75)

t0 = time.time()

# 1. Load V5 baseline features
train_df = pd.read_parquet('v5/train_features_v5.parquet')
test_df = pd.read_parquet('v5/test_features_v5.parquet')

with open('v5/v5_results.json') as f:
    v5_res = json.load(f)

v5_features = v5_res['v5_features']
best_v5_hp = v5_res['best_v5_hp']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
cat_idx = [v5_features.index(c) for c in cat_names if c in v5_features]

print(f"Loaded V5 features: {len(v5_features)} features.")
print(f"V5 baseline CV mean: {v5_res['v5_single_cb_mean_cv']:.5f}")
print(f"V5 folds: {[round(x, 5) for x in v5_res['v5_single_cb_folds']]}")

# 2. Load competition raw CSVs to compute candidates
print("\nLoading raw competition data for candidate extraction...")
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])
events_raw = pd.read_csv('events.csv', parse_dates=['timestamp'])

# Snapshots mapping
snapshots = pd.concat([
    train_raw[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']],
    test_raw[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']]
]).drop_duplicates(subset=['customer_id'])

# Filter events: timestamp <= snapshot_date
events = events_raw.merge(snapshots[['customer_id', 'snapshot_date']], on='customer_id', how='inner')
events = events[events['timestamp'] <= events['snapshot_date']].copy()
events['days_before'] = (events['snapshot_date'] - events['timestamp']).dt.total_seconds() / 86400.0
events['event_date'] = events['timestamp'].dt.date

events['is_session'] = (events['event_type'] == 'session').astype(np.int8)
events['is_support'] = (events['event_type'] == 'support_ticket').astype(np.int8)
events['is_payment_failed'] = (events['event_type'] == 'payment_failed').astype(np.int8)
events['is_plan_change'] = (events['event_type'] == 'plan_change').astype(np.int8)
events['is_upgrade'] = ((events['event_type'] == 'plan_change') & (events['value'] > 0)).astype(np.int8)
events['is_downgrade'] = ((events['event_type'] == 'plan_change') & (events['value'] < 0)).astype(np.int8)
events['session_dur'] = np.where(events['is_session'] == 1, events['value'], np.nan)

# ---------------------------------------------------------------------------
# BLOCK 4: EXTENDED RECENCY PROGRESSION (k=15, 20)
# ---------------------------------------------------------------------------
print("\nExtracting Block 4: Extended Recency Progression (k=15, 20)...")
sess = events[events['is_session'] == 1][['customer_id', 'days_before']].sort_values(['customer_id', 'days_before'])
sess['rank'] = sess.groupby('customer_id').cumcount() + 1

b4_features = []
df_b4 = pd.DataFrame(index=snapshots['customer_id'].unique())
for k in [15, 20]:
    col = f'days_since_{k}th_last_session'
    kth = sess[sess['rank'] == k].set_index('customer_id')['days_before'].rename(col)
    df_b4 = df_b4.join(kth, how='left')
    b4_features.append(col)

# ---------------------------------------------------------------------------
# BLOCK 3: EVENT COMPOSITION (30, 60, 90d)
# ---------------------------------------------------------------------------
print("Extracting Block 3: Event Composition (30, 60, 90d)...")
eps = 1e-6
# Base window counts from events
b3_cols = {}
for w in [30, 60, 90]:
    w_events = events[events['days_before'] <= w]
    wg = w_events.groupby('customer_id')
    wagg = wg.agg(
        evts = ('event_type', 'count'),
        sess = ('is_session', 'sum'),
        supp = ('is_support', 'sum'),
        pf   = ('is_payment_failed', 'sum'),
        upg  = ('is_upgrade', 'sum'),
        dwn  = ('is_downgrade', 'sum')
    )
    pc = wagg['upg'] + wagg['dwn']
    
    b3_cols[f'session_frac_{w}d'] = wagg['sess'] / (wagg['evts'] + eps)
    b3_cols[f'support_frac_{w}d'] = wagg['supp'] / (wagg['evts'] + eps)
    b3_cols[f'pf_frac_{w}d']      = wagg['pf'] / (wagg['evts'] + eps)
    b3_cols[f'pc_frac_{w}d']      = pc / (wagg['evts'] + eps)
    b3_cols[f'supp_to_pf_{w}d']   = (wagg['supp'] + eps) / (wagg['pf'] + eps)
    b3_cols[f'supp_to_sess_{w}d'] = (wagg['supp'] + eps) / (wagg['sess'] + eps)
    b3_cols[f'pf_to_sess_{w}d']   = (wagg['pf'] + eps) / (wagg['sess'] + eps)

df_b3 = pd.DataFrame(b3_cols, index=snapshots['customer_id'].unique())
df_b3 = df_b3.fillna(0)
b3_features = list(df_b3.columns)

# ---------------------------------------------------------------------------
# BLOCK 1 & 2: WEEKLY ACTIVITY PROFILE & TRAJECTORY FEATURES (W1..W8)
# ---------------------------------------------------------------------------
print("Extracting Block 1 & 2: Weekly Activity Profile & Trajectories (8 Weeks)...")
events['week_idx'] = np.floor(events['days_before'] / 7.0).astype(int) + 1
events_8w = events[(events['week_idx'] >= 1) & (events['week_idx'] <= 8)]

# Group by customer_id and week_idx
w_agg = events_8w.groupby(['customer_id', 'week_idx']).agg(
    sess_count = ('is_session', 'sum'),
    active_days = ('event_date', 'nunique'),
    sess_dur = ('session_dur', 'sum'),
    support = ('is_support', 'sum'),
    pf = ('is_payment_failed', 'sum'),
    pc = ('is_plan_change', 'sum')
).unstack(level='week_idx')

# Create complete matrix for all customers
all_custs = snapshots['customer_id'].unique()
w_agg = w_agg.reindex(all_custs).fillna(0)

# Weekly profile features (Block 1)
df_b1 = pd.DataFrame(index=all_custs)
b1_features = []
for metric in ['sess_count', 'active_days', 'sess_dur', 'support', 'pf', 'pc']:
    for w in range(1, 9):
        col_name = f'{metric}_w{w}'
        if (metric, w) in w_agg.columns:
            df_b1[col_name] = w_agg[(metric, w)].values
        else:
            df_b1[col_name] = 0.0
        b1_features.append(col_name)

# Trajectory features (Block 2)
df_b2 = pd.DataFrame(index=all_custs)

# Extract session arrays (w1 is most recent, w8 is earliest)
sess_matrix = np.array([df_b1[f'sess_count_w{w}'].values for w in range(1, 9)]).T # Shape: (N, 8)
activedays_matrix = np.array([df_b1[f'active_days_w{w}'].values for w in range(1, 9)]).T # Shape: (N, 8)
sessdur_matrix = np.array([df_b1[f'sess_dur_w{w}'].values for w in range(1, 9)]).T # Shape: (N, 8)

# Linear slope across 8 weeks (chronological: w8, w7, ..., w1)
# t = 1..8 corresponding to [w8, w7, w6, w5, w4, w3, w2, w1]
# c_i = (i - 4.5) / 42.0
weights_8w = np.array([(i - 4.5) / 42.0 for i in range(1, 9)]) # for t=1..8
# Reorder matrix chronologically: column 0 is w8, column 7 is w1
sess_chrono_8w = sess_matrix[:, ::-1] # w8..w1
dur_chrono_8w = sessdur_matrix[:, ::-1] # w8..w1
activedays_chrono_8w = activedays_matrix[:, ::-1] # w8..w1

df_b2['session_slope_8w'] = np.dot(sess_chrono_8w, weights_8w)
df_b2['dur_slope_8w'] = np.dot(dur_chrono_8w, weights_8w)
df_b2['activedays_slope_8w'] = np.dot(activedays_chrono_8w, weights_8w)

# Recent 4 weeks slope (w4..w1)
weights_4w = np.array([(i - 2.5) / 5.0 for i in range(1, 5)])
sess_chrono_4w = sess_matrix[:, 3::-1] # w4, w3, w2, w1
df_b2['session_slope_4w'] = np.dot(sess_chrono_4w, weights_4w)

# Recent-vs-earliest ratio (w1+w2)/(w7+w8 + eps)
df_b2['recent_vs_earliest_sess_ratio'] = (sess_matrix[:, 0] + sess_matrix[:, 1] + eps) / (sess_matrix[:, 6] + sess_matrix[:, 7] + eps)
df_b2['recent_vs_earliest_dur_ratio'] = (sessdur_matrix[:, 0] + sessdur_matrix[:, 1] + eps) / (sessdur_matrix[:, 6] + sessdur_matrix[:, 7] + eps)

# Extreme & dispersion
df_b2['max_weekly_sess'] = np.max(sess_matrix, axis=1)
df_b2['min_weekly_sess'] = np.min(sess_matrix, axis=1)
df_b2['std_weekly_sess'] = np.std(sess_matrix, axis=1)
sess_mean = np.mean(sess_matrix, axis=1)
df_b2['cv_weekly_sess'] = df_b2['std_weekly_sess'] / (sess_mean + eps)

# Zero-activity weeks count
df_b2['zero_sess_weeks_8w'] = np.sum(sess_matrix == 0, axis=1)
df_b2['zero_activeday_weeks_8w'] = np.sum(activedays_matrix == 0, axis=1)

# Consecutive declining weeks ending at week 1
# Check s_w1 < s_w2, s_w2 < s_w3, etc.
c_decl = np.zeros(len(all_custs), dtype=int)
for i in range(len(all_custs)):
    row = sess_matrix[i]
    cnt = 0
    for w in range(0, 7):
        if row[w] < row[w+1]:
            cnt += 1
        else:
            break
    c_decl[i] = cnt
df_b2['consec_declining_weeks'] = c_decl

# Last active week position (1 = week 1, 2 = week 2, ..., 8 = week 8, 9 = never active in 8 weeks)
last_act = np.full(len(all_custs), 9, dtype=int)
for w in range(0, 8):
    mask = (sess_matrix[:, w] > 0) & (last_act == 9)
    last_act[mask] = w + 1
df_b2['last_active_week_pos'] = last_act

b2_features = list(df_b2.columns)

print(f"Extracted features:")
print(f"  Block 4 (Extended recency) : {len(b4_features)} features: {b4_features}")
print(f"  Block 3 (Event composition): {len(b3_features)} features")
print(f"  Block 1 (Weekly profile)   : {len(b1_features)} features")
print(f"  Block 2 (Trajectories)     : {len(b2_features)} features: {b2_features}")

# Save scratch candidate parquets
df_b4.reset_index().rename(columns={'index': 'customer_id'}).to_parquet('scratch_b4.parquet')
df_b3.reset_index().rename(columns={'index': 'customer_id'}).to_parquet('scratch_b3.parquet')
df_b1.reset_index().rename(columns={'index': 'customer_id'}).to_parquet('scratch_b1.parquet')
df_b2.reset_index().rename(columns={'index': 'customer_id'}).to_parquet('scratch_b2.parquet')

print(f"\nAll candidate feature extraction completed in {time.time() - t0:.1f}s.")
