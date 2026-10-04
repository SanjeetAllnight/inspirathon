import pandas as pd
import numpy as np
import time

print("--- Testing feature extraction for Experiments A, B, and C ---")
t0 = time.time()

train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])
events = pd.read_csv('events.csv', parse_dates=['timestamp'])

snapshots = pd.concat([
    train_raw[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']],
    test_raw[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']]
]).drop_duplicates(subset=['customer_id'])

# -------------------------------------------------------------
# EXPERIMENT A: Billing-Cycle Phase
# -------------------------------------------------------------
print("Building Experiment A features...")
tenure_days = (snapshots['snapshot_date'] - snapshots['signup_date']).dt.days

is_annual = snapshots['billing_cycle'] == 'annual'
is_monthly = snapshots['billing_cycle'] == 'monthly'

annual_cycle_pos = np.where(is_annual, (tenure_days % 365.25) / 365.25, -1.0)
annual_days_to_anniv = np.where(is_annual, 365.25 - (tenure_days % 365.25), -1.0)
annual_near_renewal = np.where(is_annual & (annual_days_to_anniv <= 30), 1, 0)
monthly_cycle_pos = np.where(is_monthly, (tenure_days % 30.44) / 30.44, -1.0)
near_first_monthly_renewal = np.where(is_monthly & (tenure_days <= 31), 1, 0)
near_first_annual_renewal = np.where(is_annual & (tenure_days <= 365.25), 1, 0)

exp_a_df = pd.DataFrame({
    'customer_id': snapshots['customer_id'],
    'annual_days_to_anniversary': annual_days_to_anniv,
    'annual_near_renewal': annual_near_renewal,
    'annual_cycle_position': annual_cycle_pos,
    'monthly_cycle_position': monthly_cycle_pos,
    'near_first_monthly_renewal': near_first_monthly_renewal,
    'near_first_annual_renewal': near_first_annual_renewal
})
print("Exp A features created:", exp_a_df.shape)

# -------------------------------------------------------------
# Preprocessing events
# -------------------------------------------------------------
print("Preprocessing events...")
events = events.merge(snapshots[['customer_id', 'snapshot_date', 'signup_date']], on='customer_id', how='inner')
events = events[events['timestamp'] <= events['snapshot_date']].copy()
events['days_before'] = (events['snapshot_date'] - events['timestamp']).dt.total_seconds() / 86400.0

min_event_time = events['timestamp'].min()
events_cust_ids = snapshots[['customer_id']].copy()

# -------------------------------------------------------------
# EXPERIMENT B: Sparse-Aware Activity
# -------------------------------------------------------------
print("Building Experiment B features...")
# 1. kth-last session
sess = events[events['event_type'] == 'session'].copy()
sess = sess.sort_values(['customer_id', 'days_before'], ascending=[True, True])
sess['rank'] = sess.groupby('customer_id').cumcount() + 1

k_feats = {}
for k in [2, 3, 5, 10]:
    k_sub = sess[sess['rank'] == k].set_index('customer_id')['days_before']
    k_feats[f'days_since_{k}th_last_session'] = k_sub

exp_b_df = pd.DataFrame(k_feats, index=events_cust_ids['customer_id']).reset_index()

# 2. Exposure-adjusted activity
avail_history = (snapshots['snapshot_date'] - min_event_time).dt.total_seconds() / 86400.0
exposure = np.minimum(tenure_days, avail_history).clip(lower=1.0)
exposure_map = pd.Series(exposure.values, index=snapshots['customer_id'])

tot_events_map = events.groupby('customer_id').size()
sess_count_map = sess.groupby('customer_id').size()
sess_dur_map = sess.groupby('customer_id')['value'].sum()

exp_b_df['events_per_exposure_day'] = exp_b_df['customer_id'].map(tot_events_map).fillna(0) / exp_b_df['customer_id'].map(exposure_map)
exp_b_df['sessions_per_exposure_day'] = exp_b_df['customer_id'].map(sess_count_map).fillna(0) / exp_b_df['customer_id'].map(exposure_map)
exp_b_df['dur_per_exposure_day'] = exp_b_df['customer_id'].map(sess_dur_map).fillna(0) / exp_b_df['customer_id'].map(exposure_map)

# 3. EWMA / decayed activity
for hl in [7, 21, 60]:
    lam = np.log(2.0) / hl
    sess[f'ewma_{hl}'] = np.exp(-sess['days_before'] * lam)
    events[f'ewma_{hl}'] = np.exp(-events['days_before'] * lam)
    
    s_ewma = sess.groupby('customer_id')[f'ewma_{hl}'].sum()
    e_ewma = events.groupby('customer_id')[f'ewma_{hl}'].sum()
    exp_b_df[f'session_ewma_{hl}'] = exp_b_df['customer_id'].map(s_ewma).fillna(0.0)
    exp_b_df[f'event_ewma_{hl}'] = exp_b_df['customer_id'].map(e_ewma).fillna(0.0)

# 4. Overdue behavior
sess_time_sort = sess.sort_values(['customer_id', 'timestamp'])
sess_time_sort['prev_time'] = sess_time_sort.groupby('customer_id')['timestamp'].shift(1)
sess_time_sort['gap_days'] = (sess_time_sort['timestamp'] - sess_time_sort['prev_time']).dt.total_seconds() / 86400.0

mean_gap = sess_time_sort.groupby('customer_id')['gap_days'].mean()
median_gap = sess_time_sort.groupby('customer_id')['gap_days'].median()
days_since_1st_sess = sess[sess['rank'] == 1].set_index('customer_id')['days_before']

exp_b_df['mean_session_gap'] = exp_b_df['customer_id'].map(mean_gap)
exp_b_df['median_session_gap'] = exp_b_df['customer_id'].map(median_gap)
exp_b_df['days_since_last_sess'] = exp_b_df['customer_id'].map(days_since_1st_sess)

exp_b_df['overdue_ratio_mean'] = (exp_b_df['days_since_last_sess'] / (exp_b_df['mean_session_gap'] + 0.1)).fillna(1.0).clip(0, 50)
exp_b_df['overdue_ratio_median'] = (exp_b_df['days_since_last_sess'] / (exp_b_df['median_session_gap'] + 0.1)).fillna(1.0).clip(0, 50)

print("Exp B features created:", exp_b_df.shape)

# -------------------------------------------------------------
# EXPERIMENT C: Event State / Order
# -------------------------------------------------------------
print("Building Experiment C features...")
# Last payment failed timestamp
pf_events = events[events['event_type'] == 'payment_failed']
last_pf = pf_events.groupby('customer_id')['timestamp'].max().rename('last_pf_time')

# Last support ticket timestamp
st_events = events[events['event_type'] == 'support_ticket']
last_st = st_events.groupby('customer_id')['timestamp'].max().rename('last_st_time')

# Last plan change value
pc_events = events[events['event_type'] == 'plan_change'].sort_values(['customer_id', 'timestamp'])
last_pc_val = pc_events.groupby('customer_id').last()['value'].rename('last_plan_change_direction')

sess_annot = sess[['customer_id', 'timestamp']].copy()
sess_annot = sess_annot.merge(last_pf, on='customer_id', how='left')
sess_annot = sess_annot.merge(last_st, on='customer_id', how='left')

sess_after_pf = sess_annot[sess_annot['timestamp'] > sess_annot['last_pf_time']].groupby('customer_id').size()
sess_after_st = sess_annot[sess_annot['timestamp'] > sess_annot['last_st_time']].groupby('customer_id').size()

exp_c_df = pd.DataFrame({'customer_id': snapshots['customer_id']})
exp_c_df['has_pf'] = exp_c_df['customer_id'].isin(last_pf.index).astype(int)
exp_c_df['has_st'] = exp_c_df['customer_id'].isin(last_st.index).astype(int)

exp_c_df['sessions_since_last_pf'] = exp_c_df['customer_id'].map(sess_after_pf).fillna(0)
exp_c_df['sessions_since_last_pf'] = np.where(exp_c_df['has_pf'] == 1, exp_c_df['sessions_since_last_pf'], -1)

exp_c_df['sessions_since_last_support'] = exp_c_df['customer_id'].map(sess_after_st).fillna(0)
exp_c_df['sessions_since_last_support'] = np.where(exp_c_df['has_st'] == 1, exp_c_df['sessions_since_last_support'], -1)

exp_c_df['last_plan_change_direction'] = exp_c_df['customer_id'].map(last_pc_val).fillna(0.0)
exp_c_df['support_followed_by_inactivity'] = np.where((exp_c_df['has_st'] == 1) & (exp_c_df['sessions_since_last_support'] == 0), 1, 0)
exp_c_df['pf_followed_by_inactivity'] = np.where((exp_c_df['has_pf'] == 1) & (exp_c_df['sessions_since_last_pf'] == 0), 1, 0)

# Drop temporary flags
exp_c_df = exp_c_df.drop(columns=['has_pf', 'has_st'])
print("Exp C features created:", exp_c_df.shape)

# Save to scratch
exp_a_df.to_parquet('scratch_exp_a.parquet')
exp_b_df.to_parquet('scratch_exp_b.parquet')
exp_c_df.to_parquet('scratch_exp_c.parquet')

print(f"All candidate features built and saved in {time.time() - t0:.1f}s")
