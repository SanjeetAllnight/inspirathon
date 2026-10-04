import pandas as pd
import numpy as np
import time
import os

print("=" * 80)
print("EXTRACTING V8 CANDIDATE FEATURE BLOCKS (A, B, C, D, E)")
print("=" * 80)
t0 = time.time()

# 1. Load train and test snapshots
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])
events_raw = pd.read_csv('events.csv', parse_dates=['timestamp'])

snapshots = pd.concat([
    train_raw[['customer_id', 'snapshot_date', 'signup_date']],
    test_raw[['customer_id', 'snapshot_date', 'signup_date']]
]).drop_duplicates(subset=['customer_id'])

all_custs = snapshots['customer_id'].unique()
cust_index = pd.Index(all_custs, name='customer_id')

# Filter events: timestamp <= snapshot_date
events = events_raw.merge(snapshots[['customer_id', 'snapshot_date']], on='customer_id', how='inner')
events = events[events['timestamp'] <= events['snapshot_date']].copy()
events['days_before'] = (events['snapshot_date'] - events['timestamp']).dt.total_seconds() / 86400.0
events['event_date'] = events['timestamp'].dt.date
events['is_session'] = (events['event_type'] == 'session').astype(np.int8)
events['is_support'] = (events['event_type'] == 'support_ticket').astype(np.int8)
events['is_payment_failed'] = (events['event_type'] == 'payment_failed').astype(np.int8)
events['is_plan_change'] = (events['event_type'] == 'plan_change').astype(np.int8)
events['session_dur'] = np.where(events['is_session'] == 1, events['value'], np.nan)

# Sort events chronologically per customer for sequence & cadence features
events = events.sort_values(['customer_id', 'timestamp'])

print(f"Events loaded & filtered in {time.time() - t0:.1f}s.")

# ===========================================================================
# BLOCK A: VERY-RECENT ACTIVITY (1d, 3d, 5d, 10d, 21d)
# ===========================================================================
print("\nExtracting Block A: Very-Recent Activity...")
t_a = time.time()
df_a = pd.DataFrame(index=cust_index)

windows_a = [1, 3, 5, 10, 21]
for w in windows_a:
    sub = events[events['days_before'] <= w]
    g = sub.groupby('customer_id')
    agg = g.agg(
        sess=('is_session', 'sum'),
        evts=('event_type', 'count'),
        act=('event_date', 'nunique'),
        dur=('session_dur', 'sum'),
        pf=('is_payment_failed', 'sum'),
        supp=('is_support', 'sum')
    )
    df_a[f'sess_{w}d'] = agg['sess']
    df_a[f'events_{w}d'] = agg['evts']
    df_a[f'activedays_{w}d'] = agg['act']
    df_a[f'sess_dur_{w}d'] = agg['dur']
    df_a[f'pf_{w}d'] = agg['pf']
    df_a[f'supp_{w}d'] = agg['supp']
    df_a[f'no_session_{w}d'] = ((df_a[f'sess_{w}d'].fillna(0)) == 0).astype(np.int8)

df_a = df_a.fillna(0)
for w in windows_a:
    df_a[f'no_session_{w}d'] = (df_a[f'sess_{w}d'] == 0).astype(np.int8)

print(f"  Block A: {df_a.shape[1]} features extracted in {time.time() - t_a:.1f}s.")
df_a.reset_index().to_parquet('scratch_v8_block_a.parquet')

# ===========================================================================
# BLOCK B: EVENT SEQUENCE
# ===========================================================================
print("\nExtracting Block B: Event Sequence...")
t_b = time.time()
df_b = pd.DataFrame(index=cust_index)

# Short event codes: s=session, t=support, p=pf, c=plan_change
type_map = {
    'session': 's',
    'support_ticket': 't',
    'payment_failed': 'p',
    'plan_change': 'c'
}
events['code'] = events['event_type'].map(type_map).fillna('?')

# Last event and second last event
last_events = events.groupby('customer_id').tail(5)

# Extract last 1, 2, 3, 5 events as sequences
def extract_seq_features(df_tail):
    # Group by customer and aggregate codes
    seqs = df_tail.groupby('customer_id')['code'].apply(lambda x: ''.join(x.values))
    last1 = df_tail.groupby('customer_id')['event_type'].last()
    
    # Second last
    def get_second_last(s):
        vals = s.values
        return vals[-2] if len(vals) >= 2 else 'None'
    second_last = df_tail.groupby('customer_id')['event_type'].apply(get_second_last)
    
    return seqs, last1, second_last

seqs, last1, second_last = extract_seq_features(last_events)

df_b['last_event_type'] = last1
df_b['second_last_event_type'] = second_last
df_b['event_seq_last3'] = seqs.apply(lambda x: x[-3:] if len(x) >= 3 else x)
df_b['event_seq_last5'] = seqs.apply(lambda x: x[-5:] if len(x) >= 5 else x)

# Categorical columns
df_b['last_event_type'] = df_b['last_event_type'].fillna('None').astype('category')
df_b['second_last_event_type'] = df_b['second_last_event_type'].fillna('None').astype('category')
df_b['event_seq_last3'] = df_b['event_seq_last3'].fillna('None').astype('category')
df_b['event_seq_last5'] = df_b['event_seq_last5'].fillna('None').astype('category')

# Explicit boolean flags
df_b['last_is_support'] = (df_b['last_event_type'] == 'support_ticket').astype(np.int8)
df_b['last_is_payment_failed'] = (df_b['last_event_type'] == 'payment_failed').astype(np.int8)
df_b['last_is_plan_change'] = (df_b['last_event_type'] == 'plan_change').astype(np.int8)
df_b['second_last_is_support'] = (df_b['second_last_event_type'] == 'support_ticket').astype(np.int8)
df_b['second_last_is_pf'] = (df_b['second_last_event_type'] == 'payment_failed').astype(np.int8)

# Sequence transitions in last 5 events
df_b['has_trans_sess_to_support'] = seqs.apply(lambda s: int('st' in s if isinstance(s, str) else 0)).astype(np.int8)
df_b['has_trans_support_to_sess'] = seqs.apply(lambda s: int('ts' in s if isinstance(s, str) else 0)).astype(np.int8)
df_b['has_trans_pf_to_sess'] = seqs.apply(lambda s: int('ps' in s if isinstance(s, str) else 0)).astype(np.int8)
df_b['has_trans_pc_to_sess'] = seqs.apply(lambda s: int('cs' in s if isinstance(s, str) else 0)).astype(np.int8)
df_b['has_trans_sess_to_pf'] = seqs.apply(lambda s: int('sp' in s if isinstance(s, str) else 0)).astype(np.int8)

print(f"  Block B: {df_b.shape[1]} features extracted in {time.time() - t_b:.1f}s.")
df_b.reset_index().to_parquet('scratch_v8_block_b.parquet')

# ===========================================================================
# BLOCK C: EVENT TIMING
# ===========================================================================
print("\nExtracting Block C: Event Timing...")
t_c = time.time()
df_c = pd.DataFrame(index=cust_index)

sess_events = events[events['is_session'] == 1].copy()
sess_events['hour'] = sess_events['timestamp'].dt.hour
sess_events['dow'] = sess_events['timestamp'].dt.dayofweek
sess_events['is_weekend'] = (sess_events['dow'] >= 5).astype(np.int8)
# Hour buckets: 0: Night (0-6), 1: Morning (6-12), 2: Afternoon (12-18), 3: Evening (18-24)
sess_events['hour_bucket'] = pd.cut(sess_events['hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)

# Timing of last session
last_sess = sess_events.groupby('customer_id').last()
df_c['last_sess_hour'] = last_sess['hour'].fillna(-1).astype(int)
df_c['last_sess_dow'] = last_sess['dow'].fillna(-1).astype(int)
df_c['last_sess_is_weekend'] = last_sess['is_weekend'].fillna(0).astype(np.int8)
df_c['last_sess_hour_bucket'] = last_sess['hour_bucket'].fillna(-1).astype(int)

# Timing diversity in last 30d
sess_30d = sess_events[sess_events['days_before'] <= 30]
g_30d = sess_30d.groupby('customer_id')
df_c['n_active_weekdays_30d'] = g_30d['dow'].nunique().reindex(cust_index).fillna(0).astype(np.int8)
df_c['n_active_hour_buckets_30d'] = g_30d['hour_bucket'].nunique().reindex(cust_index).fillna(0).astype(np.int8)
df_c['weekend_session_fraction_30d'] = (g_30d['is_weekend'].mean().reindex(cust_index)).fillna(0).astype(np.float32)

print(f"  Block C: {df_c.shape[1]} features extracted in {time.time() - t_c:.1f}s.")
df_c.reset_index().to_parquet('scratch_v8_block_c.parquet')

# ===========================================================================
# BLOCK D: SESSION CADENCE
# ===========================================================================
print("\nExtracting Block D: Session Cadence...")
t_d = time.time()
df_d = pd.DataFrame(index=cust_index)

# Inter-session gaps (in days)
sess_events['prev_timestamp'] = sess_events.groupby('customer_id')['timestamp'].shift(1)
sess_events['gap_days'] = (sess_events['timestamp'] - sess_events['prev_timestamp']).dt.total_seconds() / 86400.0

gap_events = sess_events.dropna(subset=['gap_days'])
g_gaps = gap_events.groupby('customer_id')['gap_days']

df_d['median_gap_days'] = g_gaps.median().reindex(cust_index).fillna(999.0)
df_d['mean_gap_days'] = g_gaps.mean().reindex(cust_index).fillna(999.0)
df_d['std_gap_days'] = g_gaps.std().reindex(cust_index).fillna(0.0)

# Most recent gap (gap before last session)
recent_gap = gap_events.groupby('customer_id')['gap_days'].last().reindex(cust_index).fillna(999.0)
df_d['recent_gap_days'] = recent_gap
df_d['recent_vs_median_gap_ratio'] = (df_d['recent_gap_days'] + 1e-4) / (df_d['median_gap_days'] + 1e-4)

# Gaps within recent 30d and 60d
gaps_30d = gap_events[gap_events['days_before'] <= 30].groupby('customer_id')['gap_days']
df_d['median_gap_30d'] = gaps_30d.median().reindex(cust_index).fillna(df_d['median_gap_days'])

gaps_60d = gap_events[gap_events['days_before'] <= 60].groupby('customer_id')['gap_days']
df_d['median_gap_60d'] = gaps_60d.median().reindex(cust_index).fillna(df_d['median_gap_days'])

# Number of sessions to cover last 30d and 60d
df_d['sess_coverage_density_30d'] = (sess_events[sess_events['days_before'] <= 30].groupby('customer_id').size().reindex(cust_index).fillna(0)) / 30.0
df_d['sess_coverage_density_60d'] = (sess_events[sess_events['days_before'] <= 60].groupby('customer_id').size().reindex(cust_index).fillna(0)) / 60.0

print(f"  Block D: {df_d.shape[1]} features extracted in {time.time() - t_d:.1f}s.")
df_d.reset_index().to_parquet('scratch_v8_block_d.parquet')

# ===========================================================================
# BLOCK E: MIXED EVENT STATES
# ===========================================================================
print("\nExtracting Block E: Mixed Event States...")
t_e = time.time()
df_e = pd.DataFrame(index=cust_index)

# Recencies of each event type
for etype in ['session', 'support_ticket', 'payment_failed', 'plan_change']:
    sub_e = events[events['event_type'] == etype]
    rec = sub_e.groupby('customer_id')['days_before'].min().reindex(cust_index).fillna(999.0)
    df_e[f'rec_{etype}'] = rec

# State indicators:
# 1. Unresolved payment failure: last PF happened AFTER last session (days_before_pf < days_before_session)
df_e['unresolved_payment_failure'] = ((df_e['rec_payment_failed'] < df_e['rec_session']) & (df_e['rec_payment_failed'] < 90)).astype(np.int8)

# 2. Unresolved support ticket: last support ticket happened AFTER last session
df_e['unresolved_support_ticket'] = ((df_e['rec_support_ticket'] < df_e['rec_session']) & (df_e['rec_support_ticket'] < 90)).astype(np.int8)

# 3. Time elapsed between last session and last support ticket
df_e['days_diff_support_session'] = (df_e['rec_session'] - df_e['rec_support_ticket']).clip(-180, 180)

# 4. Time elapsed between last session and last payment failure
df_e['days_diff_pf_session'] = (df_e['rec_session'] - df_e['rec_payment_failed']).clip(-180, 180)

# 5. Critical risk state: payment failure within 14 days AND no session for 7+ days
df_e['pf_recent_and_inactive_7d'] = ((df_e['rec_payment_failed'] <= 14) & (df_e['rec_session'] >= 7)).astype(np.int8)
df_e['support_recent_and_inactive_7d'] = ((df_e['rec_support_ticket'] <= 14) & (df_e['rec_session'] >= 7)).astype(np.int8)

# 6. Critical risk state: payment failure within 30 days AND no session for 14+ days
df_e['pf_recent_and_inactive_14d'] = ((df_e['rec_payment_failed'] <= 30) & (df_e['rec_session'] >= 14)).astype(np.int8)

# Drop intermediate raw recencies to avoid duplicates with V6
df_e = df_e.drop(columns=[f'rec_{etype}' for etype in ['session', 'support_ticket', 'payment_failed', 'plan_change']])

print(f"  Block E: {df_e.shape[1]} features extracted in {time.time() - t_e:.1f}s.")
df_e.reset_index().to_parquet('scratch_v8_block_e.parquet')

print(f"\nAll 5 candidate blocks successfully extracted in {time.time() - t0:.1f}s.")
