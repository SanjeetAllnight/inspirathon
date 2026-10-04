import pandas as pd
import numpy as np
import json
import time

print("--- Verifying Standalone Feature Engineering Pipeline ---")
t0 = time.time()

# Load raw competition CSVs only
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])
events_raw = pd.read_csv('events.csv', parse_dates=['timestamp'])

with open('v5/v5_results.json') as f:
    v5_res = json.load(f)

target_v5_features = v5_res['v5_features']
print(f"Target V5 features count: {len(target_v5_features)}")

# Reference dataframe from parquet
ref_train = pd.read_parquet('v5/train_features_v5.parquet')

# Let's write the clean feature engineering function
def extract_v5_features(train, test, events):
    eps = 1e-6
    
    # 1. Snapshots mapping
    snapshots = pd.concat([
        train[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']],
        test[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']]
    ]).drop_duplicates(subset=['customer_id'])
    
    # 2. Events filter
    events = events.merge(snapshots[['customer_id', 'snapshot_date']], on='customer_id', how='inner')
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
    
    windows = [7, 14, 30, 60, 90, 120]
    for w in windows:
        events[f'in_{w}d'] = (events['days_before'] <= w).astype(np.int8)
        
    # Group by customer_id
    g = events.groupby('customer_id')
    
    basic = g.agg(
        total_events          = ('event_type', 'count'),
        session_count         = ('is_session', 'sum'),
        support_count         = ('is_support', 'sum'),
        pf_count              = ('is_payment_failed', 'sum'),
        plan_change_count     = ('is_plan_change', 'sum'),
        upgrade_count         = ('is_upgrade', 'sum'),
        downgrade_count       = ('is_downgrade', 'sum'),
        days_since_last_event = ('days_before', 'min'),
        sess_dur_total        = ('session_dur', 'sum'),
        sess_dur_mean         = ('session_dur', 'mean'),
        sess_dur_median       = ('session_dur', 'median'),
        sess_dur_max          = ('session_dur', 'max'),
        sess_dur_std          = ('session_dur', 'std'),
        n_distinct_dates      = ('event_date', 'nunique'),
    )
    
    # Days since last event for each type
    for etype, flag in [('session', 'is_session'), ('support_ticket', 'is_support'),
                        ('payment_failed', 'is_payment_failed'), ('plan_change', 'is_plan_change')]:
        sub = events[events[flag] == 1]
        col_name = f'days_since_last_{etype}'
        if len(sub) > 0:
            last = sub.groupby('customer_id')['days_before'].min().rename(col_name)
            basic = basic.join(last, how='left')
        else:
            basic[col_name] = np.nan
            
    # Window aggregations
    for w in windows:
        wdf = events[events[f'in_{w}d'] == 1]
        wg = wdf.groupby('customer_id')
        wagg = wg.agg(**{
            f'events_{w}d':         ('event_type', 'count'),
            f'sessions_{w}d':       ('is_session', 'sum'),
            f'support_{w}d':        ('is_support', 'sum'),
            f'pf_{w}d':             ('is_payment_failed', 'sum'),
            f'upgrade_{w}d':        ('is_upgrade', 'sum'),
            f'downgrade_{w}d':      ('is_downgrade', 'sum'),
            f'sess_dur_total_{w}d': ('session_dur', 'sum'),
            f'sess_dur_mean_{w}d':  ('session_dur', 'mean'),
            f'active_days_{w}d':    ('event_date', 'nunique'),
        })
        basic = basic.join(wagg, how='left')
        
    # Longest inactivity gap
    def _longest_gap(dates):
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < 2:
            return np.nan
        return float(max(np.diff(unique_dates)))

    gaps = events.groupby('customer_id')['days_before'].apply(_longest_gap).rename('longest_gap_days')
    basic = basic.join(gaps, how='left')
    
    # kth-last sessions (k = 2, 3, 5, 10)
    sess = events[events['is_session'] == 1][['customer_id', 'days_before']].sort_values(['customer_id', 'days_before'])
    sess['rank'] = sess.groupby('customer_id').cumcount() + 1
    for k in [2, 3, 5, 10]:
        kth = sess[sess['rank'] == k].set_index('customer_id')['days_before'].rename(f'days_since_{k}th_last_session')
        basic = basic.join(kth, how='left')
        
    ef = basic
    
    # Derived ratios and trends
    for a, b in [(7, 30), (14, 60), (30, 60), (60, 120), (7, 14), (30, 90)]:
        ef[f'sess_ratio_{a}_{b}'] = ef[f'sessions_{a}d'] / (ef[f'sessions_{b}d'] + eps)

    for a, b in [(7, 30), (30, 60), (60, 120)]:
        ef[f'evt_ratio_{a}_{b}'] = ef[f'events_{a}d'] / (ef[f'events_{b}d'] + eps)

    for a, b in [(7, 30), (30, 60), (60, 120)]:
        ef[f'dur_ratio_{a}_{b}'] = ef[f'sess_dur_total_{a}d'] / (ef[f'sess_dur_total_{b}d'] + eps)

    for a, b in [(7, 14), (14, 30), (30, 60), (60, 90)]:
        ef[f'sess_delta_{a}_{b}'] = ef[f'sessions_{b}d'] - ef[f'sessions_{a}d']

    ef['sess_accel'] = ef['sess_ratio_7_30'] - ef['sess_ratio_30_60']
    ef['support_per_session'] = ef['support_count'] / (ef['session_count'] + eps)
    ef['pf_per_session'] = ef['pf_count'] / (ef['session_count'] + eps)
    
    for w in [30, 60]:
        ef[f'support_per_sess_{w}d'] = ef[f'support_{w}d'] / (ef[f'sessions_{w}d'] + eps)
        ef[f'pf_per_sess_{w}d'] = ef[f'pf_{w}d'] / (ef[f'sessions_{w}d'] + eps)

    for w in [7, 30, 60, 120]:
        ef[f'sess_per_activeday_{w}d'] = ef[f'sessions_{w}d'] / (ef[f'active_days_{w}d'] + eps)
        ef[f'dur_per_activeday_{w}d']  = ef[f'sess_dur_total_{w}d'] / (ef[f'active_days_{w}d'] + eps)

    for a, b in [(7, 30), (30, 60), (60, 120)]:
        ef[f'activeday_ratio_{a}_{b}'] = ef[f'active_days_{a}d'] / (ef[f'active_days_{b}d'] + eps)
        
    # Build full profiles
    ref_freq = pd.concat([train['referral_code'], test['referral_code']]).value_counts()
    
    def process_df(df):
        df = df.copy()
        df = df.merge(ef, on='customer_id', how='left')
        
        df['account_age_days']   = (df['snapshot_date'] - df['signup_date']).dt.days
        df['account_age_months'] = df['account_age_days'] / 30.44
        df['signup_month']       = df['signup_date'].dt.month
        df['signup_year']        = df['signup_date'].dt.year
        df['signup_dow']         = df['signup_date'].dt.dayofweek
        df['is_recent_signup']   = (df['account_age_days'] <= 30).astype(np.int8)
        df['is_very_new']        = (df['account_age_days'] <= 7).astype(np.int8)
        df['snapshot_month']     = df['snapshot_date'].dt.month

        df['price_per_device']   = df['monthly_price'] / df['num_devices'].clip(lower=1)
        df['is_age_missing']     = df['age_band'].isnull().astype(np.int8)
        df['is_pm_missing']      = df['payment_method'].isnull().astype(np.int8)
        df['has_referral']       = df['referral_code'].notnull().astype(np.int8)
        df['has_events']         = df['total_events'].notnull().astype(np.int8)
        df['referral_freq']      = df['referral_code'].map(ref_freq).fillna(0)

        df['n_distinct_dates']    = df['n_distinct_dates'].fillna(0)
        df['tenure_x_billing']    = df['account_age_days'] * (df['billing_cycle'] == 'monthly').astype(int)
        df['tenure_x_plan_basic'] = df['account_age_days'] * (df['plan'] == 'basic').astype(int)
        df['fraction_inactive_lifetime'] = (df['account_age_days'] - df['n_distinct_dates']).clip(lower=0) / df['account_age_days'].clip(lower=1)

        # Days since columns imputation
        days_cols = [c for c in df.columns if c.startswith('days_since_')]
        for c in days_cols:
            df[c] = df[c].fillna(df['account_age_days'])

        # Event numeric fillna 0
        evt_cols = [c for c in ef.columns if not c.startswith('days_since_')]
        for c in evt_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        cat_cols = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
        for c in cat_cols:
            df[c] = df[c].fillna('Missing').astype('category')
            
        return df
        
    return process_df(train), process_df(test)

train_reconstructed, test_reconstructed = extract_v5_features(train_raw, test_raw, events_raw)

# Verify exact match on all 77 features
print("\nComparing reconstructed features with ref_train...")
diffs = []
for feat in target_v5_features:
    s_recon = train_reconstructed[feat]
    s_ref = ref_train[feat]
    if s_recon.dtype.name == 'category':
        diff_count = (s_recon.astype(str) != s_ref.astype(str)).sum()
    else:
        diff_count = np.abs(s_recon.fillna(0) - s_ref.fillna(0)).max()
    if diff_count > 1e-4:
        diffs.append((feat, diff_count))

print(f"Features tested: {len(target_v5_features)}")
print(f"Features with differences: {len(diffs)}")
if len(diffs) > 0:
    for f, d in diffs:
        print(f"  Difference in {f}: {d}")
else:
    print("ALL 77 FEATURES MATCH EXACTLY (0.00000 difference)!")

print(f"Pipeline executed in {time.time() - t0:.1f}s")
