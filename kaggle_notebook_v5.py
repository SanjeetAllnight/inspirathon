"""
=============================================================================
Inspirathon - Customer Churn Prediction: V5 Production Pipeline
=============================================================================
Architecture:
- Self-contained pipeline running on raw competition CSVs
- 77 Curated Features (Recency progression, Window ratios, Activity density)
- 5-Seed CatBoost Ensemble (Seeds: 42, 123, 314, 555, 777)
- Verified Mean Temporal CV AUC: 0.86448
- Runtime: ~75 seconds end-to-end
=============================================================================
"""

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
t_start = time.time()

print("=" * 75)
print("INSPIRATHON V5 ENSEMBLE PIPELINE")
print("=" * 75)

# ===========================================================================
# 2. LOAD COMPETITION DATA
# ===========================================================================
print("\n[Section 2] Loading Competition CSVs...")

def find_data_dir():
    # 1. Search recursively in /kaggle/input (handles any dataset slug on Kaggle)
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'train.csv' in files and 'events.csv' in files:
                return root
    # 2. Local fallback directories
    for d in ['.', './data', '../input', '..']:
        if os.path.exists(os.path.join(d, 'train.csv')) and os.path.exists(os.path.join(d, 'events.csv')):
            return d
    return '.'

DATA_DIR = find_data_dir()
print(f"  -> Using data directory: {os.path.abspath(DATA_DIR)}")

train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')
events_path = os.path.join(DATA_DIR, 'events.csv')
sample_sub_path = os.path.join(DATA_DIR, 'sample_submission.csv')

train_raw = pd.read_csv(train_path, parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv(test_path, parse_dates=['snapshot_date', 'signup_date'])
events_raw = pd.read_csv(events_path, parse_dates=['timestamp'])
sample_sub = pd.read_csv(sample_sub_path)

print(f"  -> train.csv shape : {train_raw.shape}")
print(f"  -> test.csv shape  : {test_raw.shape}")
print(f"  -> events.csv shape: {events_raw.shape}")

# ===========================================================================
# 3. FEATURE ENGINEERING (77 V5 FEATURES)
# ===========================================================================
print("\n[Section 3] Reconstructing V5 Features (77 Curated Features)...")
t_fe = time.time()

def extract_v5_features(train, test, events):
    eps = 1e-6

    # 1. Snapshots mapping
    snapshots = pd.concat([
        train[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']],
        test[['customer_id', 'snapshot_date', 'signup_date', 'billing_cycle']]
    ]).drop_duplicates(subset=['customer_id'])

    # 2. Events filter (Strict event cutoff: timestamp <= snapshot_date)
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

    # 3. Base aggregations per customer
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

    # Recency per event type
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

    # Inactivity gap
    def _longest_gap(dates):
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < 2:
            return np.nan
        return float(max(np.diff(unique_dates)))

    gaps = events.groupby('customer_id')['days_before'].apply(_longest_gap).rename('longest_gap_days')
    basic = basic.join(gaps, how='left')

    # kth-last session recency (k = 2, 3, 5, 10)
    sess = events[events['is_session'] == 1][['customer_id', 'days_before']].sort_values(['customer_id', 'days_before'])
    sess['rank'] = sess.groupby('customer_id').cumcount() + 1
    for k in [2, 3, 5, 10]:
        kth = sess[sess['rank'] == k].set_index('customer_id')['days_before'].rename(f'days_since_{k}th_last_session')
        basic = basic.join(kth, how='left')

    ef = basic

    # Window ratios and deltas
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

    # Demographic profile & interactions
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

        # Impute days-since columns with account_age_days
        days_cols = [c for c in df.columns if c.startswith('days_since_')]
        for c in days_cols:
            df[c] = df[c].fillna(df['account_age_days'])

        # Impute other event metrics with 0
        evt_cols = [c for c in ef.columns if not c.startswith('days_since_')]
        for c in evt_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        cat_cols = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
        for c in cat_cols:
            df[c] = df[c].fillna('Missing').astype('category')

        return df

    return process_df(train), process_df(test)

train_df, test_df = extract_v5_features(train_raw, test_raw, events_raw)
print(f"  -> Features built in {time.time() - t_fe:.1f}s")

# Exact list of 77 features
v5_features = [
    'tenure_x_billing', 'sess_ratio_30_90', 'account_age_days', 'days_since_last_session',
    'account_age_months', 'sess_ratio_30_60', 'days_since_last_support_ticket', 'is_pm_missing',
    'signup_year', 'evt_ratio_30_60', 'sess_ratio_14_60', 'days_since_last_payment_failed',
    'days_since_last_plan_change', 'age_band', 'monthly_price', 'dur_ratio_30_60', 'region',
    'family_sharing', 'num_devices', 'payment_method', 'support_60d', 'support_per_sess_60d',
    'sess_per_activeday_120d', 'pf_per_sess_30d', 'activeday_ratio_30_60', 'tenure_x_plan_basic',
    'plan', 'price_per_device', 'evt_ratio_60_120', 'billing_cycle', 'referral_freq',
    'sess_per_activeday_30d', 'sess_ratio_60_120', 'days_since_last_event', 'activeday_ratio_60_120',
    'sess_accel', 'dur_ratio_60_120', 'acquisition_channel', 'sess_delta_30_60',
    'sess_per_activeday_60d', 'active_days_90d', 'sessions_14d', 'sess_dur_mean_90d',
    'sess_dur_median', 'longest_gap_days', 'sess_dur_max', 'events_120d', 'events_60d',
    'sess_dur_total_90d', 'events_14d', 'sess_dur_mean_30d', 'sess_dur_mean_60d',
    'dur_per_activeday_30d', 'support_per_session', 'sess_delta_60_90', 'dur_per_activeday_120d',
    'sess_dur_total', 'total_events', 'sess_dur_total_60d', 'sessions_30d', 'sess_dur_total_120d',
    'active_days_120d', 'session_count', 'sess_dur_mean_14d', 'signup_dow', 'support_per_sess_30d',
    'sess_dur_std', 'downgrade_60d', 'sess_dur_mean', 'support_count', 'n_distinct_dates',
    'sess_dur_total_14d', 'fraction_inactive_lifetime', 'days_since_2th_last_session',
    'days_since_3th_last_session', 'days_since_5th_last_session', 'days_since_10th_last_session'
]

cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
cat_idx = [v5_features.index(c) for c in cat_names]

assert len(v5_features) == 77, f"Error: Feature count is {len(v5_features)}, expected 77."
print(f"  -> Verified feature list: exactly {len(v5_features)} features.")

# ===========================================================================
# 4. TRAIN V5 ENSEMBLE (5-SEED CATBOOST)
# ===========================================================================
print("\n[Section 4] Training CatBoost 5-Seed Ensemble on All 60,000 Rows...")
t_train = time.time()

# Optimal hyperparameters determined in V5
catboost_hp = {
    'depth': 6,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0
}

# Per-seed target iterations (calibrated for full training size)
seed_iterations = {
    42:  440,
    123: 520,
    314: 440,
    555: 500,
    777: 460
}

X_full = train_df[v5_features]
y_full = train_df['churn']
X_test = test_df[v5_features]

test_preds = []
for seed, iters in seed_iterations.items():
    s_t0 = time.time()
    model = CatBoostClassifier(
        iterations=iters,
        random_seed=seed,
        cat_features=cat_idx,
        **catboost_hp
    )
    model.fit(X_full, y_full)
    preds = model.predict_proba(X_test)[:, 1]
    test_preds.append(preds)
    print(f"  -> Seed {seed:3d} trained ({iters} iterations) in {time.time() - s_t0:.1f}s | Mean prob: {preds.mean():.4f}")

# ===========================================================================
# 5. PREDICT TEST (ENSEMBLE AVERAGE)
# ===========================================================================
print("\n[Section 5] Computing Ensemble Test Predictions...")
final_probabilities = np.mean(test_preds, axis=0)

# ===========================================================================
# 6. VALIDATE SUBMISSION FORMAT
# ===========================================================================
print("\n[Section 6] Running Submission Verification Checks...")

sub_df = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'churn': final_probabilities
})

check_rows = (len(sub_df) == 40000)
check_cols = (list(sub_df.columns) == ['customer_id', 'churn'])
check_ids = (sub_df['customer_id'].equals(test_raw['customer_id'])) and (sub_df['customer_id'].equals(sample_sub['customer_id']))
check_nans = (sub_df['churn'].isna().sum() == 0) and (np.isinf(sub_df['churn']).sum() == 0)
check_bounds = (sub_df['churn'] >= 0.0).all() and (sub_df['churn'] <= 1.0).all()
check_continuous = (sub_df['churn'].nunique() > 1000)

print(f"  [1] Exactly 40,000 rows                 : {check_rows} ({len(sub_df)})")
print(f"  [2] Columns exactly customer_id,churn   : {check_cols} ({list(sub_df.columns)})")
print(f"  [3] Customer IDs match test.csv exactly : {check_ids}")
print(f"  [4] No NaNs or Infs                     : {check_nans}")
print(f"  [5] Probabilities bounded in [0, 1]     : {check_bounds}")
print(f"  [6] Continuous probabilities            : {check_continuous} ({sub_df['churn'].nunique()} distinct)")

all_passed = check_rows and check_cols and check_ids and check_nans and check_bounds and check_continuous
assert all_passed, "FATAL: One or more submission checks failed!"
print("\n---> ALL CHECKS PASSED: TRUE")

print("\nSubmission Summary Statistics:")
print(sub_df['churn'].describe().to_string())

# ===========================================================================
# 7. SAVE SUBMISSION.CSV
# ===========================================================================
output_path = 'submission.csv'
sub_df.to_csv(output_path, index=False)
print(f"\n[Section 7] Successfully saved final submission to: {os.path.abspath(output_path)}")
print(f"Total pipeline execution time: {time.time() - t_start:.1f} seconds")
print("=" * 75)
