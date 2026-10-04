"""
=============================================================================
Inspirathon - Customer Churn Prediction: V9 Champion Production Pipeline
=============================================================================
Architecture:
- Self-contained pipeline running on raw competition CSVs
- 90 Curated Features:
    * 86 V7 Base Features (Multi-window rollups, recencies, ratios, weekly profiles)
    * 4 Leakage-Safe Target & Frequency Encodings (referral_code m=20, activity-billing, recency-billing)
- 5-Seed Depth-4 CatBoost Ensemble (Seeds: 42, 123, 314, 555, 777)
- Regularized Logistic Regression (C=0.1, Median Imputer + Scaler + OHE)
- Winning Blend: 88% CatBoost Ensemble + 12% Logistic Regression
- Verified Mean Temporal CV AUC: 0.86824
- Runtime: ~60-90 seconds end-to-end on CPU
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
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
t_start = time.time()

print("=" * 80)
print("INSPIRATHON V9 CHAMPION PRODUCTION PIPELINE")
print("=" * 80)

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
# 3. FEATURE ENGINEERING (86 BASE FEATURES + 4 ENCODINGS = 90 FEATURES)
# ===========================================================================
print("\n[Section 3] Reconstructing V9 Features (86 Base + 4 Encodings = 90 Features)...")
t_fe = time.time()

def extract_v9_features(train, test, events):
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

    # Weekly Profile & Trajectory (W1..W8)
    events['week_idx'] = np.floor(events['days_before'] / 7.0).astype(int) + 1
    events_8w = events[(events['week_idx'] >= 1) & (events['week_idx'] <= 8)]
    w_agg = events_8w.groupby(['customer_id', 'week_idx']).agg(
        active_days = ('event_date', 'nunique'),
        sess_count = ('is_session', 'sum')
    ).unstack(level='week_idx')

    all_custs = snapshots['customer_id'].unique()
    w_agg = w_agg.reindex(all_custs).fillna(0)

    weekly_df = pd.DataFrame(index=all_custs)
    for w in range(1, 9):
        weekly_df[f'active_days_w{w}'] = w_agg[('active_days', w)].values

    sess_matrix = np.array([w_agg[('sess_count', w)].values for w in range(1, 9)]).T
    weights_8w = np.array([(i - 4.5) / 42.0 for i in range(1, 9)])
    weekly_df['session_slope_8w'] = np.dot(sess_matrix[:, ::-1], weights_8w)

    basic = basic.join(weekly_df, how='left')
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

    tr_proc = process_df(train)
    te_proc = process_df(test)

    # -----------------------------------------------------------------------
    # 4 LEAKAGE-SAFE TARGET & FREQUENCY ENCODINGS
    # -----------------------------------------------------------------------
    # Helper for OOF train encoding and full test mapping
    def compute_safe_te(train_series, y_train, test_series, smoothing=20.0, n_splits=5, seed=42):
        global_mean = float(y_train.mean())
        oof_te = np.full(len(train_series), np.nan, dtype=np.float32)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_data = pd.DataFrame({'cat': train_series, 'y': y_train})

        for tr_idx, val_idx in kf.split(train_data):
            tr_part = train_data.iloc[tr_idx]
            val_part = train_data.iloc[val_idx]
            inner_prior = float(tr_part['y'].mean())
            stats = tr_part.groupby('cat')['y'].agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + smoothing * inner_prior) / (stats['count'] + smoothing)
            oof_te[val_idx] = val_part['cat'].map(smoothed).fillna(inner_prior).values

        # Full training stats for test mapping
        full_stats = train_data.groupby('cat')['y'].agg(['count', 'mean'])
        full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
        test_te = test_series.map(full_smoothed).fillna(global_mean).values.astype(np.float32)
        return oof_te, test_te

    # 1. Referral Code Encodings
    tr_ref = tr_proc['referral_code'].fillna('NO_REF').astype(str)
    te_ref = te_proc['referral_code'].fillna('NO_REF').astype(str)
    ref_counts = tr_ref.value_counts()
    
    tr_proc['ref_code_train_freq'] = tr_ref.map(ref_counts).fillna(0).values
    te_proc['ref_code_train_freq'] = te_ref.map(ref_counts).fillna(0).values

    oof_ref_te, test_ref_te = compute_safe_te(tr_ref, tr_proc['churn'].values, te_ref, smoothing=20.0)
    tr_proc['ref_code_te'] = oof_ref_te
    te_proc['ref_code_te'] = test_ref_te

    # 2. Coarse Event Activity x Billing Cycle TE
    def act_bin(df):
        s = df['session_count'].fillna(0)
        b = pd.cut(s, bins=[-1, 0, 5, 15, 30, 9999], labels=['0', '1', '2', '3', '4']).astype(str)
        return 'act_' + b + '__' + df['billing_cycle'].astype(str)

    def rec_bin(df):
        r = df['days_since_last_session'].fillna(999)
        b = pd.cut(r, bins=[-1, 7, 14, 30, 60, 9999], labels=['7', '14', '30', '60', 'inf']).astype(str)
        return 'rec_' + b + '__' + df['billing_cycle'].astype(str)

    oof_act_te, test_act_te = compute_safe_te(act_bin(tr_proc), tr_proc['churn'].values, act_bin(te_proc), smoothing=15.0)
    tr_proc['act_bin_billing_te'] = oof_act_te
    te_proc['act_bin_billing_te'] = test_act_te

    oof_rec_te, test_rec_te = compute_safe_te(rec_bin(tr_proc), tr_proc['churn'].values, rec_bin(te_proc), smoothing=15.0)
    tr_proc['rec_bin_billing_te'] = oof_rec_te
    te_proc['rec_bin_billing_te'] = test_rec_te

    return tr_proc, te_proc

train_df, test_df = extract_v9_features(train_raw, test_raw, events_raw)
print(f"  -> Features built in {time.time() - t_fe:.1f}s")

# Exact list of 90 V9 features
v9_features = [
    # 86 V7 Base Features
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
    'days_since_3th_last_session', 'days_since_5th_last_session', 'days_since_10th_last_session',
    'active_days_w1', 'active_days_w2', 'active_days_w3', 'active_days_w4', 'active_days_w5',
    'active_days_w6', 'active_days_w7', 'active_days_w8', 'session_slope_8w',
    # 4 V9 Encodings
    'ref_code_train_freq', 'ref_code_te', 'act_bin_billing_te', 'rec_bin_billing_te'
]

cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
cat_idx = [v9_features.index(c) for c in cat_names if c in v9_features]

print(f"  -> Verified feature count: exactly {len(v9_features)} features.")

# ===========================================================================
# 4. TRAIN V9 CATBOOST 5-SEED ENSEMBLE (DEPTH 4)
# ===========================================================================
print("\n[Section 4] Training Depth-4 CatBoost 5-Seed Ensemble on All 60,000 Rows...")
t_train_cb = time.time()

# Optimal hyperparameters determined in V9
catboost_hp = {
    'depth': 4,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0,
    'thread_count': -1
}

# Per-seed calibrated iterations for full training size (60,000 rows)
seed_iterations = {
    42:  600,
    123: 700,
    314: 650,
    555: 750,
    777: 750
}

X_full = train_df[v9_features]
y_full = train_df['churn'].values
X_test = test_df[v9_features]

cb_test_preds = []
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
    cb_test_preds.append(preds)
    print(f"  -> Seed {seed:3d} trained ({iters} iterations) in {time.time() - s_t0:.1f}s | Mean prob: {preds.mean():.4f}")

cb_ensemble_pred = np.mean(cb_test_preds, axis=0)
print(f"CatBoost Ensemble Complete in {time.time() - t_train_cb:.1f}s | Ensemble mean: {cb_ensemble_pred.mean():.4f}")

# ===========================================================================
# 5. TRAIN REGULARIZED LOGISTIC REGRESSION (V7 BASE FEATURES)
# ===========================================================================
print("\n[Section 5] Training Regularized Logistic Regression on Base V7 Features...")
t_train_lr = time.time()

v7_features = v9_features[:86] # Exactly the 86 base features
num_cols = [c for c in v7_features if c not in cat_names]

lr_pipeline = Pipeline([
    ('prep', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
    ])),
    ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])

lr_pipeline.fit(train_df[v7_features], y_full)
lr_test_preds = lr_pipeline.predict_proba(test_df[v7_features])[:, 1]
print(f"Logistic Regression Complete in {time.time() - t_train_lr:.1f}s | LR test mean: {lr_test_preds.mean():.4f}")

# ===========================================================================
# 6. COMPUTE FINAL WINNING BLEND (88% CATBOOST + 12% LOGISTIC REGRESSION)
# ===========================================================================
print("\n[Section 6] Computing Final V9 Winning Blend (88% CatBoost + 12% LR)...")
final_probabilities = 0.88 * cb_ensemble_pred + 0.12 * lr_test_preds

# ===========================================================================
# 7. VALIDATE SUBMISSION FORMAT
# ===========================================================================
print("\n[Section 7] Running Comprehensive Submission Format & Integrity Checks...")

sub = pd.DataFrame({
    'customer_id': test_df['customer_id'].values,
    'churn': final_probabilities
})

# Verification tests
checks_passed = True

# Check 1: Row count exactly 40,000
if len(sub) == 40000:
    print("  [PASS] Exactly 40,000 rows.")
else:
    print(f"  [FAIL] Expected 40,000 rows, got {len(sub)}")
    checks_passed = False

# Check 2: Correct column names
if list(sub.columns) == ['customer_id', 'churn']:
    print("  [PASS] Columns are exactly ['customer_id', 'churn'].")
else:
    print(f"  [FAIL] Invalid columns: {list(sub.columns)}")
    checks_passed = False

# Check 3: Customer IDs match test.csv in exact order
if (sub['customer_id'].values == test_raw['customer_id'].values).all():
    print("  [PASS] Customer IDs match test.csv in exact order.")
else:
    print("  [FAIL] Customer IDs do not match test.csv order!")
    checks_passed = False

# Check 4: No missing, null, or infinite values
if not sub['churn'].isnull().any() and not np.isinf(sub['churn']).any():
    print("  [PASS] Zero NaN or Inf values.")
else:
    print("  [FAIL] Found NaN or Inf values in predictions!")
    checks_passed = False

# Check 5: Probabilities strictly bounded in [0, 1]
if (sub['churn'] >= 0.0).all() and (sub['churn'] <= 1.0).all():
    print(f"  [PASS] Valid probabilities: min={sub['churn'].min():.5f}, max={sub['churn'].max():.5f}")
else:
    print("  [FAIL] Predictions out of [0, 1] bounds!")
    checks_passed = False

# Check 6: Probability distribution sanity
mean_p = sub['churn'].mean()
std_p = sub['churn'].std()
print(f"  [INFO] Probability distribution: Mean = {mean_p:.5f}, Std = {std_p:.5f}")
print(f"  [INFO] Deciles: 10%={sub['churn'].quantile(0.1):.4f}, 50%={sub['churn'].quantile(0.5):.4f}, 90%={sub['churn'].quantile(0.9):.4f}")

if not checks_passed:
    raise ValueError("Submission integrity checks FAILED! Aborting export.")

# ===========================================================================
# 8. SAVE SUBMISSION.CSV
# ===========================================================================
output_filename = "submission.csv"
sub.to_csv(output_filename, index=False)
print(f"\n[Section 8] Submission successfully saved to '{output_filename}' ({os.path.getsize(output_filename):,} bytes)")
print(f"Pipeline executed end-to-end in {time.time() - t_start:.1f}s.")
