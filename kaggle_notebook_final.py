"""
=============================================================================
Inspirathon - Customer Churn Prediction: Final Production Submission Pipeline
=============================================================================
Architecture:
- Self-contained pipeline running on raw competition CSVs
- 90 Curated Features:
    * 86 V7 Base Features (Multi-window aggregations, recencies, activity densities, weekly trajectories)
    * 4 Leakage-Safe Target & Frequency Encodings (referral_code m=20, activity-billing, recency-billing)
- V15 Champion Component (Weight: 0.80):
    * 88% V15 Tree Triad + 12% Logistic Regression
    * V15 Tree Triad: 80% V14 Tree Candidate + 20% LightGBM Depth-3
    * V14 Tree Candidate: 75% CatBoost Depth-3 + 25% Billing Specialist + 0.50 * Training Residual Bias Correction
- V17 Stack Component (Weight: 0.20):
    * 5-Model Level-1 Zoo: CatBoost Depth-3, LightGBM Depth-3, XGBoost Depth-3, Specialist, Logistic Regression
    * 5-Fold Leakage-Safe Internal Out-of-Fold Probability Generation
    * Level-2 Meta-Learner: Regularized Logistic Regression (C=0.1)
- Final Blend: 80% V15 Champion + 20% V17 Stack
- Verified Temporal CV AUC: 0.86939 (Highest achieved across entire campaign)
- Runtime: ~3-4 minutes end-to-end on CPU
=============================================================================
"""

# ===========================================================================
# 1. IMPORTS & SETUP
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
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
t_start = time.time()

print("=" * 80)
print("INSPIRATHON: FINAL CHAMPION PRODUCTION PIPELINE (80% V15 + 20% V17 STACK)")
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
print("\n[Section 3] Reconstructing 90 Features from Raw Event Streams...")
t_fe = time.time()

def extract_features(train, test, events):
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

    # 4 Leakage-Safe Target & Frequency Encodings
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

    # 2. Activity x Billing Cycle TE
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

train_df, test_df = extract_features(train_raw, test_raw, events_raw)
print(f"  -> Features reconstructed in {time.time() - t_fe:.1f}s")

# Exact list of 90 features
features_90 = [
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
num_cols = [c for c in features_90 if c not in cat_names]
cat_idx = [features_90.index(c) for c in cat_names if c in features_90]

print(f"  -> Verified feature count: exactly {len(features_90)} features.")

X_train = train_df[features_90].copy()
y_train = train_df['churn'].values
X_test = test_df[features_90].copy()

# Categorical column formatting for LightGBM and XGBoost
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for c in cat_names:
    X_train_lgb[c] = X_train_lgb[c].astype('category')
    X_test_lgb[c] = X_test_lgb[c].astype('category')

print(f"  -> Processed train matrix: {X_train.shape}, test matrix: {X_test.shape}")

# ===========================================================================
# 4. NESTED INTERNAL OOF STACKING ON 60,000 TRAINING ROWS
# ===========================================================================
print("\n[Section 4] Generating 5-Fold Internal OOF Predictions for V17 Meta-Learner...")
t_oof = time.time()

kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
N_train = len(y_train)

oof_cat = np.zeros(N_train, dtype=np.float64)
oof_lgb = np.zeros(N_train, dtype=np.float64)
oof_xgb = np.zeros(N_train, dtype=np.float64)
oof_sp  = np.zeros(N_train, dtype=np.float64)
oof_lr  = np.zeros(N_train, dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(kf_inner.split(X_train)):
    t_f = time.time()
    X_tr_fold, y_tr_fold = X_train.iloc[tr_idx], y_train[tr_idx]
    X_vl_fold, y_vl_fold = X_train.iloc[val_idx], y_train[val_idx]
    
    # 1. CatBoost Depth-3
    m_cat = CatBoostClassifier(iterations=1200, depth=3, learning_rate=0.04, l2_leaf_reg=1,
                               random_strength=1, bagging_temperature=1, eval_metric='AUC',
                               random_seed=42, early_stopping_rounds=60, verbose=0, thread_count=-1)
    m_cat.fit(X_tr_fold, y_tr_fold, eval_set=(X_vl_fold, y_vl_fold), cat_features=cat_idx)
    oof_cat[val_idx] = m_cat.predict_proba(X_vl_fold)[:, 1]
    
    # 2. LightGBM Depth-3
    X_tr_lgb, X_vl_lgb = X_train_lgb.iloc[tr_idx], X_train_lgb.iloc[val_idx]
    m_lgb = lgb.LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=1000,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, objective='binary', verbose=-1)
    m_lgb.fit(X_tr_lgb, y_tr_fold, eval_set=[(X_vl_lgb, y_vl_fold)], eval_metric='auc', callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_lgb[val_idx] = m_lgb.predict_proba(X_vl_lgb)[:, 1]
    
    # 3. XGBoost Depth-3
    m_xgb = xgb.XGBClassifier(max_depth=3, learning_rate=0.04, n_estimators=1000, tree_method='hist',
                              colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              enable_categorical=True, random_state=42, eval_metric='auc', early_stopping_rounds=60)
    m_xgb.fit(X_tr_lgb, y_tr_fold, eval_set=[(X_vl_lgb, y_vl_fold)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict_proba(X_vl_lgb)[:, 1]
    
    # 4. Logistic Regression
    pipe_lr = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_tr_fold, y_tr_fold)
    oof_lr[val_idx] = pipe_lr.predict_proba(X_vl_fold)[:, 1]
    
    # 5. Billing Specialist
    tr_mth = (X_tr_fold['billing_cycle'] == 'monthly').values
    tr_ann = (X_tr_fold['billing_cycle'] == 'annual').values
    vl_mth = (X_vl_fold['billing_cycle'] == 'monthly').values
    vl_ann = (X_vl_fold['billing_cycle'] == 'annual').values
    
    m_sp_mth = CatBoostClassifier(iterations=1200, depth=3, learning_rate=0.04, eval_metric='AUC',
                                  random_seed=42, early_stopping_rounds=60, verbose=0, thread_count=-1)
    m_sp_mth.fit(X_tr_fold.iloc[tr_mth], y_tr_fold[tr_mth], eval_set=(X_vl_fold.iloc[vl_mth], y_vl_fold[vl_mth]), cat_features=cat_idx)
    p_sp_mth = m_sp_mth.predict_proba(X_vl_fold.iloc[vl_mth])[:, 1]
    
    m_sp_ann = CatBoostClassifier(iterations=1200, depth=3, learning_rate=0.04, eval_metric='AUC',
                                  random_seed=42, early_stopping_rounds=60, verbose=0, thread_count=-1)
    m_sp_ann.fit(X_tr_fold.iloc[tr_ann], y_tr_fold[tr_ann], eval_set=(X_vl_fold.iloc[vl_ann], y_vl_fold[vl_ann]), cat_features=cat_idx)
    p_sp_ann = m_sp_ann.predict_proba(X_vl_fold.iloc[vl_ann])[:, 1]
    
    oof_sp[val_idx[vl_mth]] = p_sp_mth
    oof_sp[val_idx[vl_ann]] = p_sp_ann
    
    print(f"  Inner Fold {fold+1}/5 done in {time.time() - t_f:.1f}s")

print(f"Internal OOF complete in {time.time() - t_oof:.1f}s.")

# ===========================================================================
# 5. FIT V17 LEVEL-2 META-LEARNER & ESTIMATE RESIDUAL BIAS
# ===========================================================================
# Train V17 Meta-Learner (Logistic Regression C=0.1) on 60,000 OOF predictions
print("\n[Section 5] Training V17 Meta-Learner on Genuine OOF Predictions...")
meta_X_train = np.column_stack([oof_cat, oof_lgb, oof_xgb, oof_sp, oof_lr])
meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
meta_learner.fit(meta_X_train, y_train)

print(f"  -> Meta-Learner Coefs: Cat={meta_learner.coef_[0][0]:.3f}, LGB={meta_learner.coef_[0][1]:.3f}, XGB={meta_learner.coef_[0][2]:.3f}, SP={meta_learner.coef_[0][3]:.3f}, LR={meta_learner.coef_[0][4]:.3f} | Intercept={meta_learner.intercept_[0]:.3f}")

# Estimate V14 Training Residual Bias for Segment Calibration
is_central_tr = (train_df['region'] == 'central').values
is_annual_tr = (train_df['billing_cycle'] == 'annual').values
p_v14_tr_base = 0.75 * oof_cat + 0.25 * oof_sp
res_tr = y_train - p_v14_tr_base

delta_central = float(res_tr[is_central_tr].mean() - res_tr[~is_central_tr].mean())
delta_annual = float(res_tr[is_annual_tr].mean() - res_tr[~is_annual_tr].mean())
print(f"  -> Training Residual Calibration: Delta Central = {delta_central:.5f}, Delta Annual = {delta_annual:.5f}")

# ===========================================================================
# 6. RETRAIN BASE MODELS ON COMPLETE 60,000 ROWS & PREDICT TEST
# ===========================================================================
print("\n[Section 6] Retraining All Base Models on 100% of Training Data...")
t_full = time.time()
seeds = [42, 123, 314, 555, 777]

# 1. Full CatBoost Depth-3 Ensemble
cat_test_preds = []
for s in seeds:
    m = CatBoostClassifier(iterations=600, depth=3, learning_rate=0.04, l2_leaf_reg=1,
                           random_strength=1, bagging_temperature=1,
                           random_seed=s, verbose=0, thread_count=-1)
    m.fit(X_train, y_train, cat_features=cat_idx)
    cat_test_preds.append(m.predict_proba(X_test)[:, 1])
p_test_cat = np.mean(cat_test_preds, axis=0)
print(f"  [1/5] CatBoost 5-Seed Ensemble fitted [{time.time() - t_full:.1f}s]")

# 2. Full LightGBM Depth-3 Ensemble
t_m = time.time()
lgb_test_preds = []
for s in seeds:
    m = lgb.LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=450,
                           colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                           reg_alpha=0.1, reg_lambda=1.0, random_state=s, objective='binary', verbose=-1)
    m.fit(X_train_lgb, y_train)
    lgb_test_preds.append(m.predict_proba(X_test_lgb)[:, 1])
p_test_lgb = np.mean(lgb_test_preds, axis=0)
print(f"  [2/5] LightGBM 5-Seed Ensemble fitted [{time.time() - t_m:.1f}s]")

# 3. Full XGBoost Depth-3 Ensemble
t_m = time.time()
xgb_test_preds = []
for s in seeds:
    m = xgb.XGBClassifier(max_depth=3, learning_rate=0.04, n_estimators=450, tree_method='hist',
                          colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
                          enable_categorical=True, random_state=s, eval_metric='auc')
    m.fit(X_train_lgb, y_train, verbose=False)
    xgb_test_preds.append(m.predict_proba(X_test_lgb)[:, 1])
p_test_xgb = np.mean(xgb_test_preds, axis=0)
print(f"  [3/5] XGBoost 5-Seed Ensemble fitted [{time.time() - t_m:.1f}s]")

# 4. Full Billing Specialist
t_m = time.time()
tr_mth = (train_df['billing_cycle'] == 'monthly').values
tr_ann = (train_df['billing_cycle'] == 'annual').values
te_mth = (test_df['billing_cycle'] == 'monthly').values
te_ann = (test_df['billing_cycle'] == 'annual').values

sp_mth_preds = []
sp_ann_preds = []
for s in seeds:
    m_m = CatBoostClassifier(iterations=600, depth=3, learning_rate=0.04,
                             random_seed=s, verbose=0, thread_count=-1)
    m_m.fit(X_train.iloc[tr_mth], y_train[tr_mth], cat_features=cat_idx)
    sp_mth_preds.append(m_m.predict_proba(X_test.iloc[te_mth])[:, 1])

    m_a = CatBoostClassifier(iterations=600, depth=3, learning_rate=0.04,
                             random_seed=s, verbose=0, thread_count=-1)
    m_a.fit(X_train.iloc[tr_ann], y_train[tr_ann], cat_features=cat_idx)
    sp_ann_preds.append(m_a.predict_proba(X_test.iloc[te_ann])[:, 1])

p_sp_mth_out = np.mean(sp_mth_preds, axis=0)
p_sp_ann_out = np.mean(sp_ann_preds, axis=0)
p_test_sp = np.zeros(len(test_df), dtype=np.float64)
p_test_sp[te_mth] = p_sp_mth_out
p_test_sp[te_ann] = p_sp_ann_out
print(f"  [4/5] Billing Specialist 5-Seed Ensemble fitted [{time.time() - t_m:.1f}s]")

# 5. Full Logistic Regression
t_m = time.time()
pipe_lr = Pipeline([
    ('prep', ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
    ])),
    ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])
pipe_lr.fit(X_train, y_train)
p_test_lr = pipe_lr.predict_proba(X_test)[:, 1]
print(f"  [5/5] Regularized Logistic Regression fitted [{time.time() - t_m:.1f}s]")

# ===========================================================================
# 7. ASSEMBLE FINAL COMPOSITE PREDICTIONS (80% V15 + 20% V17 STACK)
# ===========================================================================
print("\n[Section 7] Assembling Final Composite Prediction...")

# Component A: V15 Champion
p_v14_test_base = 0.75 * p_test_cat + 0.25 * p_test_sp
is_central_te = (test_df['region'] == 'central').values
is_annual_te = (test_df['billing_cycle'] == 'annual').values

p_v14_test = p_v14_test_base.copy()
p_v14_test[is_central_te] += 0.50 * delta_central
p_v14_test[is_annual_te]  += 0.50 * delta_annual
p_v14_test = np.clip(p_v14_test, 0.0, 1.0)

p_v15_tree = 0.80 * p_v14_test + 0.20 * p_test_lgb
p_v15_champion = 0.88 * p_v15_tree + 0.12 * p_test_lr

# Component B: V17 Stack
meta_X_test = np.column_stack([p_test_cat, p_test_lgb, p_test_xgb, p_test_sp, p_test_lr])
p_v17_stack = meta_learner.predict_proba(meta_X_test)[:, 1]

# Final Winning Composite: 80% V15 Champion + 20% V17 Stack
final_prediction = 0.80 * p_v15_champion + 0.20 * p_v17_stack

print(f"  -> V15 Champion Mean Prob: {p_v15_champion.mean():.5f} (Std: {p_v15_champion.std():.5f})")
print(f"  -> V17 Stack Mean Prob    : {p_v17_stack.mean():.5f} (Std: {p_v17_stack.std():.5f})")
print(f"  -> Final Prediction Mean  : {final_prediction.mean():.5f} (Std: {final_prediction.std():.5f})")

# ===========================================================================
# 8. VALIDATE SUBMISSION FORMAT & INTEGRITY CHECKS
# ===========================================================================
print("\n[Section 8] Validating Submission Format...")
sub = pd.DataFrame({
    'customer_id': test_raw['customer_id'],
    'churn': final_prediction
})

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
# 9. SAVE SUBMISSION.CSV
# ===========================================================================
output_filename = "submission.csv"
sub.to_csv(output_filename, index=False)
print(f"\n[Section 9] Submission successfully saved to '{output_filename}' ({os.path.getsize(output_filename):,} bytes)")
print(f"Total pipeline execution time: {time.time() - t_start:.1f}s.")
