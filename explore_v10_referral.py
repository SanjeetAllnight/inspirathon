import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("V10: DEEP EXPLORATION OF REFERRAL NETWORK SIGNALS")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Add raw columns needed for encodings
for col in ['referral_code', 'plan', 'billing_cycle', 'region', 'acquisition_channel']:
    if col not in train_df.columns:
        train_df[col] = train_raw[col].values

# Leak-safe target encoding helper
def compute_oof_target_encoding(train_s, y_train, val_s, smoothing=20.0, n_splits=5, seed=42):
    global_mean = y_train.mean()
    oof_te = np.full(len(train_s), np.nan, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_data = pd.DataFrame({'cat': train_s, 'y': y_train})
    
    for tr_idx, val_idx in kf.split(train_data):
        tr_part = train_data.iloc[tr_idx]
        val_part = train_data.iloc[val_idx]
        inner_prior = tr_part['y'].mean()
        stats = tr_part.groupby('cat')['y'].agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * inner_prior) / (stats['count'] + smoothing)
        oof_te[val_idx] = val_part['cat'].map(smoothed).fillna(inner_prior).values
        
    full_stats = train_data.groupby('cat')['y'].agg(['count', 'mean'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
    val_te = val_s.map(full_smoothed).fillna(global_mean).values.astype(np.float32)
    return oof_te, val_te

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

hp_d4 = {
    'depth': 4,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0
}

# Base V9 features generator for baseline comparison
def get_v9_features_for_fold(tr_sub, y_tr, vl_sub, smoothing=20.0):
    tr_ref = tr_sub['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl_sub['referral_code'].fillna('NO_REF').astype(str)
    ref_counts = tr_ref.value_counts()
    
    oof_te_ref, val_te_ref = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=smoothing)
    
    def act_bin(df):
        s = df['session_count'].fillna(0)
        b = pd.cut(s, bins=[-1, 0, 5, 15, 30, 9999], labels=['0', '1', '2', '3', '4']).astype(str)
        return 'act_' + b + '__' + df['billing_cycle'].astype(str)
    def rec_bin(df):
        r = df['days_since_last_session'].fillna(999)
        b = pd.cut(r, bins=[-1, 7, 14, 30, 60, 9999], labels=['7', '14', '30', '60', 'inf']).astype(str)
        return 'rec_' + b + '__' + df['billing_cycle'].astype(str)
        
    oof_act_te, val_act_te = compute_oof_target_encoding(act_bin(tr_sub), y_tr, act_bin(vl_sub), smoothing=15.0)
    oof_rec_te, val_rec_te = compute_oof_target_encoding(rec_bin(tr_sub), y_tr, rec_bin(vl_sub), smoothing=15.0)
    
    new_tr = pd.DataFrame({
        'ref_code_train_freq': tr_ref.map(ref_counts).fillna(0).values,
        'ref_code_te': oof_te_ref,
        'act_bin_billing_te': oof_act_te,
        'rec_bin_billing_te': oof_rec_te
    })
    new_vl = pd.DataFrame({
        'ref_code_train_freq': vl_ref.map(ref_counts).fillna(0).values,
        'ref_code_te': val_te_ref,
        'act_bin_billing_te': val_act_te,
        'rec_bin_billing_te': val_rec_te
    })
    
    X_tr = pd.concat([tr_sub[v7_features].reset_index(drop=True), new_tr.reset_index(drop=True)], axis=1)
    X_vl = pd.concat([vl_sub[v7_features].reset_index(drop=True), new_vl.reset_index(drop=True)], axis=1)
    return X_tr, X_vl

# Evaluate baseline V9
print("\n[Baseline] Verifying V9 Baseline on Depth 4...")
v9_base_folds = []
for fold_name, tr_mask, vl_mask in cv_splits:
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub, smoothing=20.0)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
    m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    v9_base_folds.append(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1]))

v9_base_mean = np.mean(v9_base_folds)
print(f"V9 Baseline (Single Depth 4) -> Mean CV: {v9_base_mean:.5f} | F1: {v9_base_folds[0]:.5f}, F2: {v9_base_folds[1]:.5f}, F3: {v9_base_folds[2]:.5f}")

# ===========================================================================
# EXPERIMENT 1: SMOOTHING STRENGTH SWEEP (m = 5, 10, 20, 30, 40, 60, 80)
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 1: SMOOTHING STRENGTH SWEEP")
print("=" * 60)

smoothing_candidates = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0]
sweep_results = {}

for m_val in smoothing_candidates:
    f_res = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub, smoothing=m_val)
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_res.append(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1]))
    mean_res = np.mean(f_res)
    delta = mean_res - v9_base_mean
    sweep_results[m_val] = (mean_res, delta, f_res)
    best_marker = " *BEST*" if m_val == max(sweep_results.keys(), key=lambda k: sweep_results[k][0]) else ""
    print(f"  Smoothing m={m_val:4.1f} -> Mean CV: {mean_res:.5f} (Δ: {delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}{best_marker}")

best_m = max(sweep_results.keys(), key=lambda k: sweep_results[k][0])
print(f"\nWinning Smoothing Strength: m = {best_m} (Mean CV: {sweep_results[best_m][0]:.5f})")

# ===========================================================================
# EXPERIMENT 2: FREQUENCY REPRESENTATIONS & RARE/FREQUENT FLAGS
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: FREQUENCY ENHANCEMENTS & RARE/FREQUENT SEGMENTATION")
print("=" * 60)

def test_exp2_features(gen_fn, name):
    f_res = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask].copy()
        vl_sub = train_df[vl_mask].copy()
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        
        # Base V9 with best_m
        X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub, smoothing=best_m)
        extra_tr, extra_vl = gen_fn(tr_sub, y_tr, vl_sub)
        
        X_tr = pd.concat([X_tr, extra_tr.reset_index(drop=True)], axis=1)
        X_vl = pd.concat([X_vl, extra_vl.reset_index(drop=True)], axis=1)
        
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_res.append(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1]))
    m_res = np.mean(f_res)
    delta = m_res - sweep_results[best_m][0]
    print(f"  {name:40s} -> Mean: {m_res:.5f} (Δ: {delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}")
    return m_res, delta, f_res

# 2a: Log frequency & relative frequency
def gen_freq_features(tr, y_tr, vl):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    counts = tr_ref.value_counts()
    tr_log_freq = np.log1p(tr_ref.map(counts).fillna(0).values)
    vl_log_freq = np.log1p(vl_ref.map(counts).fillna(0).values)
    
    tr_is_rare = ((tr_ref.map(counts).fillna(0) <= 3) & (tr_ref != 'NO_REF')).astype(int).values
    vl_is_rare = ((vl_ref.map(counts).fillna(0) <= 3) & (vl_ref != 'NO_REF')).astype(int).values
    
    tr_is_frequent = (tr_ref.map(counts).fillna(0) >= 15).astype(int).values
    vl_is_frequent = (vl_ref.map(counts).fillna(0) >= 15).astype(int).values
    
    return pd.DataFrame({'ref_log_freq': tr_log_freq, 'ref_is_rare': tr_is_rare, 'ref_is_frequent': tr_is_frequent}), \
           pd.DataFrame({'ref_log_freq': vl_log_freq, 'ref_is_rare': vl_is_rare, 'ref_is_frequent': vl_is_frequent})

test_exp2_features(gen_freq_features, "2a. Log-Freq + Rare/Frequent Flags")

# ===========================================================================
# EXPERIMENT 3: INTERACTIONS WITH DEMOGRAPHICS & PROFILE
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: INTERACTIONS (BILLING, PLAN, CHANNEL, REGION)")
print("=" * 60)

# 3a: Grouped TE (referral_code x billing_cycle)
def gen_ref_x_billing(tr, y_tr, vl):
    tr_cat = tr['referral_code'].fillna('NO_REF').astype(str) + '__' + tr['billing_cycle'].astype(str)
    vl_cat = vl['referral_code'].fillna('NO_REF').astype(str) + '__' + vl['billing_cycle'].astype(str)
    oof_te, val_te = compute_oof_target_encoding(tr_cat, y_tr, vl_cat, smoothing=best_m)
    return pd.DataFrame({'ref_x_billing_te': oof_te}), pd.DataFrame({'ref_x_billing_te': val_te})

test_exp2_features(gen_ref_x_billing, "3a. Grouped TE: referral x billing_cycle")

# 3b: Profile interaction products
def gen_ref_prod_interactions(tr, y_tr, vl):
    # Compute base ref_code_te first
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    oof_te, val_te = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=best_m)
    
    tr_prod_monthly = oof_te * (tr['billing_cycle'] == 'monthly').astype(int).values
    vl_prod_monthly = val_te * (vl['billing_cycle'] == 'monthly').astype(int).values
    
    tr_prod_basic = oof_te * (tr['plan'] == 'basic').astype(int).values
    vl_prod_basic = val_te * (vl['plan'] == 'basic').astype(int).values
    
    return pd.DataFrame({'ref_te_x_monthly': tr_prod_monthly, 'ref_te_x_plan_basic': tr_prod_basic}), \
           pd.DataFrame({'ref_te_x_monthly': vl_prod_monthly, 'ref_te_x_plan_basic': vl_prod_basic})

test_exp2_features(gen_ref_prod_interactions, "3b. Products: ref_TE x Monthly & Plan Basic")

# ===========================================================================
# EXPERIMENT 4: REFERRAL BEHAVIOR + ENGAGEMENT INTERACTIONS
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENT 4: REFERRAL x ENGAGEMENT (INACTIVITY, RECENCY, MOMENTUM)")
print("=" * 60)

def gen_ref_engagement(tr, y_tr, vl):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    oof_te, val_te = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=best_m)
    
    # 1. ref_TE x inactivity
    tr_inact = oof_te * tr['fraction_inactive_lifetime'].fillna(0).values
    vl_inact = val_te * vl['fraction_inactive_lifetime'].fillna(0).values
    
    # 2. ref_TE x recency (log days since last session)
    tr_rec = oof_te * np.log1p(tr['days_since_last_session'].fillna(999).clip(lower=0).values)
    vl_rec = val_te * np.log1p(vl['days_since_last_session'].fillna(999).clip(lower=0).values)
    
    # 3. ref_TE x session slope (momentum)
    tr_slope = oof_te * tr['session_slope_8w'].fillna(0).values
    vl_slope = val_te * vl['session_slope_8w'].fillna(0).values
    
    return pd.DataFrame({'ref_te_x_inactivity': tr_inact, 'ref_te_x_log_recency': tr_rec, 'ref_te_x_slope': tr_slope}), \
           pd.DataFrame({'ref_te_x_inactivity': vl_inact, 'ref_te_x_log_recency': vl_rec, 'ref_te_x_slope': vl_slope})

test_exp2_features(gen_ref_engagement, "4. ref_TE x Inactivity, Recency, Momentum")

# Test individual engagement components
def gen_ref_inact_only(tr, y_tr, vl):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    oof_te, val_te = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=best_m)
    return pd.DataFrame({'ref_te_x_inactivity': oof_te * tr['fraction_inactive_lifetime'].fillna(0).values}), \
           pd.DataFrame({'ref_te_x_inactivity': val_te * vl['fraction_inactive_lifetime'].fillna(0).values})

test_exp2_features(gen_ref_inact_only, "4a. ref_TE x Inactivity Only")

def gen_ref_rec_only(tr, y_tr, vl):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    oof_te, val_te = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=best_m)
    return pd.DataFrame({'ref_te_x_log_recency': oof_te * np.log1p(tr['days_since_last_session'].fillna(999).clip(lower=0).values)}), \
           pd.DataFrame({'ref_te_x_log_recency': val_te * np.log1p(vl['days_since_last_session'].fillna(999).clip(lower=0).values)})

test_exp2_features(gen_ref_rec_only, "4b. ref_TE x Log Recency Only")

def gen_ref_slope_only(tr, y_tr, vl):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    oof_te, val_te = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=best_m)
    return pd.DataFrame({'ref_te_x_slope': oof_te * tr['session_slope_8w'].fillna(0).values}), \
           pd.DataFrame({'ref_te_x_slope': val_te * vl['session_slope_8w'].fillna(0).values})

test_exp2_features(gen_ref_slope_only, "4c. ref_TE x Slope Only")

print("\nV10 exploration complete.")
