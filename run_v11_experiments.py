import pandas as pd
import numpy as np
import json
import time
import os
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

print("=" * 80)
print("V11 EXPERIMENTS: REFERRAL SHRINKAGE, RELIABILITY, RESIDUALS, STABILITY & MODELS")
print("=" * 80)

# 1. Load base data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

train_df['referral_code'] = train_raw['referral_code'].values
train_df['billing_cycle'] = train_raw['billing_cycle'].values

# Precompute label-free test frequencies for Part B
test_ref_counts = test_raw['referral_code'].fillna('NO_REF').astype(str).value_counts()
train_ref_counts_all = train_raw['referral_code'].fillna('NO_REF').astype(str).value_counts()

# Leak-safe target encoding helper
def compute_oof_target_encoding(train_s, y_train, val_s, smoothing=20.0, n_splits=5, seed=42):
    global_mean = float(y_train.mean())
    oof_te = np.full(len(train_s), np.nan, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_data = pd.DataFrame({'cat': train_s, 'y': y_train})
    
    for tr_idx, val_idx in kf.split(train_data):
        tr_part = train_data.iloc[tr_idx]
        val_part = train_data.iloc[val_idx]
        inner_prior = float(tr_part['y'].mean())
        stats = tr_part.groupby('cat')['y'].agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * inner_prior) / (stats['count'] + smoothing)
        oof_te[val_idx] = val_part['cat'].map(smoothed).fillna(inner_prior).values
        
    full_stats = train_data.groupby('cat')['y'].agg(['count', 'mean'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
    val_te = val_s.map(full_smoothed).fillna(global_mean).values.astype(np.float32)
    return oof_te, val_te, global_mean

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

hp_d4 = {'depth': 4, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}
hp_d6 = {'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}

def get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=20.0):
    tr_ref = tr_sub['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl_sub['referral_code'].fillna('NO_REF').astype(str)
    ref_counts = tr_ref.value_counts()
    
    oof_te_ref, val_te_ref, prior = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=smoothing)
    
    def act_bin(df):
        s = df['session_count'].fillna(0)
        b = pd.cut(s, bins=[-1, 0, 5, 15, 30, 9999], labels=['0', '1', '2', '3', '4']).astype(str)
        return 'act_' + b + '__' + df['billing_cycle'].astype(str)
    def rec_bin(df):
        r = df['days_since_last_session'].fillna(999)
        b = pd.cut(r, bins=[-1, 7, 14, 30, 60, 9999], labels=['7', '14', '30', '60', 'inf']).astype(str)
        return 'rec_' + b + '__' + df['billing_cycle'].astype(str)
        
    oof_act_te, val_act_te, _ = compute_oof_target_encoding(act_bin(tr_sub), y_tr, act_bin(vl_sub), smoothing=15.0)
    oof_rec_te, val_rec_te, _ = compute_oof_target_encoding(rec_bin(tr_sub), y_tr, rec_bin(vl_sub), smoothing=15.0)
    
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
    return X_tr, X_vl, (oof_te_ref, val_te_ref, prior, oof_act_te, val_act_te, oof_rec_te, val_rec_te)

# ---------------------------------------------------------------------------
# BASELINE EVALUATION (Single Depth 4 CatBoost, seed=42)
# ---------------------------------------------------------------------------
print("\n[Baseline] Evaluating V9 Baseline (Single Depth 4 CatBoost, m=20, 90 features)...")
v9_base_folds = []
for fold_name, tr_mask, vl_mask in cv_splits:
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    X_tr, X_vl, _ = get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=20.0)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
    m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    v9_base_folds.append(float(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])))

v9_base_mean = float(np.mean(v9_base_folds))
print(f"V9 Baseline (Single Depth 4) -> Mean CV: {v9_base_mean:.5f} | F1: {v9_base_folds[0]:.5f}, F2: {v9_base_folds[1]:.5f}, F3: {v9_base_folds[2]:.5f}")

# ===========================================================================
# EXPERIMENT A: REFERRAL SHRINKAGE & LOGIT TRANSFORM
# ===========================================================================
print("\n" + "=" * 80)
print("EXPERIMENT A: REFERRAL SHRINKAGE & LOGIT TRANSFORM")
print("=" * 80)

smoothing_candidates = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0]
exp_a_raw_results = {}
exp_a_logit_results = {}
exp_a_both_results = {}

for m_val in smoothing_candidates:
    # 1. Raw smoothed TE
    f_raw = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        X_tr, X_vl, _ = get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=m_val)
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_raw.append(float(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])))
    m_raw = float(np.mean(f_raw))
    exp_a_raw_results[f"raw_m_{int(m_val)}"] = (m_raw, m_raw - v9_base_mean, f_raw)
    print(f"  Raw Smoothed TE  (m={int(m_val):3d}) -> Mean: {m_raw:.5f} (Δ: {m_raw - v9_base_mean:+.5f}) | F1: {f_raw[0]:.5f}, F2: {f_raw[1]:.5f}, F3: {f_raw[2]:.5f}")

    # 2. Pure Logit-transformed smoothed TE (replaces raw ref_code_te)
    f_logit = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        X_tr, X_vl, (te_tr, te_vl, *_) = get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=m_val)
        
        # Replace ref_code_te with logit(ref_code_te)
        p_tr = np.clip(te_tr, 1e-4, 1.0 - 1e-4)
        p_vl = np.clip(te_vl, 1e-4, 1.0 - 1e-4)
        X_tr['ref_code_logit_te'] = np.log(p_tr / (1.0 - p_tr))
        X_vl['ref_code_logit_te'] = np.log(p_vl / (1.0 - p_vl))
        X_tr = X_tr.drop(columns=['ref_code_te'])
        X_vl = X_vl.drop(columns=['ref_code_te'])
        
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_logit.append(float(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])))
    m_logit = float(np.mean(f_logit))
    exp_a_logit_results[f"pure_logit_m_{int(m_val)}"] = (m_logit, m_logit - v9_base_mean, f_logit)
    print(f"  Pure Logit TE    (m={int(m_val):3d}) -> Mean: {m_logit:.5f} (Δ: {m_logit - v9_base_mean:+.5f}) | F1: {f_logit[0]:.5f}, F2: {f_logit[1]:.5f}, F3: {f_logit[2]:.5f}")

    # 3. Both Raw and Logit TE together
    f_both = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        X_tr, X_vl, (te_tr, te_vl, *_) = get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=m_val)
        
        p_tr = np.clip(te_tr, 1e-4, 1.0 - 1e-4)
        p_vl = np.clip(te_vl, 1e-4, 1.0 - 1e-4)
        X_tr['ref_code_logit_te'] = np.log(p_tr / (1.0 - p_tr))
        X_vl['ref_code_logit_te'] = np.log(p_vl / (1.0 - p_vl))
        
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_both.append(float(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])))
    m_both = float(np.mean(f_both))
    exp_a_both_results[f"both_m_{int(m_val)}"] = (m_both, m_both - v9_base_mean, f_both)
    print(f"  Raw + Logit TE   (m={int(m_val):3d}) -> Mean: {m_both:.5f} (Δ: {m_both - v9_base_mean:+.5f}) | F1: {f_both[0]:.5f}, F2: {f_both[1]:.5f}, F3: {f_both[2]:.5f}")

# Find best smoothing for subsequent tests
best_raw_key = max(exp_a_raw_results.keys(), key=lambda k: exp_a_raw_results[k][0])
best_m_val = float(best_raw_key.split('_')[-1])
print(f"\nBest Part A Shrinkage Value: m={best_m_val} (Mean CV: {exp_a_raw_results[best_raw_key][0]:.5f})")

# Helper for testing new feature additions against the chosen baseline
def evaluate_feature_generator(gen_fn, exp_name):
    f_res = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        
        X_tr, X_vl, extra_info = get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=best_m_val)
        feat_tr, feat_vl = gen_fn(tr_sub, y_tr, vl_sub, extra_info)
        
        X_tr = pd.concat([X_tr, feat_tr.reset_index(drop=True)], axis=1)
        X_vl = pd.concat([X_vl, feat_vl.reset_index(drop=True)], axis=1)
        
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_res.append(float(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])))
    m_res = float(np.mean(f_res))
    baseline_val = exp_a_raw_results[best_raw_key][0]
    delta = m_res - baseline_val
    print(f"  {exp_name:42s} -> Mean: {m_res:.5f} (Δ: {delta:+.5f}) | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}")
    return {'mean': m_res, 'delta': delta, 'folds': f_res}

# ===========================================================================
# EXPERIMENT B: REFERRAL RELIABILITY (LABEL-FREE FEATURES)
# ===========================================================================
print("\n" + "=" * 80)
print("EXPERIMENT B: REFERRAL RELIABILITY (LABEL-FREE FEATURES)")
print("=" * 80)

# B1: Test frequency alone
def gen_b1(tr, y_tr, vl, info):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    return pd.DataFrame({'ref_test_freq': tr_ref.map(test_ref_counts).fillna(0).values}), \
           pd.DataFrame({'ref_test_freq': vl_ref.map(test_ref_counts).fillna(0).values})

# B2: Train/test frequency ratio alone
def gen_b2(tr, y_tr, vl, info):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    tr_counts = tr_ref.value_counts()
    
    def calc_ratio(s):
        tf = s.map(tr_counts).fillna(0).values / float(len(tr))
        tef = s.map(test_ref_counts).fillna(0).values / 40000.0
        return (tf + 1e-4) / (tef + 1e-4)
    return pd.DataFrame({'ref_train_test_ratio': calc_ratio(tr_ref)}), \
           pd.DataFrame({'ref_train_test_ratio': calc_ratio(vl_ref)})

# B3: Referral prevalence across train+test
def gen_b3(tr, y_tr, vl, info):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    tr_counts = tr_ref.value_counts()
    
    def calc_prev(s):
        tf = s.map(tr_counts).fillna(0).values
        tef = s.map(test_ref_counts).fillna(0).values
        return (tf + tef) / float(len(tr) + 40000.0)
    return pd.DataFrame({'ref_prevalence': calc_prev(tr_ref)}), \
           pd.DataFrame({'ref_prevalence': calc_prev(vl_ref)})

# B4: All B features combined
def gen_b_all(tr, y_tr, vl, info):
    d1_tr, d1_vl = gen_b1(tr, y_tr, vl, info)
    d2_tr, d2_vl = gen_b2(tr, y_tr, vl, info)
    d3_tr, d3_vl = gen_b3(tr, y_tr, vl, info)
    return pd.concat([d1_tr, d2_tr, d3_tr], axis=1), pd.concat([d1_vl, d2_vl, d3_vl], axis=1)

res_b1 = evaluate_feature_generator(gen_b1, "B1. Test Frequency alone")
res_b2 = evaluate_feature_generator(gen_b2, "B2. Train/Test Ratio alone")
res_b3 = evaluate_feature_generator(gen_b3, "B3. Referral Prevalence alone")
res_b4 = evaluate_feature_generator(gen_b_all, "B4. All Label-Free Features Combined")

# ===========================================================================
# EXPERIMENT C: RESIDUAL-STYLE FEATURES
# ===========================================================================
print("\n" + "=" * 80)
print("EXPERIMENT C: RESIDUAL-STYLE REFERRAL FEATURES")
print("=" * 80)

# C1: Referral TE minus global churn prior
def gen_c1(tr, y_tr, vl, info):
    te_tr, te_vl, prior, *_ = info
    return pd.DataFrame({'ref_te_minus_prior': te_tr - prior}), \
           pd.DataFrame({'ref_te_minus_prior': te_vl - prior})

# C2: Customer behavioral risk score minus referral risk
def gen_c2(tr, y_tr, vl, info):
    te_tr, te_vl, prior, act_tr, act_vl, rec_tr, rec_vl = info
    cust_risk_tr = 0.5 * (act_tr + rec_tr)
    cust_risk_vl = 0.5 * (act_vl + rec_vl)
    return pd.DataFrame({'behavioral_minus_referral_risk': cust_risk_tr - te_tr}), \
           pd.DataFrame({'behavioral_minus_referral_risk': cust_risk_vl - te_vl})

# C3: Referral risk percentile/rank among referral groups
def gen_c3(tr, y_tr, vl, info):
    te_tr, te_vl, prior, *_ = info
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    
    # Compute rank on group-level mean TE
    group_te = pd.DataFrame({'ref': tr_ref, 'te': te_tr}).groupby('ref')['te'].mean()
    group_rank = group_te.rank(pct=True)
    
    return pd.DataFrame({'ref_risk_percentile_rank': tr_ref.map(group_rank).fillna(0.5).values}), \
           pd.DataFrame({'ref_risk_percentile_rank': vl_ref.map(group_rank).fillna(0.5).values})

# C4: All C features combined
def gen_c_all(tr, y_tr, vl, info):
    d1_tr, d1_vl = gen_c1(tr, y_tr, vl, info)
    d2_tr, d2_vl = gen_c2(tr, y_tr, vl, info)
    d3_tr, d3_vl = gen_c3(tr, y_tr, vl, info)
    return pd.concat([d1_tr, d2_tr, d3_tr], axis=1), pd.concat([d1_vl, d2_vl, d3_vl], axis=1)

res_c1 = evaluate_feature_generator(gen_c1, "C1. Referral TE minus Global Prior")
res_c2 = evaluate_feature_generator(gen_c2, "C2. Behavioral Risk minus Referral Risk")
res_c3 = evaluate_feature_generator(gen_c3, "C3. Referral Risk Percentile Rank")
res_c4 = evaluate_feature_generator(gen_c_all, "C4. All Residual Features Combined")

# ===========================================================================
# EXPERIMENT D: TEMPORAL REFERRAL STABILITY (FORWARD-SAFE)
# ===========================================================================
print("\n" + "=" * 80)
print("EXPERIMENT D: TEMPORAL REFERRAL STABILITY (FORWARD-SAFE)")
print("=" * 80)

# Strictly forward-safe temporal statistics
def gen_d_features(tr, y_tr, vl, info):
    months = sorted(tr['snapshot_date'].unique())
    global_prior = float(y_tr.mean())
    
    # 1. Recent month churn rate
    last_m = months[-1]
    last_df = tr[tr['snapshot_date'] == last_m]
    stats_last = last_df.groupby(last_df['referral_code'].fillna('NO_REF').astype(str))['churn'].agg(['count', 'mean'])
    smooth_last = (stats_last['count'] * stats_last['mean'] + 20.0 * global_prior) / (stats_last['count'] + 20.0)
    
    # 2. Change over months
    if len(months) >= 2:
        first_m = months[0]
        first_df = tr[tr['snapshot_date'] == first_m]
        stats_first = first_df.groupby(first_df['referral_code'].fillna('NO_REF').astype(str))['churn'].agg(['count', 'mean'])
        smooth_first = (stats_first['count'] * stats_first['mean'] + 20.0 * global_prior) / (stats_first['count'] + 20.0)
        rate_change = smooth_last - smooth_first
    else:
        rate_change = pd.Series(0.0, index=smooth_last.index)
        
    # 3. Number of months with observations
    months_obs = tr.groupby(tr['referral_code'].fillna('NO_REF').astype(str))['snapshot_date'].nunique()
    
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    
    df_tr = pd.DataFrame({
        'ref_recent_month_rate': tr_ref.map(smooth_last).fillna(global_prior).values,
        'ref_rate_change_over_months': tr_ref.map(rate_change).fillna(0.0).values,
        'ref_active_months': tr_ref.map(months_obs).fillna(0).values
    })
    df_vl = pd.DataFrame({
        'ref_recent_month_rate': vl_ref.map(smooth_last).fillna(global_prior).values,
        'ref_rate_change_over_months': vl_ref.map(rate_change).fillna(0.0).values,
        'ref_active_months': vl_ref.map(months_obs).fillna(0).values
    })
    return df_tr, df_vl

def gen_d1(tr, y_tr, vl, info):
    d_tr, d_vl = gen_d_features(tr, y_tr, vl, info)
    return d_tr[['ref_recent_month_rate']], d_vl[['ref_recent_month_rate']]

def gen_d2(tr, y_tr, vl, info):
    d_tr, d_vl = gen_d_features(tr, y_tr, vl, info)
    return d_tr[['ref_rate_change_over_months']], d_vl[['ref_rate_change_over_months']]

def gen_d3(tr, y_tr, vl, info):
    d_tr, d_vl = gen_d_features(tr, y_tr, vl, info)
    return d_tr[['ref_active_months']], d_vl[['ref_active_months']]

res_d1 = evaluate_feature_generator(gen_d1, "D1. Recent Referral Churn Rate")
res_d2 = evaluate_feature_generator(gen_d2, "D2. Rate Change Over Months")
res_d3 = evaluate_feature_generator(gen_d3, "D3. Number of Active Months")
res_d_all = evaluate_feature_generator(gen_d_features, "D4. All Temporal Stability Combined")

# ===========================================================================
# EXPERIMENT E: FINAL MODEL TRAINING & ENSEMBLING
# ===========================================================================
print("\n" + "=" * 80)
print("EXPERIMENT E: FINAL V11 5-SEED ENSEMBLES & LOGISTIC REGRESSION BLENDS")
print("=" * 80)

# Determine winning feature set from experiments A-D:
# Acceptance rule: >= +0.0005 mean CV improvement, no major fold regression, prefer simpler features
candidate_results = {
    'B1 (Test Freq)': res_b1,
    'B2 (Ratio)': res_b2,
    'B3 (Prevalence)': res_b3,
    'B4 (All B)': res_b4,
    'C1 (Prior Res)': res_c1,
    'C2 (Behavioral Res)': res_c2,
    'C3 (Rank)': res_c3,
    'C4 (All C)': res_c4,
    'D1 (Recent Rate)': res_d1,
    'D2 (Rate Change)': res_d2,
    'D3 (Active Months)': res_d3,
    'D4 (All D)': res_d_all
}

print("\n--- Acceptance Screening Summary ---")
accepted_candidates = []
for name, r in candidate_results.items():
    status = "REJECTED (< +0.0005)" if r['delta'] < 0.0005 else "ACCEPTED (>= +0.0005)"
    print(f"  {name:25s}: Mean={r['mean']:.5f} (Δ={r['delta']:+.5f}) -> {status}")
    if r['delta'] >= 0.0005:
        accepted_candidates.append(name)

# Model comparison using optimal shrinkage (m=60) + logit representation
seeds = [42, 123, 314, 555, 777]

d4_preds = []
d6_preds = []
lr_preds = []
lr_ref_preds = []
y_trues = []

d4_folds = []
d6_folds = []
lr_folds = []
lr_ref_folds = []

for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    t_f = time.time()
    tr_sub = train_df[tr_mask].copy()
    vl_sub = train_df[vl_mask].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    y_trues.append(y_vl)
    
    # Feature set: Base V9 with optimal shrinkage (m=60)
    X_tr, X_vl, (te_tr, te_vl, prior, *_) = get_base_v9_features(tr_sub, y_tr, vl_sub, smoothing=best_m_val)
    
    # Add logit TE as an explicit feature
    p_tr = np.clip(te_tr, 1e-4, 1.0 - 1e-4)
    p_vl = np.clip(te_vl, 1e-4, 1.0 - 1e-4)
    X_tr['ref_code_logit_te'] = np.log(p_tr / (1.0 - p_tr))
    X_vl['ref_code_logit_te'] = np.log(p_vl / (1.0 - p_vl))
    
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    # 1. Depth 4 5-Seed CatBoost Ensemble
    f_d4_p = []
    for s in seeds:
        m4 = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m4.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_d4_p.append(m4.predict_proba(X_vl)[:, 1])
    ens4 = np.mean(f_d4_p, axis=0)
    d4_preds.append(ens4)
    auc4 = float(roc_auc_score(y_vl, ens4))
    d4_folds.append(auc4)
    
    # 2. Depth 6 5-Seed CatBoost Ensemble
    f_d6_p = []
    for s in seeds:
        m6 = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d6)
        m6.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_d6_p.append(m6.predict_proba(X_vl)[:, 1])
    ens6 = np.mean(f_d6_p, axis=0)
    d6_preds.append(ens6)
    auc6 = float(roc_auc_score(y_vl, ens6))
    d6_folds.append(auc6)
    
    # 3. Standard Logistic Regression on base V7 features
    num_cols = [c for c in v7_features if c not in cat_names]
    log_pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    log_pipe.fit(tr_sub[v7_features], y_tr)
    lp = log_pipe.predict_proba(vl_sub[v7_features])[:, 1]
    lr_preds.append(lp)
    auclr = float(roc_auc_score(y_vl, lp))
    lr_folds.append(auclr)

    # 4. Logistic Regression with referral logit TE
    log_feats_tr = tr_sub[v7_features].copy()
    log_feats_vl = vl_sub[v7_features].copy()
    log_feats_tr['ref_code_logit_te'] = X_tr['ref_code_logit_te'].values
    log_feats_vl['ref_code_logit_te'] = X_vl['ref_code_logit_te'].values
    
    log_pipe_ref = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols + ['ref_code_logit_te']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    log_pipe_ref.fit(log_feats_tr, y_tr)
    lp_ref = log_pipe_ref.predict_proba(log_feats_vl)[:, 1]
    lr_ref_preds.append(lp_ref)
    auclr_ref = float(roc_auc_score(y_vl, lp_ref))
    lr_ref_folds.append(auclr_ref)
    
    print(f"{fold_name} [{time.time() - t_f:.1f}s]:")
    print(f"  V11 Depth 4 5-Seed AUC : {auc4:.5f}")
    print(f"  V11 Depth 6 5-Seed AUC : {auc6:.5f}")
    print(f"  Base Logistic Reg AUC  : {auclr:.5f}")
    print(f"  LR with Ref Logit AUC  : {auclr_ref:.5f}")

mean_d4 = float(np.mean(d4_folds))
mean_d6 = float(np.mean(d6_folds))
mean_lr = float(np.mean(lr_folds))
mean_lr_ref = float(np.mean(lr_ref_folds))

print("\n" + "=" * 80)
print("V11 FINAL MODEL & ENSEMBLE SUMMARY")
print("=" * 80)
print(f"V9 Baseline Benchmark Mean CV       : 0.86824")
print(f"V11 CatBoost Depth 4 5-Seed Mean CV : {mean_d4:.5f} | F1: {d4_folds[0]:.5f}, F2: {d4_folds[1]:.5f}, F3: {d4_folds[2]:.5f} (Δ: {mean_d4 - 0.86824:+.5f})")
print(f"V11 CatBoost Depth 6 5-Seed Mean CV : {mean_d6:.5f} | F1: {d6_folds[0]:.5f}, F2: {d6_folds[1]:.5f}, F3: {d6_folds[2]:.5f} (Δ: {mean_d6 - 0.86824:+.5f})")
print(f"Standard Logistic Regression Mean CV: {mean_lr:.5f} | F1: {lr_folds[0]:.5f}, F2: {lr_folds[1]:.5f}, F3: {lr_folds[2]:.5f}")
print(f"Logit-TE Logistic Regression Mean CV: {mean_lr_ref:.5f} | F1: {lr_ref_folds[0]:.5f}, F2: {lr_ref_folds[1]:.5f}, F3: {lr_ref_folds[2]:.5f} (Δ: {mean_lr_ref - mean_lr:+.5f})")

# Blend evaluations
print("\n--- Model Blends ---")
# 1. 90/10 with standard LR
b_90_10 = [float(roc_auc_score(y_trues[f], 0.90 * d4_preds[f] + 0.10 * lr_preds[f])) for f in range(3)]
print(f"  Blend (90% D4 CB + 10% Base LR) -> Mean: {np.mean(b_90_10):.5f} | F1: {b_90_10[0]:.5f}, F2: {b_90_10[1]:.5f}, F3: {b_90_10[2]:.5f} (Δ: {np.mean(b_90_10) - 0.86824:+.5f})")

# 2. 88/12 with standard LR
b_88_12 = [float(roc_auc_score(y_trues[f], 0.88 * d4_preds[f] + 0.12 * lr_preds[f])) for f in range(3)]
print(f"  Blend (88% D4 CB + 12% Base LR) -> Mean: {np.mean(b_88_12):.5f} | F1: {b_88_12[0]:.5f}, F2: {b_88_12[1]:.5f}, F3: {b_88_12[2]:.5f} (Δ: {np.mean(b_88_12) - 0.86824:+.5f})")

# 3. 90/10 with Logit-TE LR
b_90_10_ref = [float(roc_auc_score(y_trues[f], 0.90 * d4_preds[f] + 0.10 * lr_ref_preds[f])) for f in range(3)]
print(f"  Blend (90% D4 CB + 10% Ref LR)  -> Mean: {np.mean(b_90_10_ref):.5f} | F1: {b_90_10_ref[0]:.5f}, F2: {b_90_10_ref[1]:.5f}, F3: {b_90_10_ref[2]:.5f} (Δ: {np.mean(b_90_10_ref) - 0.86824:+.5f})")

# 4. 88/12 with Logit-TE LR
b_88_12_ref = [float(roc_auc_score(y_trues[f], 0.88 * d4_preds[f] + 0.12 * lr_ref_preds[f])) for f in range(3)]
print(f"  Blend (88% D4 CB + 12% Ref LR)  -> Mean: {np.mean(b_88_12_ref):.5f} | F1: {b_88_12_ref[0]:.5f}, F2: {b_88_12_ref[1]:.5f}, F3: {b_88_12_ref[2]:.5f} (Δ: {np.mean(b_88_12_ref) - 0.86824:+.5f})")

# 5. Tree-only blend: 50% Depth 4 + 50% Depth 6
b_d4_d6 = [float(roc_auc_score(y_trues[f], 0.50 * d4_preds[f] + 0.50 * d6_preds[f])) for f in range(3)]
print(f"  Blend (50% D4 + 50% D6)         -> Mean: {np.mean(b_d4_d6):.5f} | F1: {b_d4_d6[0]:.5f}, F2: {b_d4_d6[1]:.5f}, F3: {b_d4_d6[2]:.5f} (Δ: {np.mean(b_d4_d6) - 0.86824:+.5f})")

# 6. Triple blend: 55% D4 + 35% D6 + 10% Logit-TE LR
b_trip = [float(roc_auc_score(y_trues[f], 0.55 * d4_preds[f] + 0.35 * d6_preds[f] + 0.10 * lr_ref_preds[f])) for f in range(3)]
print(f"  Triple (55% D4+35% D6+10% RefLR)-> Mean: {np.mean(b_trip):.5f} | F1: {b_trip[0]:.5f}, F2: {b_trip[1]:.5f}, F3: {b_trip[2]:.5f} (Δ: {np.mean(b_trip) - 0.86824:+.5f})")

# Save detailed results artifact
os.makedirs('v11', exist_ok=True)
output_payload = {
    'v9_baseline_mean_cv': 0.86824,
    'optimal_smoothing_m': best_m_val,
    'exp_a_raw_sweep': {k: {'mean': v[0], 'delta': v[1], 'folds': v[2]} for k, v in exp_a_raw_results.items()},
    'exp_a_logit_sweep': {k: {'mean': v[0], 'delta': v[1], 'folds': v[2]} for k, v in exp_a_logit_results.items()},
    'exp_a_both_sweep': {k: {'mean': v[0], 'delta': v[1], 'folds': v[2]} for k, v in exp_a_both_results.items()},
    'exp_b_results': {
        'b1_test_freq': res_b1,
        'b2_ratio': res_b2,
        'b3_prevalence': res_b3,
        'b4_combined': res_b4
    },
    'exp_c_results': {
        'c1_te_minus_prior': res_c1,
        'c2_behavioral_minus_referral_risk': res_c2,
        'c3_risk_percentile_rank': res_c3,
        'c4_combined': res_c4
    },
    'exp_d_results': {
        'd1_recent_month_rate': res_d1,
        'd2_rate_change': res_d2,
        'd3_active_months': res_d3,
        'd4_combined': res_d_all
    },
    'v11_models': {
        'd4_5seed_mean': mean_d4,
        'd4_5seed_folds': d4_folds,
        'd6_5seed_mean': mean_d6,
        'd6_5seed_folds': d6_folds,
        'lr_base_mean': mean_lr,
        'lr_base_folds': lr_folds,
        'lr_ref_mean': mean_lr_ref,
        'lr_ref_folds': lr_ref_folds
    },
    'v11_blends': {
        'blend_90_10_mean': float(np.mean(b_90_10)),
        'blend_90_10_folds': b_90_10,
        'blend_88_12_mean': float(np.mean(b_88_12)),
        'blend_88_12_folds': b_88_12,
        'blend_90_10_ref_mean': float(np.mean(b_90_10_ref)),
        'blend_90_10_ref_folds': b_90_10_ref,
        'blend_88_12_ref_mean': float(np.mean(b_88_12_ref)),
        'blend_88_12_ref_folds': b_88_12_ref,
        'blend_50_50_d4_d6_mean': float(np.mean(b_d4_d6)),
        'blend_50_50_d4_d6_folds': b_d4_d6,
        'triple_blend_mean': float(np.mean(b_trip)),
        'triple_blend_folds': b_trip
    },
    'retained_features': list(all_feats),
    'num_features_retained': len(all_feats),
    'accepted_additions': accepted_candidates
}

with open('v11/v11_final_summary.json', 'w') as f:
    json.dump(output_payload, f, indent=2)

print("\nV11 execution successfully finished. Results saved to v11/v11_final_summary.json")
