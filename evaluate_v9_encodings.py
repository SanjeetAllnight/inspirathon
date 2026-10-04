import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

print("=" * 80)
print("V9: EVALUATING LEAK-SAFE TARGET & FREQUENCY ENCODINGS")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
c_idx_base = [v7_features.index(c) for c in cat_names if c in v7_features]

# Add raw columns needed for interaction creation
raw_cols = ['referral_code', 'plan', 'billing_cycle', 'region', 'acquisition_channel', 'signup_date']
for col in raw_cols:
    if col not in train_df.columns:
        train_df[col] = train_raw[col].values

# Target encoding helper
def compute_oof_target_encoding(train_s, y_train, val_s, smoothing=15.0, n_splits=5, seed=42):
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

# Baseline CatBoost hyperparameters (Depth 4, lr=0.04)
hp_d4 = {
    'depth': 4,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0
}

# Baseline evaluation function
def eval_baseline(feature_list, hp, seed=42):
    c_idx = [feature_list.index(c) for c in cat_names if c in feature_list]
    fold_aucs = []
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        m = CatBoostClassifier(iterations=1500, random_seed=seed, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp)
        m.fit(tr_sub[feature_list], tr_sub['churn'], eval_set=(vl_sub[feature_list], vl_sub['churn']))
        preds = m.predict_proba(vl_sub[feature_list])[:, 1]
        fold_aucs.append(roc_auc_score(vl_sub['churn'], preds))
    return fold_aucs, np.mean(fold_aucs)

print("\n[Baseline] Evaluating V7 Baseline (Depth 4, 86 features)...")
base_folds, base_mean = eval_baseline(v7_features, hp_d4)
print(f"V7 Baseline -> Mean CV: {base_mean:.5f} | F1: {base_folds[0]:.5f}, F2: {base_folds[1]:.5f}, F3: {base_folds[2]:.5f}")

# Function to evaluate dynamic in-fold feature encodings
def eval_dynamic_encodings(encoding_generator, block_name, smoothing=15.0):
    t_blk = time.time()
    fold_aucs = []
    
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask].copy()
        vl_sub = train_df[vl_mask].copy()
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        
        # Generate new features dynamically strictly on training set
        new_feats_tr, new_feats_vl, new_col_names = encoding_generator(tr_sub, y_tr, vl_sub, smoothing)
        
        X_tr = pd.concat([tr_sub[v7_features].reset_index(drop=True), new_feats_tr.reset_index(drop=True)], axis=1)
        X_vl = pd.concat([vl_sub[v7_features].reset_index(drop=True), new_feats_vl.reset_index(drop=True)], axis=1)
        
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        preds = m.predict_proba(X_vl)[:, 1]
        auc = roc_auc_score(y_vl, preds)
        fold_aucs.append(auc)
        
    m_res = np.mean(fold_aucs)
    delta = m_res - base_mean
    status = "*ACCEPTED*" if delta >= 0.0003 else ("FLAT" if abs(delta) < 0.0003 else "DEGRADED")
    print(f"\n{block_name:38s} [{time.time() - t_blk:.1f}s]:")
    print(f"  Mean CV: {m_res:.5f} (Δ: {delta:+.5f}) [{status}] | F1: {fold_aucs[0]:.5f}, F2: {fold_aucs[1]:.5f}, F3: {fold_aucs[2]:.5f}")
    return fold_aucs, m_res, delta

# ---------------------------------------------------------------------------
# BLOCK 1: REFERRAL CODE ENCODINGS
# ---------------------------------------------------------------------------
def gen_block_1(tr, y_tr, vl, smoothing=20.0):
    tr_ref = tr['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl['referral_code'].fillna('NO_REF').astype(str)
    
    # 1. Frequency encoding strictly from train
    ref_counts = tr_ref.value_counts()
    tr_freq = tr_ref.map(ref_counts).fillna(0)
    vl_freq = vl_ref.map(ref_counts).fillna(0)
    
    # 2. Leak-safe smoothed target encoding
    oof_te, val_te = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=smoothing)
    
    cols = ['ref_code_train_freq', 'ref_code_te']
    df_tr = pd.DataFrame({'ref_code_train_freq': tr_freq.values, 'ref_code_te': oof_te})
    df_vl = pd.DataFrame({'ref_code_train_freq': vl_freq.values, 'ref_code_te': val_te})
    return df_tr, df_vl, cols

f1, m1, d1 = eval_dynamic_encodings(gen_block_1, "Block 1: referral_code (freq + TE)")

# ---------------------------------------------------------------------------
# BLOCK 2: PROFILE-GROUP INTERACTIONS
# ---------------------------------------------------------------------------
def gen_block_2(tr, y_tr, vl, smoothing=15.0):
    pairs = [
        ('plan', 'billing_cycle'),
        ('plan', 'region'),
        ('billing_cycle', 'region'),
        ('acquisition_channel', 'plan'),
        ('plan', 'billing_cycle', 'region')
    ]
    df_tr = pd.DataFrame()
    df_vl = pd.DataFrame()
    col_names = []
    
    for p in pairs:
        tag = '_x_'.join(p)
        if len(p) == 2:
            s_tr = tr[p[0]].astype(str) + '__' + tr[p[1]].astype(str)
            s_vl = vl[p[0]].astype(str) + '__' + vl[p[1]].astype(str)
        else:
            s_tr = tr[p[0]].astype(str) + '__' + tr[p[1]].astype(str) + '__' + tr[p[2]].astype(str)
            s_vl = vl[p[0]].astype(str) + '__' + vl[p[1]].astype(str) + '__' + vl[p[2]].astype(str)
            
        # Frequency encoding
        counts = s_tr.value_counts(normalize=True)
        col_freq = f'{tag}_freq'
        df_tr[col_freq] = s_tr.map(counts).fillna(0).values
        df_vl[col_freq] = s_vl.map(counts).fillna(0).values
        col_names.append(col_freq)
        
        # Target encoding
        col_te = f'{tag}_te'
        oof_te, val_te = compute_oof_target_encoding(s_tr, y_tr, s_vl, smoothing=smoothing)
        df_tr[col_te] = oof_te
        df_vl[col_te] = val_te
        col_names.append(col_te)
        
    return df_tr, df_vl, col_names

f2, m2, d2 = eval_dynamic_encodings(gen_block_2, "Block 2: Profile-Group Interactions")

# ---------------------------------------------------------------------------
# BLOCK 3: SIGNUP COHORTS
# ---------------------------------------------------------------------------
def gen_block_3(tr, y_tr, vl, smoothing=15.0):
    tr_date = pd.to_datetime(tr['signup_date'])
    vl_date = pd.to_datetime(vl['signup_date'])
    
    s_tr = tr_date.dt.year.astype(str) + '_' + tr_date.dt.month.astype(str)
    s_vl = vl_date.dt.year.astype(str) + '_' + vl_date.dt.month.astype(str)
    
    counts = s_tr.value_counts(normalize=True)
    oof_te, val_te = compute_oof_target_encoding(s_tr, y_tr, s_vl, smoothing=smoothing)
    
    df_tr = pd.DataFrame({
        'signup_cohort_freq': s_tr.map(counts).fillna(0).values,
        'signup_cohort_te': oof_te
    })
    df_vl = pd.DataFrame({
        'signup_cohort_freq': s_vl.map(counts).fillna(0).values,
        'signup_cohort_te': val_te
    })
    return df_tr, df_vl, ['signup_cohort_freq', 'signup_cohort_te']

f3, m3, d3 = eval_dynamic_encodings(gen_block_3, "Block 3: Signup Cohorts (freq + TE)")

# ---------------------------------------------------------------------------
# BLOCK 4: EVENT-DERIVED COARSE GROUPS X BILLING CYCLE
# ---------------------------------------------------------------------------
def gen_block_4(tr, y_tr, vl, smoothing=15.0):
    # Coarse activity levels: 0 (no sess), 1 (1-5), 2 (6-15), 3 (16-30), 4 (>30)
    def act_bin(df):
        s = df['session_count'].fillna(0)
        b = pd.cut(s, bins=[-1, 0, 5, 15, 30, 9999], labels=['0', '1', '2', '3', '4']).astype(str)
        return 'act_' + b + '__' + df['billing_cycle'].astype(str)
        
    def rec_bin(df):
        r = df['days_since_last_session'].fillna(999)
        b = pd.cut(r, bins=[-1, 7, 14, 30, 60, 9999], labels=['7', '14', '30', '60', 'inf']).astype(str)
        return 'rec_' + b + '__' + df['billing_cycle'].astype(str)
        
    act_tr, act_vl = act_bin(tr), act_bin(vl)
    rec_tr, rec_vl = rec_bin(tr), rec_bin(vl)
    
    oof_act_te, val_act_te = compute_oof_target_encoding(act_tr, y_tr, act_vl, smoothing=smoothing)
    oof_rec_te, val_rec_te = compute_oof_target_encoding(rec_tr, y_tr, rec_vl, smoothing=smoothing)
    
    df_tr = pd.DataFrame({
        'act_bin_billing_te': oof_act_te,
        'rec_bin_billing_te': oof_rec_te,
    })
    df_vl = pd.DataFrame({
        'act_bin_billing_te': val_act_te,
        'rec_bin_billing_te': val_rec_te,
    })
    return df_tr, df_vl, ['act_bin_billing_te', 'rec_bin_billing_te']

f4, m4, d4 = eval_dynamic_encodings(gen_block_4, "Block 4: Event Coarse Groups x Billing")

# ---------------------------------------------------------------------------
# BLOCK COMBINATION & LOGISTIC REGRESSION BLEND INVESTIGATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("BLOCK EVALUATION SUMMARY")
print("=" * 80)
print(f"{'V7 Baseline':35s} -> Mean: {base_mean:.5f}")
print(f"{'Block 1: referral_code':35s} -> Mean: {m1:.5f} (Δ: {d1:+.5f})")
print(f"{'Block 2: Profile Interactions':35s} -> Mean: {m2:.5f} (Δ: {d2:+.5f})")
print(f"{'Block 3: Signup Cohorts':35s} -> Mean: {m3:.5f} (Δ: {d3:+.5f})")
print(f"{'Block 4: Coarse Groups x Billing':35s} -> Mean: {m4:.5f} (Δ: {d4:+.5f})")

# Combined positive blocks
pos_blocks = []
if d1 > 0: pos_blocks.append(gen_block_1)
if d2 > 0: pos_blocks.append(gen_block_2)
if d3 > 0: pos_blocks.append(gen_block_3)
if d4 > 0: pos_blocks.append(gen_block_4)

def gen_combined(tr, y_tr, vl, smoothing=15.0):
    dfs_tr, dfs_vl, cols = [], [], []
    for gen in pos_blocks:
        t_tr, t_vl, c = gen(tr, y_tr, vl, smoothing)
        dfs_tr.append(t_tr)
        dfs_vl.append(t_vl)
        cols.extend(c)
    if not dfs_tr:
        return pd.DataFrame(), pd.DataFrame(), []
    return pd.concat(dfs_tr, axis=1), pd.concat(dfs_vl, axis=1), cols

if pos_blocks:
    f_comb, m_comb, d_comb = eval_dynamic_encodings(gen_combined, "Combined Positive Encoding Blocks")
else:
    print("\nNo encoding blocks produced positive delta on CatBoost Depth 4.")
    f_comb, m_comb, d_comb = base_folds, base_mean, 0.0

# Now test whether these encodings improve Logistic Regression!
print("\n--- Testing Whether Encodings Boost Regularized Logistic Regression ---")
lr_base_folds = []
lr_enc_folds = []

for fold_name, tr_mask, vl_mask in cv_splits:
    tr_sub = train_df[tr_mask].copy()
    vl_sub = train_df[vl_mask].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    
    # 1. Base LR on V7
    num_cols_base = [c for c in v7_features if c not in cat_names]
    pipe_base = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols_base),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    pipe_base.fit(tr_sub[v7_features], y_tr)
    lp_base = pipe_base.predict_proba(vl_sub[v7_features])[:, 1]
    lr_base_folds.append(roc_auc_score(y_vl, lp_base))
    
    # 2. LR with all profile interaction & TE features (Block 2 + Block 4 + Block 1)
    df_tr_all, df_vl_all, enc_cols = gen_block_2(tr_sub, y_tr, vl_sub, smoothing=15.0)
    d4_tr, d4_vl, c4 = gen_block_4(tr_sub, y_tr, vl_sub, smoothing=15.0)
    df_tr_all = pd.concat([df_tr_all, d4_tr], axis=1)
    df_vl_all = pd.concat([df_vl_all, d4_vl], axis=1)
    enc_cols.extend(c4)
    
    X_tr_enc = pd.concat([tr_sub[v7_features].reset_index(drop=True), df_tr_all.reset_index(drop=True)], axis=1)
    X_vl_enc = pd.concat([vl_sub[v7_features].reset_index(drop=True), df_vl_all.reset_index(drop=True)], axis=1)
    
    num_cols_enc = [c for c in X_tr_enc.columns if c not in cat_names]
    pipe_enc = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols_enc),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    pipe_enc.fit(X_tr_enc, y_tr)
    lp_enc = pipe_enc.predict_proba(X_vl_enc)[:, 1]
    lr_enc_folds.append(roc_auc_score(y_vl, lp_enc))

print(f"Logistic Regression Standalone Base Mean CV : {np.mean(lr_base_folds):.5f} | F1: {lr_base_folds[0]:.5f}, F2: {lr_base_folds[1]:.5f}, F3: {lr_base_folds[2]:.5f}")
print(f"Logistic Regression with Target Encodings    : {np.mean(lr_enc_folds):.5f} | F1: {lr_enc_folds[0]:.5f}, F2: {lr_enc_folds[1]:.5f}, F3: {lr_enc_folds[2]:.5f} (Δ: {np.mean(lr_enc_folds) - np.mean(lr_base_folds):+.5f})")

# Save results
import os
os.makedirs('v9', exist_ok=True)
with open('v9/v9_screening_results.json', 'w') as f:
    json.dump({
        'base_mean': base_mean,
        'base_folds': base_folds,
        'block_1_referral': {'mean': m1, 'delta': d1, 'folds': f1},
        'block_2_interactions': {'mean': m2, 'delta': d2, 'folds': f2},
        'block_3_signup_cohort': {'mean': m3, 'delta': d3, 'folds': f3},
        'block_4_coarse_groups': {'mean': m4, 'delta': d4, 'folds': f4},
        'combined': {'mean': m_comb, 'delta': d_comb, 'folds': f_comb},
        'lr_base_mean': float(np.mean(lr_base_folds)),
        'lr_enc_mean': float(np.mean(lr_enc_folds)),
        'lr_delta': float(np.mean(lr_enc_folds) - np.mean(lr_base_folds))
    }, f, indent=2)

print("\nV9 screening completed and saved to v9/v9_screening_results.json")
