import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

print("=" * 80)
print("V13 PHASES C, D, E: RESIDUAL AUDIT, SEGMENTxMONTH DIAGNOSTICS & LOMO")
print("=" * 80)

# 1. Load Data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

train_df['referral_code'] = train_raw['referral_code'].values
train_df['billing_cycle'] = train_raw['billing_cycle'].values
train_df['signup_date'] = train_raw['signup_date'].values
train_df['snapshot_date'] = train_raw['snapshot_date'].values

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
        stats_grp = tr_part.groupby('cat')['y'].agg(['count', 'mean'])
        smoothed = (stats_grp['count'] * stats_grp['mean'] + smoothing * inner_prior) / (stats_grp['count'] + smoothing)
        oof_te[val_idx] = val_part['cat'].map(smoothed).fillna(inner_prior).values
        
    full_stats = train_data.groupby('cat')['y'].agg(['count', 'mean'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
    val_te = val_s.map(full_smoothed).fillna(global_mean).values.astype(np.float32)
    return oof_te, val_te

def get_v9_features_for_fold(tr_sub, y_tr, vl_sub):
    tr_ref = tr_sub['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl_sub['referral_code'].fillna('NO_REF').astype(str)
    ref_counts = tr_ref.value_counts()
    
    oof_te_ref, val_te_ref = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=20.0)
    
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

cv_splits = [
    ("Fold 1 (Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28', '2026-02-28'),
    ("Fold 2 (Mar)",       train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31', '2026-03-31'),
    ("Fold 3 (Apr)",       train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30', '2026-04-30'),
]

seeds = [42, 123, 314, 555, 777]
hp_d4 = {'depth': 4, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}

# Generate Full V9 Validation Predictions
print("\n[Step 1] Generating V9 Validation Predictions (45,000 Validation Rows across Feb, Mar, Apr)...")
val_dfs = []

for fold_name, tr_mask, vl_mask, month_str in cv_splits:
    t0 = time.time()
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    # 5-seed CatBoost
    cb_preds = []
    for s in seeds:
        m = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        cb_preds.append(m.predict_proba(X_vl)[:, 1])
    cb_ens = np.mean(cb_preds, axis=0)
    
    # Logistic Regression
    num_cols = [c for c in v7_features if c not in cat_names]
    log_pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    log_pipe.fit(tr_sub[v7_features], y_tr)
    lr_pred = log_pipe.predict_proba(vl_sub[v7_features])[:, 1]
    
    v9_pred = 0.88 * cb_ens + 0.12 * lr_pred
    vl_sub['v9_pred'] = v9_pred
    vl_sub['residual'] = vl_sub['churn'] - v9_pred
    vl_sub['val_month'] = month_str
    
    auc_f = roc_auc_score(y_vl, v9_pred)
    print(f"  {fold_name}: AUC = {auc_f:.5f} (computed in {time.time() - t0:.1f}s)")
    val_dfs.append(vl_sub)

oof_df = pd.concat(val_dfs, axis=0).reset_index(drop=True)
print(f"Total validation rows collected: {len(oof_df)}")

# ===========================================================================
# PHASE C: V9 RESIDUAL & STRUCTURE AUDIT
# ===========================================================================
print("\n" + "=" * 80)
print("PHASE C: V9 OOF RESIDUAL & STRUCTURE AUDIT")
print("=" * 80)

# Enrich oof_df with candidate grouping variables
oof_df['signup_date'] = pd.to_datetime(oof_df['signup_date'])
oof_df['signup_year'] = oof_df['signup_date'].dt.year
oof_df['signup_month'] = oof_df['signup_date'].dt.month
oof_df['signup_week'] = oof_df['signup_date'].dt.isocalendar().week
oof_df['has_referral'] = oof_df['referral_code'].notnull().astype(int)
oof_df['ref_prefix'] = oof_df['referral_code'].fillna('NONE').str[:3]
oof_df['is_pm_missing'] = oof_df['payment_method'].isnull().astype(int)
oof_df['event_count_bin'] = pd.cut(oof_df['total_events'].fillna(0), bins=[-1, 0, 5, 15, 30, 100, 9999], labels=['0', '1-5', '6-15', '16-30', '31-100', '100+'])

candidate_groups = [
    'billing_cycle',
    'plan',
    'region',
    'acquisition_channel',
    'payment_method',
    'age_band',
    'has_referral',
    'is_pm_missing',
    'signup_year',
    'event_count_bin',
    'val_month'
]

residual_findings = {}

print("\n--- Summary of Systematic Residual Biases by Group ---")
for grp_col in candidate_groups:
    grp_stats = oof_df.groupby(grp_col).agg(
        size=('residual', 'count'),
        positive_count=('churn', 'sum'),
        mean_target=('churn', 'mean'),
        mean_pred=('v9_pred', 'mean'),
        mean_residual=('residual', 'mean'),
        std_residual=('residual', 'std')
    ).reset_index()
    
    # Compute z-score: mean_residual / (std / sqrt(size))
    grp_stats['z_stat'] = grp_stats['mean_residual'] / (grp_stats['std_residual'] / np.sqrt(grp_stats['size']) + 1e-6)
    grp_stats['significant'] = grp_stats['z_stat'].abs() > 3.0 # p < 0.003
    
    print(f"\nGrouping: [{grp_col}]")
    for _, row in grp_stats.iterrows():
        val = row[grp_col]
        n = int(row['size'])
        y_m = row['mean_target']
        p_m = row['mean_pred']
        res = row['mean_residual']
        z = row['z_stat']
        sig_str = "*** BIAS ***" if abs(z) > 3.0 else ""
        print(f"  {str(val):18s} | N={n:5d} | Target={y_m:.4f} | Pred={p_m:.4f} | Res={res:+.4f} (z={z:+.2f}) {sig_str}")
        
    residual_findings[grp_col] = grp_stats.to_dict(orient='records')

# ===========================================================================
# PHASE D: SEGMENT x MONTH DIAGNOSTICS
# ===========================================================================
print("\n" + "=" * 80)
print("PHASE D: SEGMENT x MONTH DIAGNOSTICS")
print("=" * 80)

months = ['2026-02-28', '2026-03-31', '2026-04-30']
month_names = ['Feb (Fold 1)', 'Mar (Fold 2)', 'Apr (Fold 3)']

# Monthly Positive Rate vs Predicted Positive Rate (Calibration check)
print("\n--- 1. Monthly Churn Calibration & Intrinsic AUC ---")
calib_table = []
for m_str, m_name in zip(months, month_names):
    sub = oof_df[oof_df['val_month'] == m_str]
    actual_r = sub['churn'].mean()
    pred_r = sub['v9_pred'].mean()
    m_auc = roc_auc_score(sub['churn'], sub['v9_pred'])
    calib_table.append({'month': m_name, 'size': len(sub), 'actual_churn_rate': actual_r, 'pred_churn_rate': pred_r, 'auc': m_auc})
    print(f"  {m_name:15s} | N={len(sub)} | Actual Churn: {actual_r:.4f} | Pred Churn: {pred_r:.4f} | AUC: {m_auc:.5f}")

# Segment x Month AUC Breakdown
print("\n--- 2. Segment x Month AUC Matrix ---")
segment_auc_results = {}

for seg_col in ['billing_cycle', 'plan', 'acquisition_channel', 'region', 'has_referral']:
    print(f"\nSegment: [{seg_col}] AUC by Month:")
    cat_vals = oof_df[seg_col].unique()
    seg_dict = {}
    for cv in cat_vals:
        row_str = f"  {str(cv):18s}"
        fold_aucs = []
        for m_str in months:
            sub = oof_df[(oof_df['val_month'] == m_str) & (oof_df[seg_col] == cv)]
            if len(sub['churn'].unique()) > 1:
                auc_val = roc_auc_score(sub['churn'], sub['v9_pred'])
                fold_aucs.append(auc_val)
                row_str += f" | {m_str[-5:]}: {auc_val:.4f}"
            else:
                fold_aucs.append(np.nan)
                row_str += f" | {m_str[-5:]}:   N/A "
        row_str += f" | Mean: {np.nanmean(fold_aucs):.4f}"
        print(row_str)
        seg_dict[str(cv)] = [float(x) for x in fold_aucs]
    segment_auc_results[seg_col] = seg_dict

# ===========================================================================
# PHASE E: LEAVE-ONE-MONTH-OUT (LOMO) DIAGNOSTIC FOR MARCH
# ===========================================================================
print("\n" + "=" * 80)
print("PHASE E: LEAVE-ONE-MONTH-OUT (LOMO) DIAGNOSTIC FOR MARCH")
print("=" * 80)

# In standard Fold 2: Train is Jan-Feb (<= 2026-02-28), Val is Mar (2026-03-31) -> AUC = 0.86258
# Let's test training variations targeting March:
# E1: Jan alone -> Mar
# E2: Feb alone -> Mar
# E3: Full LOMO: Jan + Feb + Apr -> Mar (non-causal upper bound test)

mar_sub = train_df[train_df['snapshot_date'] == '2026-03-31']
y_mar = mar_sub['churn'].values

lomo_configs = [
    ("E1: Jan alone -> Mar",               train_df['snapshot_date'] == '2026-01-31'),
    ("E2: Feb alone -> Mar",               train_df['snapshot_date'] == '2026-02-28'),
    ("Standard Fold 2: Jan-Feb -> Mar",    train_df['snapshot_date'] <= '2026-02-28'),
    ("E3: LOMO (Jan + Feb + Apr) -> Mar",  train_df['snapshot_date'] != '2026-03-31')
]

lomo_results = {}
for name, tr_mask in lomo_configs:
    tr_sub = train_df[tr_mask]
    y_tr = tr_sub['churn'].values
    
    X_tr, X_mar = get_v9_features_for_fold(tr_sub, y_tr, mar_sub)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
    m.fit(X_tr, y_tr, eval_set=(X_mar, y_mar))
    auc_res = float(roc_auc_score(y_mar, m.predict_proba(X_mar)[:, 1]))
    lomo_results[name] = auc_res
    print(f"  {name:38s} -> March AUC: {auc_res:.5f}")

# Save results of C, D, E
out_cde = {
    'residual_findings': residual_findings,
    'monthly_calibration': calib_table,
    'segment_month_auc': segment_auc_results,
    'lomo_march_results': lomo_results
}

os.makedirs('v13', exist_ok=True)
with open('v13/v13_phases_c_d_e.json', 'w') as f:
    json.dump(out_cde, f, indent=2, default=str)

print("\nPhases C, D, E complete! Saved to v13/v13_phases_c_d_e.json")
