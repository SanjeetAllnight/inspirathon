import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("V14 PHASE B & C: BILLING-CYCLE SPECIALIZED EXPERTS & BLENDING")
print("=" * 80)

t0 = time.time()

# 1. Load data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

train_df['referral_code'] = train_raw['referral_code'].values
train_df['billing_cycle'] = train_raw['billing_cycle'].values
train_df['region'] = train_raw['region'].values

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
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

v9_champion_mean = 0.86824
v13_depth3_mean = 0.8683525
seeds = [42, 123, 314, 555, 777]

hp = {
    'depth': 3,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0,
    'thread_count': -1
}

# Storage for predictions across folds
fold_y_true = []
fold_global_preds = []
fold_expert_preds = []
fold_monthly_masks = []
fold_annual_masks = []
fold_regions = []

# Metrics storage
global_fold_aucs = []
expert_fold_aucs = []

global_monthly_aucs = []
global_annual_aucs = []
expert_monthly_aucs = []
expert_annual_aucs = []

print(f"Starting training across {len(cv_splits)} temporal folds with 5 seeds per model...")

for f_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    t_f = time.time()
    print(f"\n>>> Processing {fold_name}...")
    
    tr_sub = train_df[tr_mask].copy()
    vl_sub = train_df[vl_mask].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    # Billing masks
    tr_monthly = (tr_sub['billing_cycle'] == 'monthly').values
    tr_annual = (tr_sub['billing_cycle'] == 'annual').values
    vl_monthly = (vl_sub['billing_cycle'] == 'monthly').values
    vl_annual = (vl_sub['billing_cycle'] == 'annual').values
    
    print(f"  Train size: {len(X_tr)} (Monthly: {tr_monthly.sum()}, Annual: {tr_annual.sum()})")
    print(f"  Val size:   {len(X_vl)} (Monthly: {vl_monthly.sum()}, Annual: {vl_annual.sum()})")
    
    # 1. Global Depth-3
    seed_preds_global = []
    # 2. Monthly Expert
    seed_preds_monthly = []
    # 3. Annual Expert
    seed_preds_annual = []
    
    for s_idx, s in enumerate(seeds):
        # Global
        m_glob = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp)
        m_glob.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        seed_preds_global.append(m_glob.predict_proba(X_vl)[:, 1])
        
        # Monthly expert
        m_mth = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp)
        m_mth.fit(X_tr.iloc[tr_monthly], y_tr[tr_monthly], eval_set=(X_vl.iloc[vl_monthly], y_vl[vl_monthly]))
        seed_preds_monthly.append(m_mth.predict_proba(X_vl.iloc[vl_monthly])[:, 1])
        
        # Annual expert
        m_ann = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp)
        m_ann.fit(X_tr.iloc[tr_annual], y_tr[tr_annual], eval_set=(X_vl.iloc[vl_annual], y_vl[vl_annual]))
        seed_preds_annual.append(m_ann.predict_proba(X_vl.iloc[vl_annual])[:, 1])
        
    p_glob = np.mean(seed_preds_global, axis=0)
    p_mth = np.mean(seed_preds_monthly, axis=0)
    p_ann = np.mean(seed_preds_annual, axis=0)
    
    # Construct combined expert prediction
    p_expert = np.zeros(len(y_vl), dtype=np.float64)
    p_expert[vl_monthly] = p_mth
    p_expert[vl_annual] = p_ann
    
    # Overall AUCs
    auc_glob = float(roc_auc_score(y_vl, p_glob))
    auc_exp = float(roc_auc_score(y_vl, p_expert))
    
    # Segment AUCs
    glob_mth_auc = float(roc_auc_score(y_vl[vl_monthly], p_glob[vl_monthly]))
    glob_ann_auc = float(roc_auc_score(y_vl[vl_annual], p_glob[vl_annual]))
    exp_mth_auc = float(roc_auc_score(y_vl[vl_monthly], p_expert[vl_monthly]))
    exp_ann_auc = float(roc_auc_score(y_vl[vl_annual], p_expert[vl_annual]))
    
    global_fold_aucs.append(auc_glob)
    expert_fold_aucs.append(auc_exp)
    global_monthly_aucs.append(glob_mth_auc)
    global_annual_aucs.append(glob_ann_auc)
    expert_monthly_aucs.append(exp_mth_auc)
    expert_annual_aucs.append(exp_ann_auc)
    
    # Store for global blending and residual analysis
    fold_y_true.append(y_vl)
    fold_global_preds.append(p_glob)
    fold_expert_preds.append(p_expert)
    fold_monthly_masks.append(vl_monthly)
    fold_annual_masks.append(vl_annual)
    fold_regions.append(vl_sub['region'].values)
    
    print(f"  Fold {f_idx+1} completed in {time.time() - t_f:.1f}s")
    print(f"    Global Depth-3  : {auc_glob:.5f} (Monthly: {glob_mth_auc:.5f}, Annual: {glob_ann_auc:.5f})")
    print(f"    Combined Expert : {auc_exp:.5f} (Monthly: {exp_mth_auc:.5f}, Annual: {exp_ann_auc:.5f})")

# Mean metrics
mean_glob = float(np.mean(global_fold_aucs))
mean_exp = float(np.mean(expert_fold_aucs))

print("\n" + "=" * 80)
print("PHASE B SUMMARY RESULTS")
print("=" * 80)
print(f"Global Depth-3 Backbone : Mean CV = {mean_glob:.5f} | F1: {global_fold_aucs[0]:.5f}, F2: {global_fold_aucs[1]:.5f}, F3: {global_fold_aucs[2]:.5f}")
print(f"Combined Expert System  : Mean CV = {mean_exp:.5f} | F1: {expert_fold_aucs[0]:.5f}, F2: {expert_fold_aucs[1]:.5f}, F3: {expert_fold_aucs[2]:.5f}")
print(f"  Delta vs V9 Champion  : {mean_exp - v9_champion_mean:+.5f}")
print(f"  Delta vs Depth-3 Base : {mean_exp - mean_glob:+.5f}")
print("\nSegment Breakdown:")
print(f"  Monthly AUCs - Global: {global_monthly_aucs} (Mean: {np.mean(global_monthly_aucs):.5f})")
print(f"  Monthly AUCs - Expert: {expert_monthly_aucs} (Mean: {np.mean(expert_monthly_aucs):.5f})")
print(f"  Annual AUCs  - Global: {global_annual_aucs} (Mean: {np.mean(global_annual_aucs):.5f})")
print(f"  Annual AUCs  - Expert: {expert_annual_aucs} (Mean: {np.mean(expert_annual_aucs):.5f})")
print(f"  ANNUAL F2 (March) -> Global: {global_annual_aucs[1]:.5f} | Expert: {expert_annual_aucs[1]:.5f} (Diff: {expert_annual_aucs[1] - global_annual_aucs[1]:+.5f})")

print("\n" + "=" * 80)
print("PHASE C: EXPERT BLENDING SWEEP")
print("=" * 80)
blend_weights = [1.0, 0.90, 0.75, 0.50, 0.25, 0.0]
blend_results = []

for w in blend_weights:
    b_aucs = []
    b_ann_f2 = None
    b_mth_f2 = None
    
    for f in range(3):
        y_val = fold_y_true[f]
        p_b = w * fold_global_preds[f] + (1.0 - w) * fold_expert_preds[f]
        score = float(roc_auc_score(y_val, p_b))
        b_aucs.append(score)
        
        if f == 1:
            ann_m = fold_annual_masks[1]
            mth_m = fold_monthly_masks[1]
            b_ann_f2 = float(roc_auc_score(y_val[ann_m], p_b[ann_m]))
            b_mth_f2 = float(roc_auc_score(y_val[mth_m], p_b[mth_m]))
            
    m_score = float(np.mean(b_aucs))
    delta_v9 = m_score - v9_champion_mean
    delta_d3 = m_score - mean_glob
    
    res = {
        'weight_global': w,
        'weight_expert': round(1.0 - w, 2),
        'f1': b_aucs[0],
        'f2': b_aucs[1],
        'f3': b_aucs[2],
        'mean_cv': m_score,
        'delta_vs_v9': delta_v9,
        'delta_vs_depth3': delta_d3,
        'annual_f2': b_ann_f2,
        'monthly_f2': b_mth_f2
    }
    blend_results.append(res)
    print(f"Blend {int(w*100):3d} Global : {int((1-w)*100):3d} Expert -> Mean CV: {m_score:.5f} | F1: {b_aucs[0]:.5f}, F2: {b_aucs[1]:.5f}, F3: {b_aucs[2]:.5f} | Δ(V9): {delta_v9:+.5f} | Annual F2: {b_ann_f2:.5f}")

# Save artifacts
output_payload = {
    'phase_b_global_depth3': {
        'mean_cv': mean_glob,
        'folds': global_fold_aucs,
        'monthly_folds': global_monthly_aucs,
        'annual_folds': global_annual_aucs
    },
    'phase_b_combined_expert': {
        'mean_cv': mean_exp,
        'folds': expert_fold_aucs,
        'monthly_folds': expert_monthly_aucs,
        'annual_folds': expert_annual_aucs
    },
    'phase_c_blends': blend_results,
    'runtime_total_seconds': time.time() - t0
}

os.makedirs('v14', exist_ok=True)
with open('v14/v14_expert_results.json', 'w') as f:
    json.dump(output_payload, f, indent=2)

# Save arrays for Phase D
np.savez_compressed(
    'v14/v14_oof_preds.npz',
    y_true_f1=fold_y_true[0],
    y_true_f2=fold_y_true[1],
    y_true_f3=fold_y_true[2],
    p_glob_f1=fold_global_preds[0],
    p_glob_f2=fold_global_preds[1],
    p_glob_f3=fold_global_preds[2],
    p_exp_f1=fold_expert_preds[0],
    p_exp_f2=fold_expert_preds[1],
    p_exp_f3=fold_expert_preds[2],
    mth_m_f1=fold_monthly_masks[0],
    mth_m_f2=fold_monthly_masks[1],
    mth_m_f3=fold_monthly_masks[2],
    ann_m_f1=fold_annual_masks[0],
    ann_m_f2=fold_annual_masks[1],
    ann_m_f3=fold_annual_masks[2],
    reg_f1=fold_regions[0],
    reg_f2=fold_regions[1],
    reg_f3=fold_regions[2]
)

print(f"\nPhase B & C execution complete in {time.time() - t0:.1f}s.")
print("Results saved to v14/v14_expert_results.json and v14/v14_oof_preds.npz.")
