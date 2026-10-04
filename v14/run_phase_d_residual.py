import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("V14 PHASE D: LEAKAGE-SAFE RESIDUAL / SEGMENT CORRECTION")
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

# Load saved predictions from Phase B & C
if not os.path.exists('v14/v14_oof_preds.npz'):
    raise FileNotFoundError("v14/v14_oof_preds.npz not found. Run Phase B & C first.")

oof_data = np.load('v14/v14_oof_preds.npz', allow_pickle=True)
val_y = [oof_data['y_true_f1'], oof_data['y_true_f2'], oof_data['y_true_f3']]
val_p_glob = [oof_data['p_glob_f1'], oof_data['p_glob_f2'], oof_data['p_glob_f3']]
val_p_exp = [oof_data['p_exp_f1'], oof_data['p_exp_f2'], oof_data['p_exp_f3']]
val_regions = [oof_data['reg_f1'], oof_data['reg_f2'], oof_data['reg_f3']]
val_annual_m = [oof_data['ann_m_f1'], oof_data['ann_m_f2'], oof_data['ann_m_f3']]

v9_champion_mean = 0.86824
v13_depth3_mean = 0.8683525

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

print("Generating training-only OOF predictions to estimate leakage-safe residual effects...")

fold_training_effects = []

for f_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    t_f = time.time()
    tr_sub = train_df[tr_mask].copy().reset_index(drop=True)
    vl_sub = train_df[vl_mask].copy().reset_index(drop=True)
    y_tr = tr_sub['churn'].values
    
    # 5-fold CV on training set ONLY to get leak-safe p_hat_tr
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    p_tr_oof = np.zeros(len(y_tr), dtype=np.float64)
    
    X_tr_fold, _ = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    all_feats = list(X_tr_fold.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    for in_tr_idx, in_vl_idx in kf_inner.split(X_tr_fold):
        in_X_tr = X_tr_fold.iloc[in_tr_idx]
        in_y_tr = y_tr[in_tr_idx]
        in_X_vl = X_tr_fold.iloc[in_vl_idx]
        in_y_vl = y_tr[in_vl_idx]
        
        m_inner = CatBoostClassifier(iterations=1000, random_seed=42, cat_features=c_idx, early_stopping_rounds=50, use_best_model=True, **hp)
        m_inner.fit(in_X_tr, in_y_tr, eval_set=(in_X_vl, in_y_vl))
        p_tr_oof[in_vl_idx] = m_inner.predict_proba(in_X_vl)[:, 1]
        
    residuals = y_tr - p_tr_oof
    
    # Compute group effects from training residuals ONLY
    is_central_tr = (tr_sub['region'] == 'central').values
    res_central = float(np.mean(residuals[is_central_tr]))
    res_non_central = float(np.mean(residuals[~is_central_tr]))
    delta_central = res_central - res_non_central  # Expected to be negative (overpredicted)
    
    is_annual_tr = (tr_sub['billing_cycle'] == 'annual').values
    res_annual = float(np.mean(residuals[is_annual_tr]))
    res_monthly = float(np.mean(residuals[~is_annual_tr]))
    delta_annual = res_annual - res_monthly  # Expected to be negative (overpredicted)
    
    fold_training_effects.append({
        'fold': f_idx + 1,
        'res_central': res_central,
        'res_non_central': res_non_central,
        'delta_central': delta_central,
        'res_annual': res_annual,
        'res_monthly': res_monthly,
        'delta_annual': delta_annual
    })
    
    print(f"  {fold_name} Training Residual Effects [{time.time() - t_f:.1f}s]:")
    print(f"    Central Residual: {res_central:+.5f} vs Non-Central: {res_non_central:+.5f} -> Delta: {delta_central:+.5f}")
    print(f"    Annual Residual : {res_annual:+.5f} vs Monthly:     {res_monthly:+.5f} -> Delta: {delta_annual:+.5f}")

# Evaluate residual correction strengths
strengths = [0.0, 0.25, 0.50, 0.75, 1.00]
modes = ['central_only', 'annual_only', 'joint']

results_residual = {}

print("\nEvaluating residual corrections on Global Depth-3 predictions...")
for mode in modes:
    results_residual[mode] = []
    print(f"\n--- Strategy: {mode.upper()} ---")
    for alpha in strengths:
        fold_scores = []
        for f in range(3):
            p_orig = val_p_glob[f].copy()
            y_v = val_y[f]
            is_central_vl = (val_regions[f] == 'central')
            is_annual_vl = val_annual_m[f]
            
            eff = fold_training_effects[f]
            
            p_adj = p_orig.copy()
            if mode in ['central_only', 'joint']:
                # Add alpha * delta_central to central rows (shifting downwards if delta < 0)
                p_adj[is_central_vl] += alpha * eff['delta_central']
            if mode in ['annual_only', 'joint']:
                # Add alpha * delta_annual to annual rows (shifting downwards if delta < 0)
                p_adj[is_annual_vl] += alpha * eff['delta_annual']
                
            p_adj = np.clip(p_adj, 0.0, 1.0)
            score = float(roc_auc_score(y_v, p_adj))
            fold_scores.append(score)
            
        m_auc = float(np.mean(fold_scores))
        delta_v9 = m_auc - v9_champion_mean
        delta_d3 = m_auc - v13_depth3_mean
        
        row = {
            'mode': mode,
            'strength': alpha,
            'f1': fold_scores[0],
            'f2': fold_scores[1],
            'f3': fold_scores[2],
            'mean_cv': m_auc,
            'delta_vs_v9': delta_v9,
            'delta_vs_depth3': delta_d3
        }
        results_residual[mode].append(row)
        print(f"  alpha = {alpha:.2f} -> Mean CV: {m_auc:.5f} | F1: {fold_scores[0]:.5f}, F2: {fold_scores[1]:.5f}, F3: {fold_scores[2]:.5f} | Δ(D3): {delta_d3:+.5f}")

# Also test on best Expert Blend (e.g. 50/50 or whatever is tested)
print("\nEvaluating residual corrections on 50/50 Global + Expert Blend...")
blend_50 = [0.5 * val_p_glob[f] + 0.5 * val_p_exp[f] for f in range(3)]
blend_results_residual = {}

for mode in modes:
    blend_results_residual[mode] = []
    for alpha in strengths:
        fold_scores = []
        for f in range(3):
            p_orig = blend_50[f].copy()
            y_v = val_y[f]
            is_central_vl = (val_regions[f] == 'central')
            is_annual_vl = val_annual_m[f]
            
            eff = fold_training_effects[f]
            
            p_adj = p_orig.copy()
            if mode in ['central_only', 'joint']:
                p_adj[is_central_vl] += alpha * eff['delta_central']
            if mode in ['annual_only', 'joint']:
                p_adj[is_annual_vl] += alpha * eff['delta_annual']
                
            p_adj = np.clip(p_adj, 0.0, 1.0)
            score = float(roc_auc_score(y_v, p_adj))
            fold_scores.append(score)
            
        m_auc = float(np.mean(fold_scores))
        delta_v9 = m_auc - v9_champion_mean
        delta_d3 = m_auc - v13_depth3_mean
        
        row = {
            'mode': mode,
            'strength': alpha,
            'f1': fold_scores[0],
            'f2': fold_scores[1],
            'f3': fold_scores[2],
            'mean_cv': m_auc,
            'delta_vs_v9': delta_v9,
            'delta_vs_depth3': delta_d3
        }
        blend_results_residual[mode].append(row)

# Save results
output_payload = {
    'training_effects': fold_training_effects,
    'global_depth3_corrections': results_residual,
    'expert_blend_corrections': blend_results_residual,
    'runtime_seconds': time.time() - t0
}

with open('v14/v14_residual_results.json', 'w') as f:
    json.dump(output_payload, f, indent=2)

print(f"\nPhase D execution complete in {time.time() - t0:.1f}s.")
print("Results saved to v14/v14_residual_results.json.")
