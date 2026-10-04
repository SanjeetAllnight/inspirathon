import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("V13 PHASE B: CATBOOST DEPTH DIAGNOSTIC (DEPTH 1, 2, 3, 4)")
print("=" * 80)

# Load base data
t_start = time.time()
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

train_df['referral_code'] = train_raw['referral_code'].values
train_df['billing_cycle'] = train_raw['billing_cycle'].values

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

v9_baseline_mean = 0.86824
v9_baseline_single_d4 = 0.86722

depth_levels = [1, 2, 3, 4]
results_by_depth = {}

# Evaluate single-seed first, then 5-seed
seeds = [42, 123, 314, 555, 777]

for depth in depth_levels:
    t_d = time.time()
    hp = {
        'depth': depth,
        'learning_rate': 0.04,
        'l2_leaf_reg': 1,
        'random_strength': 1,
        'bagging_temperature': 1,
        'eval_metric': 'AUC',
        'verbose': 0,
        'thread_count': -1
    }
    
    fold_single_aucs = []
    fold_ens_aucs = []
    
    for f_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        
        X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        
        # 5 seeds
        seed_preds = []
        for s in seeds:
            m = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp)
            m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
            seed_preds.append(m.predict_proba(X_vl)[:, 1])
        
        fold_single_aucs.append(float(roc_auc_score(y_vl, seed_preds[0])))
        ens_pred = np.mean(seed_preds, axis=0)
        fold_ens_aucs.append(float(roc_auc_score(y_vl, ens_pred)))
        
    duration = time.time() - t_d
    mean_single = float(np.mean(fold_single_aucs))
    mean_ens = float(np.mean(fold_ens_aucs))
    delta_vs_v9 = mean_ens - v9_baseline_mean
    
    results_by_depth[f"depth_{depth}"] = {
        'depth': depth,
        'single_seed_mean': mean_single,
        'single_seed_folds': fold_single_aucs,
        'ensemble_mean': mean_ens,
        'ensemble_folds': fold_ens_aucs,
        'delta_vs_v9_champion': delta_vs_v9,
        'runtime_seconds': duration
    }
    
    print(f"\n--- CatBoost Depth {depth} Results [{duration:.1f}s] ---")
    print(f"  Single Seed (42) -> Mean CV: {mean_single:.5f} | F1: {fold_single_aucs[0]:.5f}, F2: {fold_single_aucs[1]:.5f}, F3: {fold_single_aucs[2]:.5f}")
    print(f"  5-Seed Ensemble  -> Mean CV: {mean_ens:.5f} | F1: {fold_ens_aucs[0]:.5f}, F2: {fold_ens_aucs[1]:.5f}, F3: {fold_ens_aucs[2]:.5f} (Δ: {delta_vs_v9:+.5f})")

# Determine signal nature based on guidance
gap_d1_d4 = results_by_depth['depth_4']['ensemble_mean'] - results_by_depth['depth_1']['ensemble_mean']
gap_d2_d4 = results_by_depth['depth_4']['ensemble_mean'] - results_by_depth['depth_2']['ensemble_mean']

if gap_d2_d4 <= 0.0005:
    hypothesis = "SMOOTH/ADDITIVE (depth 1/2 is close to depth 4, signal is largely additive)"
    recommendation = "V14-A: Smooth-model branch (spline-GAM, small regularized MLP, blend vs V9)"
else:
    hypothesis = "INTERACTION/STRUCTURE (depth 4 materially beats shallow depths, nonlinear interactions are critical)"
    recommendation = "V14-B: Interaction/structure branch (billing-cycle experts, lifecycle/renewal structure, residual/random-effects)"

print("\n" + "=" * 80)
print("PHASE B SUMMARY & INTERPRETATION")
print("=" * 80)
print(f"Gap Depth 4 vs Depth 1 (Stumps)       : {gap_d1_d4:+.5f}")
print(f"Gap Depth 4 vs Depth 2 (Pairs)         : {gap_d2_d4:+.5f}")
print(f"Empirical Signal Characterization     : {hypothesis}")
print(f"Prescribed Direction for V14           : {recommendation}")

output_payload = {
    'depth_sweep': results_by_depth,
    'gap_d4_minus_d1': gap_d1_d4,
    'gap_d4_minus_d2': gap_d2_d4,
    'signal_characterization': hypothesis,
    'recommended_direction': recommendation
}

os.makedirs('v13', exist_ok=True)
with open('v13/v13_phase_b_depth.json', 'w') as f:
    json.dump(output_payload, f, indent=2)

print("\nPhase B Depth Diagnostic finished. Saved to v13/v13_phase_b_depth.json")
