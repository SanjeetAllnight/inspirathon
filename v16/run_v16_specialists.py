import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

print("=" * 80)
print("V16 PIPELINE: FEATURE-VIEW TREE SPECIALISTS & DIVERSITY BLENDING")
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

# 2. Load V15 Baseline Champion Predictions
oof_v15 = np.load('v15/v15_oof_preds.npz')
val_y = [oof_v15['y_true_f1'], oof_v15['y_true_f2'], oof_v15['y_true_f3']]
v14_preds = [oof_v15['v14_pred_f1'], oof_v15['v14_pred_f2'], oof_v15['v14_pred_f3']]
lgb_preds = [oof_v15['lgb_pred_f1'], oof_v15['lgb_pred_f2'], oof_v15['lgb_pred_f3']]

# V15 Champion: 80% V14 + 20% LGBM (AUC = 0.86936)
v15_champion_preds = [0.80 * v14_preds[f] + 0.20 * lgb_preds[f] for f in range(3)]
v15_scores = [float(roc_auc_score(val_y[f], v15_champion_preds[f])) for f in range(3)]
v15_champion_mean = float(np.mean(v15_scores))
v9_champion_mean = 0.86824

print(f"Verified V15 Champion: Mean CV = {v15_champion_mean:.5f} | F1: {v15_scores[0]:.5f}, F2: {v15_scores[1]:.5f}, F3: {v15_scores[2]:.5f}")

# 3. Define Feature Views
enc_feats = ['ref_code_train_freq', 'ref_code_te', 'act_bin_billing_te', 'rec_bin_billing_te']
all_90 = v7_features + enc_feats

profile_billing_feats = [
    'plan', 'region', 'age_band', 'acquisition_channel', 'payment_method',
    'is_pm_missing', 'family_sharing', 'num_devices', 'billing_cycle',
    'monthly_price', 'price_per_device', 'tenure_x_billing', 'tenure_x_plan_basic',
    'account_age_days', 'account_age_months', 'signup_year', 'signup_dow',
    'act_bin_billing_te', 'rec_bin_billing_te'
]

referral_behavior_feats = [
    'referral_freq', 'ref_code_train_freq', 'ref_code_te', 'acquisition_channel',
    'days_since_last_session', 'session_count', 'total_events', 'active_days_90d',
    'active_days_120d', 'sess_accel', 'session_slope_8w', 'longest_gap_days',
    'days_since_last_event', 'sess_ratio_30_60', 'sess_ratio_14_60', 'support_count',
    'pf_per_sess_30d', 'act_bin_billing_te', 'rec_bin_billing_te', 'billing_cycle'
]

behavior_lifecycle_feats = [f for f in all_90 if f not in [
    'plan', 'region', 'age_band', 'acquisition_channel', 'payment_method',
    'is_pm_missing', 'family_sharing', 'num_devices', 'monthly_price', 'price_per_device'
]]

views = {
    'behavior_lifecycle': {
        'name': 'Behavior + Lifecycle Specialist',
        'features': behavior_lifecycle_feats
    },
    'profile_billing': {
        'name': 'Profile + Billing Specialist',
        'features': profile_billing_feats
    },
    'referral_behavior': {
        'name': 'Referral + Behavior Specialist',
        'features': referral_behavior_feats
    }
}

seeds = [42, 123, 314, 555, 777]

# 4. Train Specialists (Depth-3 and Depth-2)
specialist_models = {}

for v_key, v_info in views.items():
    feat_list = v_info['features']
    print(f"\n" + "=" * 60)
    print(f"Training Feature View: {v_info['name']} ({len(feat_list)} features)")
    print("=" * 60)
    
    for depth in [3, 2]:
        t_sp = time.time()
        tag = f"{v_key}_d{depth}"
        
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
        
        fold_preds = []
        fold_aucs = []
        
        for f, (fname, tr_m, vl_m) in enumerate(cv_splits):
            tr_sub = train_df[tr_m].copy()
            vl_sub = train_df[vl_m].copy()
            y_tr = tr_sub['churn'].values
            y_vl = vl_sub['churn'].values
            
            X_tr_full, X_vl_full = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
            X_tr_sp = X_tr_full[feat_list].copy()
            X_vl_sp = X_vl_full[feat_list].copy()
            
            sp_cats = [c for c in cat_names if c in feat_list]
            c_idx = [feat_list.index(c) for c in sp_cats]
            
            seed_preds = []
            for s in seeds:
                m = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp)
                m.fit(X_tr_sp, y_tr, eval_set=(X_vl_sp, y_vl))
                seed_preds.append(m.predict_proba(X_vl_sp)[:, 1])
                
            p_sp = np.mean(seed_preds, axis=0)
            fold_preds.append(p_sp)
            fold_aucs.append(float(roc_auc_score(y_vl, p_sp)))
            
        m_auc = float(np.mean(fold_aucs))
        delta_v15 = m_auc - v15_champion_mean
        delta_v9 = m_auc - v9_champion_mean
        
        # Diversity metrics
        concat_y = np.concatenate(val_y)
        concat_v15 = np.concatenate(v15_champion_preds)
        concat_sp = np.concatenate(fold_preds)
        
        p_corr = float(pearsonr(concat_v15, concat_sp)[0])
        s_corr = float(spearmanr(concat_v15, concat_sp)[0])
        
        res_v15 = concat_y - concat_v15
        res_sp = concat_y - concat_sp
        r_corr = float(pearsonr(res_v15, res_sp)[0])
        
        dur = time.time() - t_sp
        
        specialist_models[tag] = {
            'view': v_key,
            'name': f"{v_info['name']} (Depth {depth})",
            'depth': depth,
            'feature_count': len(feat_list),
            'mean_cv': m_auc,
            'folds': fold_aucs,
            'delta_vs_v15': delta_v15,
            'delta_vs_v9': delta_v9,
            'pearson_corr_v15': p_corr,
            'spearman_corr_v15': s_corr,
            'residual_corr_v15': r_corr,
            'runtime_seconds': dur,
            'fold_preds': fold_preds
        }
        
        print(f"[{dur:.1f}s] {v_info['name']} (Depth {depth}): Mean CV = {m_auc:.5f} | F1: {fold_aucs[0]:.5f}, F2: {fold_aucs[1]:.5f}, F3: {fold_aucs[2]:.5f} | Δ(V15): {delta_v15:+.5f} | P-Corr: {p_corr:.4f} | R-Corr: {r_corr:.4f}")

# 5. Phase F: Blend Feature-View Specialists with V15
print("\n" + "=" * 80)
print("PHASE F: BLENDING FEATURE-VIEW SPECIALISTS WITH V15 CHAMPION")
print("=" * 80)

blend_results = []

# Single specialist blends: V15 + [5%, 10%, 15%, 20%] specialist
for tag, sp_data in specialist_models.items():
    sp_preds = sp_data['fold_preds']
    for w_sp in [0.05, 0.10, 0.15, 0.20]:
        w_v15 = 1.0 - w_sp
        b_aucs = []
        for f in range(3):
            p_b = w_v15 * v15_champion_preds[f] + w_sp * sp_preds[f]
            b_aucs.append(float(roc_auc_score(val_y[f], p_b)))
            
        m_b = float(np.mean(b_aucs))
        delta_v15 = m_b - v15_champion_mean
        delta_v9 = m_b - v9_champion_mean
        
        b_res = {
            'name': f"V15 ({int(w_v15*100)}) + {sp_data['name']} ({int(w_sp*100)})",
            'specialist_tag': tag,
            'w_v15': w_v15,
            'w_specialist': w_sp,
            'f1': b_aucs[0],
            'f2': b_aucs[1],
            'f3': b_aucs[2],
            'mean_cv': m_b,
            'delta_vs_v15': delta_v15,
            'delta_vs_v9': delta_v9
        }
        blend_results.append(b_res)
        print(f"{b_res['name']:<55s} -> Mean CV: {m_b:.5f} | F1: {b_aucs[0]:.5f}, F2: {b_aucs[1]:.5f}, F3: {b_aucs[2]:.5f} | Δ(V15): {delta_v15:+.5f}")

# Multi-specialist combinations (e.g. V15 + Behavior/Lifecycle + Profile/Billing)
multi_combos = [
    # V15 + Profile/Billing (d3) + Behavior/Lifecycle (d3)
    (0.85, 'profile_billing_d3', 0.10, 'behavior_lifecycle_d3', 0.05),
    (0.85, 'profile_billing_d3', 0.05, 'behavior_lifecycle_d3', 0.10),
    (0.80, 'profile_billing_d3', 0.10, 'behavior_lifecycle_d3', 0.10),
    (0.85, 'profile_billing_d2', 0.10, 'referral_behavior_d3', 0.05),
    (0.85, 'profile_billing_d3', 0.10, 'referral_behavior_d3', 0.05)
]

for w_v15, tag_a, w_a, tag_b, w_b in multi_combos:
    b_aucs = []
    for f in range(3):
        p_b = w_v15 * v15_champion_preds[f] + w_a * specialist_models[tag_a]['fold_preds'][f] + w_b * specialist_models[tag_b]['fold_preds'][f]
        b_aucs.append(float(roc_auc_score(val_y[f], p_b)))
    m_b = float(np.mean(b_aucs))
    name = f"V15 ({int(w_v15*100)}) + {specialist_models[tag_a]['name']} ({int(w_a*100)}) + {specialist_models[tag_b]['name']} ({int(w_b*100)})"
    b_res = {
        'name': name,
        'specialist_tags': [tag_a, tag_b],
        'w_v15': w_v15,
        'w_a': w_a,
        'w_b': w_b,
        'f1': b_aucs[0],
        'f2': b_aucs[1],
        'f3': b_aucs[2],
        'mean_cv': m_b,
        'delta_vs_v15': m_b - v15_champion_mean,
        'delta_vs_v9': m_b - v9_champion_mean
    }
    blend_results.append(b_res)
    print(f"{name:<55s} -> Mean CV: {m_b:.5f} | F1: {b_aucs[0]:.5f}, F2: {b_aucs[1]:.5f}, F3: {b_aucs[2]:.5f} | Δ(V15): {b_res['delta_vs_v15']:+.5f}")

blend_results.sort(key=lambda x: x['mean_cv'], reverse=True)
best_blend = blend_results[0]

print("\n" + "=" * 80)
print(f"PEAK V16 BLEND: {best_blend['name']}")
print(f"Mean CV: {best_blend['mean_cv']:.5f} (Δ vs V15: {best_blend['delta_vs_v15']:+.5f}, Δ vs V9: {best_blend['delta_vs_v9']:+.5f})")
print("=" * 80)

# Save JSON artifacts
# Strip out large numpy arrays for clean JSON storage
specialist_summary = {}
for k, v in specialist_models.items():
    item = {kk: vv for kk, vv in v.items() if kk != 'fold_preds'}
    specialist_summary[k] = item

diversity_summary = {
    tag: {
        'name': d['name'],
        'pearson_corr_v15': d['pearson_corr_v15'],
        'spearman_corr_v15': d['spearman_corr_v15'],
        'residual_corr_v15': d['residual_corr_v15'],
        'mean_cv': d['mean_cv']
    }
    for tag, d in specialist_models.items()
}

os.makedirs('v16', exist_ok=True)

with open('v16/v16_specialist_results.json', 'w') as f:
    json.dump(specialist_summary, f, indent=2)

with open('v16/v16_diversity_results.json', 'w') as f:
    json.dump(diversity_summary, f, indent=2)

with open('v16/v16_blend_results.json', 'w') as f:
    json.dump({'all_blends': blend_results, 'best_blend': best_blend}, f, indent=2)

# Save master results
master_v16 = {
    'campaign': 'V16 Feature-View Tree Specialists',
    'hard_target_threshold': 0.87000,
    'v9_baseline': v9_champion_mean,
    'v15_baseline': v15_champion_mean,
    'specialists': specialist_summary,
    'best_blend': best_blend,
    'winning_candidate': best_blend,
    'gap_to_target_0_87000': 0.87000 - best_blend['mean_cv'],
    'submission_created': False,
    'submission_uploaded': False,
    'runtime_total_seconds': time.time() - t0
}

with open('v16/v16_results.json', 'w') as f:
    json.dump(master_v16, f, indent=2)

# Save arrays
np.savez_compressed(
    'v16/v16_oof_preds.npz',
    y_true_f1=val_y[0],
    y_true_f2=val_y[1],
    y_true_f3=val_y[2],
    v15_pred_f1=v15_champion_preds[0],
    v15_pred_f2=v15_champion_preds[1],
    v15_pred_f3=v15_champion_preds[2],
    best_blend_f1=0.90 * v15_champion_preds[0] + 0.10 * specialist_models['profile_billing_d3']['fold_preds'][0],
    best_blend_f2=0.90 * v15_champion_preds[1] + 0.10 * specialist_models['profile_billing_d3']['fold_preds'][1],
    best_blend_f3=0.90 * v15_champion_preds[2] + 0.10 * specialist_models['profile_billing_d3']['fold_preds'][2]
)

print(f"\nV16 execution completed in {time.time() - t0:.1f}s.")
