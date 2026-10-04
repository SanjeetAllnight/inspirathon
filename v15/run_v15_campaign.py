import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore', category=UserWarning)

print("=" * 80)
print("V15 PIPELINE: DIVERSE TREE TRIAD (CATBOOST + LIGHTGBM + XGBOOST)")
print("=" * 80)

t0 = time.time()

# 1. Load data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
num_cols = [c for c in v7_features if c not in cat_names]

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

# 2. Load V14 Baseline Predictions
oof_v14 = np.load('v14/v14_oof_preds.npz', allow_pickle=True)
val_y = [oof_v14['y_true_f1'], oof_v14['y_true_f2'], oof_v14['y_true_f3']]
val_p_glob = [oof_v14['p_glob_f1'], oof_v14['p_glob_f2'], oof_v14['p_glob_f3']]
val_p_exp = [oof_v14['p_exp_f1'], oof_v14['p_exp_f2'], oof_v14['p_exp_f3']]
val_regions = [oof_v14['reg_f1'], oof_v14['reg_f2'], oof_v14['reg_f3']]
val_annual_m = [oof_v14['ann_m_f1'], oof_v14['ann_m_f2'], oof_v14['ann_m_f3']]
val_monthly_m = [oof_v14['mth_m_f1'], oof_v14['mth_m_f2'], oof_v14['mth_m_f3']]

with open('v14/v14_residual_results.json') as f:
    res_data = json.load(f)
effects = res_data['training_effects']

v9_champion_mean = 0.86824
v14_champion_mean = 0.8691821

# Reconstruct exact V14 tree candidate and V14 champion
v14_tree_preds = []
v14_lr_preds = []
v14_champion_preds = []

for f, (fname, tr_m, vl_m) in enumerate(cv_splits):
    # Tree: 75% Global + 25% Expert + Joint Residual Correction (alpha=0.50)
    p_base = 0.75 * val_p_glob[f] + 0.25 * val_p_exp[f]
    p_tree = p_base.copy()
    eff = effects[f]
    p_tree[val_regions[f] == 'central'] += 0.5 * eff['delta_central']
    p_tree[val_annual_m[f]] += 0.5 * eff['delta_annual']
    p_tree = np.clip(p_tree, 0, 1)
    v14_tree_preds.append(p_tree)
    
    # Fit LR on base features
    tr_sub = train_df[tr_m]
    vl_sub = train_df[vl_m]
    log_pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    log_pipe.fit(tr_sub[v7_features], tr_sub['churn'])
    p_lr = log_pipe.predict_proba(vl_sub[v7_features])[:, 1]
    v14_lr_preds.append(p_lr)
    
    # V14 Champion: 88% Tree + 12% LR
    p_v14 = 0.88 * p_tree + 0.12 * p_lr
    v14_champion_preds.append(p_v14)

v14_champion_folds = [float(roc_auc_score(val_y[f], v14_champion_preds[f])) for f in range(3)]
print(f"Verified V14 Baseline Champion: Mean CV = {np.mean(v14_champion_folds):.5f} | F1: {v14_champion_folds[0]:.5f}, F2: {v14_champion_folds[1]:.5f}, F3: {v14_champion_folds[2]:.5f}")

# 3. Phase B: Train LightGBM & XGBoost (5 seeds each: 42, 123, 314, 555, 777)
seeds = [42, 123, 314, 555, 777]

lgbm_val_preds = []
xgb_val_preds = []

lgbm_fold_aucs = []
xgb_fold_aucs = []

t_models_start = time.time()

for f, (fname, tr_m, vl_m) in enumerate(cv_splits):
    t_f = time.time()
    print(f"\n>>> Training LightGBM & XGBoost for {fname}...")
    
    tr_sub = train_df[tr_m].copy()
    vl_sub = train_df[vl_m].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    
    # Prepare category dtype for categoricals
    for c in cat_names:
        X_tr[c] = X_tr[c].astype('category')
        X_vl[c] = X_vl[c].astype('category')
        
    # --- LightGBM (Depth 3, num_leaves=7) ---
    seed_lgb = []
    for s in seeds:
        m_lgb = lgb.LGBMClassifier(
            max_depth=3,
            num_leaves=7,
            learning_rate=0.04,
            n_estimators=1500,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=s,
            objective='binary',
            verbose=-1
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        seed_lgb.append(m_lgb.predict_proba(X_vl)[:, 1])
    p_lgb = np.mean(seed_lgb, axis=0)
    lgbm_val_preds.append(p_lgb)
    auc_lgb = float(roc_auc_score(y_vl, p_lgb))
    lgbm_fold_aucs.append(auc_lgb)
    
    # --- XGBoost (Depth 3) ---
    seed_xgb = []
    for s in seeds:
        m_xgb = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.04,
            n_estimators=1500,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            enable_categorical=True,
            random_state=s,
            eval_metric='auc',
            early_stopping_rounds=100
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        seed_xgb.append(m_xgb.predict_proba(X_vl)[:, 1])
    p_xgb = np.mean(seed_xgb, axis=0)
    xgb_val_preds.append(p_xgb)
    auc_xgb = float(roc_auc_score(y_vl, p_xgb))
    xgb_fold_aucs.append(auc_xgb)
    
    print(f"  {fname} done in {time.time() - t_f:.1f}s | LightGBM AUC: {auc_lgb:.5f} | XGBoost AUC: {auc_xgb:.5f}")

mean_lgb = float(np.mean(lgbm_fold_aucs))
mean_xgb = float(np.mean(xgb_fold_aucs))

print("\n" + "=" * 80)
print("PHASE C: INDIVIDUAL MODEL RESULTS")
print("=" * 80)
print(f"LightGBM Depth-3 (5 seeds) : Mean CV = {mean_lgb:.5f} | F1: {lgbm_fold_aucs[0]:.5f}, F2: {lgbm_fold_aucs[1]:.5f}, F3: {lgbm_fold_aucs[2]:.5f}")
print(f"  Delta vs V9: {mean_lgb - v9_champion_mean:+.5f} | Delta vs V14: {mean_lgb - v14_champion_mean:+.5f}")
print(f"XGBoost Depth-3 (5 seeds)  : Mean CV = {mean_xgb:.5f} | F1: {xgb_fold_aucs[0]:.5f}, F2: {xgb_fold_aucs[1]:.5f}, F3: {xgb_fold_aucs[2]:.5f}")
print(f"  Delta vs V9: {mean_xgb - v9_champion_mean:+.5f} | Delta vs V14: {mean_xgb - v14_champion_mean:+.5f}")

# Phase C: Correlation Analysis
concat_y = np.concatenate(val_y)
concat_v14 = np.concatenate(v14_champion_preds)
concat_lgb = np.concatenate(lgbm_val_preds)
concat_xgb = np.concatenate(xgb_val_preds)

corr_v14_lgb = float(pearsonr(concat_v14, concat_lgb)[0])
corr_v14_xgb = float(pearsonr(concat_v14, concat_xgb)[0])
corr_lgb_xgb = float(pearsonr(concat_lgb, concat_xgb)[0])

res_v14 = concat_y - concat_v14
res_lgb = concat_y - concat_lgb
res_xgb = concat_y - concat_xgb

res_corr_v14_lgb = float(pearsonr(res_v14, res_lgb)[0])
res_corr_v14_xgb = float(pearsonr(res_v14, res_xgb)[0])
res_corr_lgb_xgb = float(pearsonr(res_lgb, res_xgb)[0])

per_fold_corr = []
for f in range(3):
    per_fold_corr.append({
        'fold': f + 1,
        'v14_lgb_corr': float(pearsonr(v14_champion_preds[f], lgbm_val_preds[f])[0]),
        'v14_xgb_corr': float(pearsonr(v14_champion_preds[f], xgb_val_preds[f])[0]),
        'lgb_xgb_corr': float(pearsonr(lgbm_val_preds[f], xgb_val_preds[f])[0])
    })

print(f"\nPrediction Correlations with V14 Champion: LightGBM = {corr_v14_lgb:.4f}, XGBoost = {corr_v14_xgb:.4f}")
print(f"Residual Correlations with V14:           LightGBM = {res_corr_v14_lgb:.4f}, XGBoost = {res_corr_v14_xgb:.4f}")

# Save Phase C results
model_results = {
    'lightgbm': {
        'mean_cv': mean_lgb,
        'folds': lgbm_fold_aucs,
        'delta_vs_v9': mean_lgb - v9_champion_mean,
        'delta_vs_v14': mean_lgb - v14_champion_mean,
        'prediction_corr_v14': corr_v14_lgb,
        'residual_corr_v14': res_corr_v14_lgb
    },
    'xgboost': {
        'mean_cv': mean_xgb,
        'folds': xgb_fold_aucs,
        'delta_vs_v9': mean_xgb - v9_champion_mean,
        'delta_vs_v14': mean_xgb - v14_champion_mean,
        'prediction_corr_v14': corr_v14_xgb,
        'residual_corr_v14': res_corr_v14_xgb
    },
    'per_fold_correlations': per_fold_corr
}

with open('v15/v15_model_results.json', 'w') as f:
    json.dump(model_results, f, indent=2)

diversity_results = {
    'prediction_correlation_matrix': {
        'v14_vs_lgb': corr_v14_lgb,
        'v14_vs_xgb': corr_v14_xgb,
        'lgb_vs_xgb': corr_lgb_xgb
    },
    'residual_correlation_matrix': {
        'v14_vs_lgb': res_corr_v14_lgb,
        'v14_vs_xgb': res_corr_v14_xgb,
        'lgb_vs_xgb': res_corr_lgb_xgb
    },
    'per_fold_correlations': per_fold_corr
}

with open('v15/v15_diversity_results.json', 'w') as f:
    json.dump(diversity_results, f, indent=2)

# ==============================================================================
# PHASE D: BLENDING WITH V14
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE D: BLENDING WITH V14 CHAMPION")
print("=" * 80)

blend_configs = []

# 1. V14 + LightGBM blends
for w_lgb in [0.05, 0.10, 0.15, 0.20]:
    w_v14 = 1.0 - w_lgb
    blend_configs.append({'name': f"V14 ({int(w_v14*100)}) + LGBM ({int(w_lgb*100)})", 'w_v14': w_v14, 'w_lgb': w_lgb, 'w_xgb': 0.0})

# 2. V14 + XGBoost blends
for w_xgb in [0.05, 0.10, 0.15, 0.20]:
    w_v14 = 1.0 - w_xgb
    blend_configs.append({'name': f"V14 ({int(w_v14*100)}) + XGB ({int(w_xgb*100)})", 'w_v14': w_v14, 'w_lgb': 0.0, 'w_xgb': w_xgb})

# 3. Three-Model Combinations: V14 + LightGBM + XGBoost
triad_weights = [
    (0.90, 0.05, 0.05),
    (0.85, 0.10, 0.05),
    (0.85, 0.05, 0.10),
    (0.80, 0.10, 0.10),
    (0.75, 0.125, 0.125),
    (0.70, 0.15, 0.15)
]

for w_v14, w_lgb, w_xgb in triad_weights:
    blend_configs.append({
        'name': f"V14 ({w_v14*100:.1f}) + LGBM ({w_lgb*100:.1f}) + XGB ({w_xgb*100:.1f})",
        'w_v14': w_v14,
        'w_lgb': w_lgb,
        'w_xgb': w_xgb
    })

blend_results = []

for cfg in blend_configs:
    w1, w2, w3 = cfg['w_v14'], cfg['w_lgb'], cfg['w_xgb']
    f_aucs = []
    ann_f2 = None
    mth_f2 = None
    
    for f in range(3):
        p_b = w1 * v14_champion_preds[f] + w2 * lgbm_val_preds[f] + w3 * xgb_val_preds[f]
        score = float(roc_auc_score(val_y[f], p_b))
        f_aucs.append(score)
        if f == 1:
            ann_m = val_annual_m[1]
            mth_m = val_monthly_m[1]
            ann_f2 = float(roc_auc_score(val_y[1][ann_m], p_b[ann_m]))
            mth_f2 = float(roc_auc_score(val_y[1][mth_m], p_b[mth_m]))
            
    m_cv = float(np.mean(f_aucs))
    delta_v14 = m_cv - v14_champion_mean
    delta_v9 = m_cv - v9_champion_mean
    
    res = {
        'name': cfg['name'],
        'w_v14': w1,
        'w_lgb': w2,
        'w_xgb': w3,
        'f1': f_aucs[0],
        'f2': f_aucs[1],
        'f3': f_aucs[2],
        'mean_cv': m_cv,
        'delta_vs_v14': delta_v14,
        'delta_vs_v9': delta_v9,
        'annual_f2': ann_f2,
        'monthly_f2': mth_f2
    }
    blend_results.append(res)
    print(f"{cfg['name']:<42s} -> Mean CV: {m_cv:.5f} | F1: {f_aucs[0]:.5f}, F2: {f_aucs[1]:.5f}, F3: {f_aucs[2]:.5f} | Δ(V14): {delta_v14:+.5f} | Ann F2: {ann_f2:.5f}")

# Sort by mean CV
blend_results.sort(key=lambda x: x['mean_cv'], reverse=True)
best_blend = blend_results[0]

with open('v15/v15_blend_results.json', 'w') as f:
    json.dump({'all_blends': blend_results, 'best_blend': best_blend}, f, indent=2)

print("\n" + "=" * 80)
print(f"PEAK BLEND: {best_blend['name']}")
print(f"Mean CV: {best_blend['mean_cv']:.5f} (Δ vs V14: {best_blend['delta_vs_v14']:+.5f}, Δ vs V9: {best_blend['delta_vs_v9']:+.5f})")
print("=" * 80)

# ==============================================================================
# PHASE G: LOGISTIC REGRESSION REFINEMENT ON THE TREE TRIAD
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE G: LOGISTIC REGRESSION REFINEMENT")
print("=" * 80)

# Tree Triad alone (without LR baked into V14):
# Recall v14_tree_preds is the pure tree component of V14
# Let's test pure Tree Triad (CatBoost V14 Tree + LGBM + XGB) blended with LR!
w1, w2, w3 = best_blend['w_v14'], best_blend['w_lgb'], best_blend['w_xgb']

# Pure tree triad base:
tree_triad_preds = []
for f in range(3):
    p_tt = (w1 * v14_tree_preds[f] + w2 * lgbm_val_preds[f] + w3 * xgb_val_preds[f]) / (w1 + w2 + w3)
    tree_triad_preds.append(p_tt)

lr_weights = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15]
lr_sweep_results = []

for w_lr in lr_weights:
    f_aucs = []
    for f in range(3):
        p_final = (1.0 - w_lr) * tree_triad_preds[f] + w_lr * v14_lr_preds[f]
        f_aucs.append(float(roc_auc_score(val_y[f], p_final)))
    m_cv = float(np.mean(f_aucs))
    delta_v14 = m_cv - v14_champion_mean
    delta_v9 = m_cv - v9_champion_mean
    lr_sweep_results.append({
        'weight_lr': w_lr,
        'weight_tree_triad': round(1.0 - w_lr, 2),
        'f1': f_aucs[0],
        'f2': f_aucs[1],
        'f3': f_aucs[2],
        'mean_cv': m_cv,
        'delta_vs_v14': delta_v14,
        'delta_vs_v9': delta_v9
    })
    print(f"Tree Triad ({1-w_lr:.2f}) + LR ({w_lr:.2f}) -> Mean CV: {m_cv:.5f} | F1: {f_aucs[0]:.5f}, F2: {f_aucs[1]:.5f}, F3: {f_aucs[2]:.5f} | Δ(V14): {delta_v14:+.5f}")

lr_sweep_results.sort(key=lambda x: x['mean_cv'], reverse=True)
best_lr_candidate = lr_sweep_results[0]

# ==============================================================================
# PHASE H: SOFT COHORT WEIGHTING CHECK
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE H: SOFT COHORT WEIGHTING TEST")
print("=" * 80)

# Test mild annual sample weighting (1.15x) on LightGBM across folds
weighted_lgb_aucs = []
for f, (fname, tr_m, vl_m) in enumerate(cv_splits):
    tr_sub = train_df[tr_m].copy()
    vl_sub = train_df[vl_m].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    for c in cat_names:
        X_tr[c] = X_tr[c].astype('category')
        X_vl[c] = X_vl[c].astype('category')
        
    # Mild weighting: upweight annual subscribers by 1.15x
    sw_tr = np.ones(len(y_tr), dtype=np.float32)
    sw_tr[tr_sub['billing_cycle'] == 'annual'] = 1.15
    
    m_lgb_w = lgb.LGBMClassifier(
        max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=1500,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, objective='binary', verbose=-1
    )
    m_lgb_w.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_vl, y_vl)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    p_lgb_w = m_lgb_w.predict_proba(X_vl)[:, 1]
    weighted_lgb_aucs.append(float(roc_auc_score(y_vl, p_lgb_w)))

m_lgb_weighted = float(np.mean(weighted_lgb_aucs))
delta_w = m_lgb_weighted - mean_lgb
print(f"LightGBM with 1.15x Annual Sample Weight -> Mean CV: {m_lgb_weighted:.5f} (Δ vs unweighted: {delta_w:+.5f})")
if abs(delta_w) < 0.0003:
    print("Decision on Soft Weighting: REJECT (< 0.0003 delta, treated as noise per rule).")
    weighting_accepted = False
else:
    print("Decision on Soft Weighting: ACCEPT")
    weighting_accepted = True

# ==============================================================================
# FINAL CANDIDATE SELECTION & ARTIFACTS
# ==============================================================================
# Compare best overall candidates
candidate_a = best_blend  # Blend directly on V14 champion
candidate_b = best_lr_candidate  # Re-tuned tree triad + LR

if candidate_b['mean_cv'] >= candidate_a['mean_cv']:
    winning_candidate = candidate_b
    winner_name = f"Tree Triad ({candidate_b['weight_tree_triad']}) + LR ({candidate_b['weight_lr']})"
else:
    winning_candidate = candidate_a
    winner_name = candidate_a['name']

print("\n" + "=" * 80)
print(f"WINNING V15 CANDIDATE: {winner_name}")
print(f"Fold 1: {winning_candidate['f1']:.5f}")
print(f"Fold 2: {winning_candidate['f2']:.5f}")
print(f"Fold 3: {winning_candidate['f3']:.5f}")
print(f"Mean CV: {winning_candidate['mean_cv']:.5f}")
print(f"Delta vs V14 Champion : {winning_candidate['delta_vs_v14']:+.5f}")
print(f"Delta vs V9 Champion  : {winning_candidate['delta_vs_v9']:+.5f}")
print(f"Gap to 0.87000        : {0.87000 - winning_candidate['mean_cv']:.5f}")
print("=" * 80)

# Save master results JSON
master_results = {
    'campaign': 'V15 Tree Triad Campaign',
    'hard_target_threshold': 0.87000,
    'v9_baseline': v9_champion_mean,
    'v14_baseline': v14_champion_mean,
    'models': model_results,
    'best_blend_on_v14': best_blend,
    'best_lr_re_blend': best_lr_candidate,
    'winning_candidate': winning_candidate,
    'soft_weighting_evaluated': {
        'tested': '1.15x annual sample weight on LightGBM',
        'mean_cv': m_lgb_weighted,
        'delta': delta_w,
        'accepted': weighting_accepted
    },
    'gap_to_target_0_87000': 0.87000 - winning_candidate['mean_cv'],
    'submission_created': False,
    'submission_uploaded': False,
    'runtime_total_seconds': time.time() - t0
}

with open('v15/v15_results.json', 'w') as f:
    json.dump(master_results, f, indent=2)

# Save predictions for audits
np.savez_compressed(
    'v15/v15_oof_preds.npz',
    y_true_f1=val_y[0],
    y_true_f2=val_y[1],
    y_true_f3=val_y[2],
    v14_pred_f1=v14_champion_preds[0],
    v14_pred_f2=v14_champion_preds[1],
    v14_pred_f3=v14_champion_preds[2],
    lgb_pred_f1=lgbm_val_preds[0],
    lgb_pred_f2=lgbm_val_preds[1],
    lgb_pred_f3=lgbm_val_preds[2],
    xgb_pred_f1=xgb_val_preds[0],
    xgb_pred_f2=xgb_val_preds[1],
    xgb_pred_f3=xgb_val_preds[2]
)

print(f"\nV15 Campaign execution complete in {time.time() - t0:.1f}s.")
print("Results saved to v15/v15_results.json, v15/v15_model_results.json, v15/v15_blend_results.json, v15/v15_diversity_results.json.")
