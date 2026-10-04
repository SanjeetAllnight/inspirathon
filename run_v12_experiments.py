import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

print("=" * 85)
print("V12 EXPERIMENTS: DATASET SHIFT, ADVERSARIAL WEIGHTING, MODEL DIVERSITY & BLENDS")
print("=" * 85)

# 1. Load Data
t_start = time.time()
train_df = pd.read_parquet('v6/train_features_v6.parquet')
test_df = pd.read_parquet('v6/test_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

train_df['referral_code'] = train_raw['referral_code'].values
test_df['referral_code'] = test_raw['referral_code'].values
train_df['billing_cycle'] = train_raw['billing_cycle'].values
test_df['billing_cycle'] = test_raw['billing_cycle'].values

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

# ===========================================================================
# PART A: ADVERSARIAL VALIDATION (TRAIN VS TEST & TEMPORAL SHIFT)
# ===========================================================================
print("\n" + "=" * 80)
print("PART A: ADVERSARIAL VALIDATION (TRAIN VS TEST & TEMPORAL SHIFT)")
print("=" * 80)

# 1. Full Train vs Test Adversarial Model
print("\n--- 1. Full Train (Jan-Apr) vs Test (May-Jun) Shift Analysis ---")
X_tr_all = train_df[v7_features].copy()
X_te_all = test_df[v7_features].copy()

X_adv_full = pd.concat([X_tr_all, X_te_all], axis=0).reset_index(drop=True)
y_adv_full = np.array([0] * len(X_tr_all) + [1] * len(X_te_all))

c_idx_v7 = [v7_features.index(c) for c in cat_names if c in v7_features]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
full_adv_preds = np.zeros(len(X_adv_full))
full_feat_imp = np.zeros(len(v7_features))

for tr_idx, val_idx in skf.split(X_adv_full, y_adv_full):
    clf = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.1, cat_features=c_idx_v7, eval_metric='AUC', verbose=0, thread_count=-1)
    clf.fit(X_adv_full.iloc[tr_idx], y_adv_full[tr_idx], eval_set=(X_adv_full.iloc[val_idx], y_adv_full[val_idx]))
    full_adv_preds[val_idx] = clf.predict_proba(X_adv_full.iloc[val_idx])[:, 1]
    full_feat_imp += clf.get_feature_importance() / 3.0

full_adv_auc = roc_auc_score(y_adv_full, full_adv_preds)
print(f"Full Train-vs-Test Adversarial AUC: {full_adv_auc:.5f}")

shift_series = pd.Series(full_feat_imp, index=v7_features).sort_values(ascending=False)
print("\nTop 10 Features Driving Train-to-Test Covariate Shift:")
for f_name, imp in shift_series.head(10).items():
    print(f"  {f_name:32s}: {imp:6.2f}%")

# 2. In-Fold Adversarial Validation & Propensity Weights
# Strictly forward-safe: simulate train (source 0) vs validation (target 1) for each fold
fold_adv_aucs = []
fold_weights_raw = []
fold_weights_mild = []
fold_weights_damped = []

print("\n--- 2. Forward-Fold Adversarial Propensity Estimation ---")
for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    
    # Feature set for adversarial check: base 86 features
    X_s = tr_sub[v7_features].copy()
    X_t = vl_sub[v7_features].copy()
    
    N_s = len(X_s)
    N_t = len(X_t)
    
    X_comb = pd.concat([X_s, X_t], axis=0).reset_index(drop=True)
    y_comb = np.array([0] * N_s + [1] * N_t)
    
    # Run 5-fold CV on combined set to get out-of-fold predictions for source instances
    oof_p = np.zeros(N_s)
    skf_f = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_preds_fold = np.zeros(len(X_comb))
    
    for tr_i, val_i in skf_f.split(X_comb, y_comb):
        m_adv = CatBoostClassifier(iterations=250, depth=4, learning_rate=0.1, cat_features=c_idx_v7, eval_metric='AUC', verbose=0, thread_count=-1)
        m_adv.fit(X_comb.iloc[tr_i], y_comb[tr_i], eval_set=(X_comb.iloc[val_i], y_comb[val_i]))
        val_preds_fold[val_i] = m_adv.predict_proba(X_comb.iloc[val_i])[:, 1]
    
    fold_auc = roc_auc_score(y_comb, val_preds_fold)
    fold_adv_aucs.append(fold_auc)
    
    # Source probabilities
    p_s = val_preds_fold[:N_s]
    # Bound probabilities to avoid extreme division
    p_s_clipped = np.clip(p_s, 0.01, 0.99)
    
    # Density ratio: w = [p(x) / (1 - p(x))] * (N_source / N_target)
    w_raw = (p_s_clipped / (1.0 - p_s_clipped)) * (float(N_s) / float(N_t))
    w_raw = w_raw / np.mean(w_raw) # normalize to mean 1.0
    
    # Strategy A: Clipped [0.2, 5.0]
    w_clipped = np.clip(w_raw, 0.2, 5.0)
    w_clipped = w_clipped / np.mean(w_clipped)
    
    # Strategy B: Mild Clipped [0.5, 2.0]
    w_mild = np.clip(w_raw, 0.5, 2.0)
    w_mild = w_mild / np.mean(w_mild)
    
    # Strategy C: Square-root damped
    w_damped = np.sqrt(w_clipped)
    w_damped = w_damped / np.mean(w_damped)
    
    fold_weights_raw.append(w_clipped)
    fold_weights_mild.append(w_mild)
    fold_weights_damped.append(w_damped)
    
    print(f"{fold_name}: Adversarial AUC = {fold_auc:.5f} | Raw w: [{w_raw.min():.2f}, {w_raw.max():.2f}], Mild w: [{w_mild.min():.2f}, {w_mild.max():.2f}]")

# ===========================================================================
# PART B: TEST-DISTRIBUTION-WEIGHTED CATBOOST
# ===========================================================================
print("\n" + "=" * 80)
print("PART B: TEST-DISTRIBUTION-WEIGHTED CATBOOST EVALUATION")
print("=" * 80)

hp_d4 = {'depth': 4, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}

def evaluate_weighting_strategy(weights_list, strategy_name):
    folds_auc = []
    for f_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        
        X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        
        sw = weights_list[f_idx] if weights_list is not None else None
        
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, sample_weight=sw, eval_set=(X_vl, y_vl))
        pred = m.predict_proba(X_vl)[:, 1]
        folds_auc.append(float(roc_auc_score(y_vl, pred)))
        
    m_auc = float(np.mean(folds_auc))
    print(f"  {strategy_name:38s} -> Mean CV: {m_auc:.5f} | F1: {folds_auc[0]:.5f}, F2: {folds_auc[1]:.5f}, F3: {folds_auc[2]:.5f}")
    return {'mean': m_auc, 'folds': folds_auc}

res_uniform = evaluate_weighting_strategy(None, "1. Uniform Weights (V9 Baseline)")
res_adv_w   = evaluate_weighting_strategy(fold_weights_raw, "2. Adversarial Weights (Clipped 0.2-5.0)")
res_mild_w  = evaluate_weighting_strategy(fold_weights_mild, "3. Mild Clipped Weights (0.5-2.0)")
res_damp_w  = evaluate_weighting_strategy(fold_weights_damped, "4. Damped Sqrt Weights")

# ===========================================================================
# PART C: CATBOOST DIVERSITY (ORDERED BOOSTING & LOSSGUIDE GROW POLICY)
# ===========================================================================
print("\n" + "=" * 80)
print("PART C: CATBOOST ARCHITECTURAL DIVERSITY EVALUATION")
print("=" * 80)

diverse_configs = [
    ("SymmetricTree Plain Depth 4 (Baseline)", {'depth': 4, 'boosting_type': 'Plain', 'grow_policy': 'SymmetricTree', 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}),
    ("SymmetricTree Ordered Depth 4",          {'depth': 4, 'boosting_type': 'Ordered', 'grow_policy': 'SymmetricTree', 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}),
    ("Lossguide Plain Depth 4 (max_leaves=16)",{'depth': 4, 'boosting_type': 'Plain', 'grow_policy': 'Lossguide', 'max_leaves': 16, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}),
    ("Lossguide Plain Depth 6 (max_leaves=32)",{'depth': 6, 'boosting_type': 'Plain', 'grow_policy': 'Lossguide', 'max_leaves': 32, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1}),
    ("SymmetricTree Plain Depth 6",            {'depth': 6, 'boosting_type': 'Plain', 'grow_policy': 'SymmetricTree', 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0, 'thread_count': -1})
]

cat_eval_results = {}
for name, hp_dict in diverse_configs:
    f_aucs = []
    for f_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        y_tr = tr_sub['churn'].values
        y_vl = vl_sub['churn'].values
        
        X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
        all_feats = list(X_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        
        m = CatBoostClassifier(iterations=1500, random_seed=42, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_dict)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_aucs.append(float(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])))
    m_res = float(np.mean(f_aucs))
    delta = m_res - res_uniform['mean']
    cat_eval_results[name] = {'mean': m_res, 'delta': delta, 'folds': f_aucs}
    print(f"  {name:42s} -> Mean: {m_res:.5f} (Δ: {delta:+.5f}) | F1: {f_aucs[0]:.5f}, F2: {f_aucs[1]:.5f}, F3: {f_aucs[2]:.5f}")

# ===========================================================================
# PART D: BLENDING (PROBABILITY-AVERAGE VS RANK-AVERAGE & RATIOS)
# ===========================================================================
print("\n" + "=" * 80)
print("PART D: ENSEMBLE TRAINING & BLENDING (PROBABILITY VS RANK AVERAGE)")
print("=" * 80)

seeds = [42, 123, 314, 555, 777]

# 5-Seed Ensemble for Best CatBoost model
cb_best_preds = []
lr_preds = []
y_trues = []

d4_ens_folds = []
lr_folds = []

print("Training 5-Seed CatBoost Ensemble & Logistic Regression across 3 forward folds...")
for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    t_f = time.time()
    tr_sub = train_df[tr_mask].copy()
    vl_sub = train_df[vl_mask].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    y_trues.append(y_vl)
    
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    # 5-Seed CatBoost
    f_seed_preds = []
    for s in seeds:
        m = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_seed_preds.append(m.predict_proba(X_vl)[:, 1])
    cb_ens = np.mean(f_seed_preds, axis=0)
    cb_best_preds.append(cb_ens)
    auc_cb = float(roc_auc_score(y_vl, cb_ens))
    d4_ens_folds.append(auc_cb)
    
    # Logistic Regression on base V7 features
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
    auc_lr = float(roc_auc_score(y_vl, lp))
    lr_folds.append(auc_lr)
    
    print(f"  {fold_name} [{time.time() - t_f:.1f}s]: CatBoost Ens AUC = {auc_cb:.5f} | Logistic Reg AUC = {auc_lr:.5f}")

print(f"\nCatBoost Depth-4 5-Seed Mean CV : {np.mean(d4_ens_folds):.5f}")
print(f"Logistic Regression Mean CV     : {np.mean(lr_folds):.5f}")

# Compare Blending Strategies: Probability vs Rank Average at 90/10 and 95/5
def eval_blend(p_cb, p_lr, w_cb, w_lr, method='prob'):
    blend_folds = []
    for f in range(3):
        y_true = y_trues[f]
        if method == 'prob':
            comb = w_cb * p_cb[f] + w_lr * p_lr[f]
        else: # rank-average
            r_cb = rankdata(p_cb[f]) / float(len(p_cb[f]))
            r_lr = rankdata(p_lr[f]) / float(len(p_lr[f]))
            comb = w_cb * r_cb + w_lr * r_lr
        blend_folds.append(float(roc_auc_score(y_true, comb)))
    return float(np.mean(blend_folds)), blend_folds

# 1. Probability Average 90/10
mean_p90, folds_p90 = eval_blend(cb_best_preds, lr_preds, 0.90, 0.10, 'prob')
print(f"  Blend 90/10 (Probability Average) -> Mean CV: {mean_p90:.5f} | F1: {folds_p90[0]:.5f}, F2: {folds_p90[1]:.5f}, F3: {folds_p90[2]:.5f} (Δ vs V9: {mean_p90 - 0.86824:+.5f})")

# 2. Probability Average 95/5
mean_p95, folds_p95 = eval_blend(cb_best_preds, lr_preds, 0.95, 0.05, 'prob')
print(f"  Blend 95/05 (Probability Average) -> Mean CV: {mean_p95:.5f} | F1: {folds_p95[0]:.5f}, F2: {folds_p95[1]:.5f}, F3: {folds_p95[2]:.5f} (Δ vs V9: {mean_p95 - 0.86824:+.5f})")

# 3. Probability Average 88/12 (V9 benchmark)
mean_p88, folds_p88 = eval_blend(cb_best_preds, lr_preds, 0.88, 0.12, 'prob')
print(f"  Blend 88/12 (Probability Average) -> Mean CV: {mean_p88:.5f} | F1: {folds_p88[0]:.5f}, F2: {folds_p88[1]:.5f}, F3: {folds_p88[2]:.5f} (Δ vs V9: {mean_p88 - 0.86824:+.5f})")

# 4. Rank Average 90/10
mean_r90, folds_r90 = eval_blend(cb_best_preds, lr_preds, 0.90, 0.10, 'rank')
print(f"  Blend 90/10 (Rank-Average)        -> Mean CV: {mean_r90:.5f} | F1: {folds_r90[0]:.5f}, F2: {folds_r90[1]:.5f}, F3: {folds_r90[2]:.5f} (Δ vs V9: {mean_r90 - 0.86824:+.5f})")

# 5. Rank Average 95/5
mean_r95, folds_r95 = eval_blend(cb_best_preds, lr_preds, 0.95, 0.05, 'rank')
print(f"  Blend 95/05 (Rank-Average)        -> Mean CV: {mean_r95:.5f} | F1: {folds_r95[0]:.5f}, F2: {folds_r95[1]:.5f}, F3: {folds_r95[2]:.5f} (Δ vs V9: {mean_r95 - 0.86824:+.5f})")

# Save detailed summary
os.makedirs('v12', exist_ok=True)
v12_output = {
    'v9_baseline_mean_cv': 0.86824,
    'full_train_vs_test_adversarial_auc': full_adv_auc,
    'top_shifted_features': {k: float(v) for k, v in shift_series.head(15).items()},
    'temporal_adversarial_aucs': {
        'fold_1_jan_vs_feb': fold_adv_aucs[0],
        'fold_2_janfeb_vs_mar': fold_adv_aucs[1],
        'fold_3_janmar_vs_apr': fold_adv_aucs[2],
        'mean_temporal_adv_auc': float(np.mean(fold_adv_aucs))
    },
    'part_b_weighting_experiments': {
        'uniform': res_uniform,
        'adversarial_clipped_0.2_5.0': res_adv_w,
        'mild_clipped_0.5_2.0': res_mild_w,
        'damped_sqrt': res_damp_w
    },
    'part_c_catboost_diversity': cat_eval_results,
    'part_d_blending': {
        'prob_90_10': {'mean': mean_p90, 'folds': folds_p90, 'delta_vs_v9': mean_p90 - 0.86824},
        'prob_95_05': {'mean': mean_p95, 'folds': folds_p95, 'delta_vs_v9': mean_p95 - 0.86824},
        'prob_88_12': {'mean': mean_p88, 'folds': folds_p88, 'delta_vs_v9': mean_p88 - 0.86824},
        'rank_90_10': {'mean': mean_r90, 'folds': folds_r90, 'delta_vs_v9': mean_r90 - 0.86824},
        'rank_95_05': {'mean': mean_r95, 'folds': folds_r95, 'delta_vs_v9': mean_r95 - 0.86824}
    }
}

with open('v12/v12_final_summary.json', 'w') as f:
    json.dump(v12_output, f, indent=2)

print("\nV12 execution complete! Results saved to v12/v12_final_summary.json")
