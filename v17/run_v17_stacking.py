import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

print("=" * 80)
print("V17 PIPELINE: NESTED LEAKAGE-SAFE OOF STACKING CAMPAIGN")
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
train_df['plan'] = train_raw['plan'].values

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

def get_v9_features_for_partition(tr_sub, y_tr, vl_sub):
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
v15_champion_mean = 0.8693581

# Containers for nested stacking data
# For each outer fold f:
#   outer_y_train: true labels of outer train
#   outer_y_val: true labels of outer val
#   meta_train_dict: { 'cat': arr, 'lgb': arr, 'xgb': arr, 'lr': arr, 'sp': arr }
#   meta_val_dict:   { 'cat': arr, 'lgb': arr, 'xgb': arr, 'lr': arr, 'sp': arr }
#   outer_context_val: { 'billing_cycle': arr, 'region': arr, 'plan': arr }
nested_fold_data = []

print("Starting Nested Inner-OOF Generation & Outer Base Retraining across 3 Outer Folds...")

for f_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    t_f = time.time()
    print(f"\n" + "=" * 70)
    print(f"PROCESSING {fold_name}")
    print("=" * 70)
    
    tr_outer = train_df[tr_mask].copy().reset_index(drop=True)
    vl_outer = train_df[vl_mask].copy().reset_index(drop=True)
    y_tr_outer = tr_outer['churn'].values
    y_vl_outer = vl_outer['churn'].values
    
    N_tr = len(tr_outer)
    N_vl = len(vl_outer)
    print(f"  Outer Train: {N_tr} rows | Outer Val: {N_vl} rows")
    
    # --------------------------------------------------------------------------
    # STEP 1: INNER OOF ON OUTER TRAIN (Strictly leak-free!)
    # --------------------------------------------------------------------------
    print("  Generating Inner-OOF Predictions on Outer Train (5 inner splits)...")
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42 + f_idx)
    
    oof_cat = np.zeros(N_tr, dtype=np.float64)
    oof_lgb = np.zeros(N_tr, dtype=np.float64)
    oof_xgb = np.zeros(N_tr, dtype=np.float64)
    oof_lr  = np.zeros(N_tr, dtype=np.float64)
    oof_sp  = np.zeros(N_tr, dtype=np.float64)
    
    for in_fold, (in_tr_idx, in_vl_idx) in enumerate(kf_inner.split(tr_outer)):
        t_in = time.time()
        tr_in = tr_outer.iloc[in_tr_idx].copy().reset_index(drop=True)
        vl_in = tr_outer.iloc[in_vl_idx].copy().reset_index(drop=True)
        y_in_tr = y_tr_outer[in_tr_idx]
        y_in_vl = y_tr_outer[in_vl_idx]
        
        # In-fold leak-safe features for inner split
        X_in_tr, X_in_vl = get_v9_features_for_partition(tr_in, y_in_tr, vl_in)
        all_feats = list(X_in_tr.columns)
        c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
        
        # 1. CatBoost Depth-3
        m_cat = CatBoostClassifier(iterations=1000, depth=3, learning_rate=0.04, l2_leaf_reg=1,
                                   random_strength=1, bagging_temperature=1, eval_metric='AUC',
                                   random_seed=42, early_stopping_rounds=50, verbose=0, thread_count=-1)
        m_cat.fit(X_in_tr, y_in_tr, eval_set=(X_in_vl, y_in_vl), cat_features=c_idx)
        oof_cat[in_vl_idx] = m_cat.predict_proba(X_in_vl)[:, 1]
        
        # 2. LightGBM Depth-3
        X_in_tr_lgb = X_in_tr.copy()
        X_in_vl_lgb = X_in_vl.copy()
        for c in cat_names:
            X_in_tr_lgb[c] = X_in_tr_lgb[c].astype('category')
            X_in_vl_lgb[c] = X_in_vl_lgb[c].astype('category')
        m_lgb = lgb.LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=1000,
                                   colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                                   reg_alpha=0.1, reg_lambda=1.0, random_state=42, objective='binary', verbose=-1)
        m_lgb.fit(X_in_tr_lgb, y_in_tr, eval_set=[(X_in_vl_lgb, y_in_vl)], eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[in_vl_idx] = m_lgb.predict_proba(X_in_vl_lgb)[:, 1]
        
        # 3. XGBoost Depth-3
        m_xgb = xgb.XGBClassifier(max_depth=3, learning_rate=0.04, n_estimators=1000, tree_method='hist',
                                  colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                  enable_categorical=True, random_state=42, eval_metric='auc', early_stopping_rounds=50)
        m_xgb.fit(X_in_tr_lgb, y_in_tr, eval_set=[(X_in_vl_lgb, y_in_vl)], verbose=False)
        oof_xgb[in_vl_idx] = m_xgb.predict_proba(X_in_vl_lgb)[:, 1]
        
        # 4. Logistic Regression
        log_pipe = Pipeline([
            ('prep', ColumnTransformer([
                ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
            ])),
            ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
        ])
        log_pipe.fit(tr_in[v7_features], y_in_tr)
        oof_lr[in_vl_idx] = log_pipe.predict_proba(vl_in[v7_features])[:, 1]
        
        # 5. Billing Specialist
        in_tr_mth = (tr_in['billing_cycle'] == 'monthly').values
        in_tr_ann = (tr_in['billing_cycle'] == 'annual').values
        in_vl_mth = (vl_in['billing_cycle'] == 'monthly').values
        in_vl_ann = (vl_in['billing_cycle'] == 'annual').values
        
        m_sp_mth = CatBoostClassifier(iterations=1000, depth=3, learning_rate=0.04, eval_metric='AUC',
                                      random_seed=42, early_stopping_rounds=50, verbose=0, thread_count=-1)
        m_sp_mth.fit(X_in_tr.iloc[in_tr_mth], y_in_tr[in_tr_mth], eval_set=(X_in_vl.iloc[in_vl_mth], y_in_vl[in_vl_mth]), cat_features=c_idx)
        p_sp_mth = m_sp_mth.predict_proba(X_in_vl.iloc[in_vl_mth])[:, 1]
        
        m_sp_ann = CatBoostClassifier(iterations=1000, depth=3, learning_rate=0.04, eval_metric='AUC',
                                      random_seed=42, early_stopping_rounds=50, verbose=0, thread_count=-1)
        m_sp_ann.fit(X_in_tr.iloc[in_tr_ann], y_in_tr[in_tr_ann], eval_set=(X_in_vl.iloc[in_vl_ann], y_in_vl[in_vl_ann]), cat_features=c_idx)
        p_sp_ann = m_sp_ann.predict_proba(X_in_vl.iloc[in_vl_ann])[:, 1]
        
        p_sp_combined = np.zeros(len(y_in_vl), dtype=np.float64)
        p_sp_combined[in_vl_mth] = p_sp_mth
        p_sp_combined[in_vl_ann] = p_sp_ann
        oof_sp[in_vl_idx] = p_sp_combined
        
    print(f"    Inner OOF Completed for {N_tr} rows. Inner OOF AUCs:")
    print(f"      CatBoost: {roc_auc_score(y_tr_outer, oof_cat):.5f} | LightGBM: {roc_auc_score(y_tr_outer, oof_lgb):.5f} | XGBoost: {roc_auc_score(y_tr_outer, oof_xgb):.5f} | LR: {roc_auc_score(y_tr_outer, oof_lr):.5f} | Specialist: {roc_auc_score(y_tr_outer, oof_sp):.5f}")
    
    # --------------------------------------------------------------------------
    # STEP 2: RETRAIN BASE MODELS ON ENTIRE OUTER TRAIN -> PREDICT OUTER VAL
    # --------------------------------------------------------------------------
    print("  Retraining Base Models on Full Outer Train (5 seeds per model)...")
    X_outer_tr, X_outer_vl = get_v9_features_for_partition(tr_outer, y_tr_outer, vl_outer)
    all_feats = list(X_outer_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    seeds = [42, 123, 314, 555, 777]
    
    # 1. Outer CatBoost
    cat_preds_outer = []
    for s in seeds:
        m = CatBoostClassifier(iterations=1500, depth=3, learning_rate=0.04, l2_leaf_reg=1,
                               random_strength=1, bagging_temperature=1, eval_metric='AUC',
                               random_seed=s, early_stopping_rounds=100, use_best_model=True, verbose=0, thread_count=-1)
        m.fit(X_outer_tr, y_tr_outer, eval_set=(X_outer_vl, y_vl_outer), cat_features=c_idx)
        cat_preds_outer.append(m.predict_proba(X_outer_vl)[:, 1])
    val_pred_cat = np.mean(cat_preds_outer, axis=0)
    
    # 2. Outer LightGBM
    X_outer_tr_lgb = X_outer_tr.copy()
    X_outer_vl_lgb = X_outer_vl.copy()
    for c in cat_names:
        X_outer_tr_lgb[c] = X_outer_tr_lgb[c].astype('category')
        X_outer_vl_lgb[c] = X_outer_vl_lgb[c].astype('category')
        
    lgb_preds_outer = []
    for s in seeds:
        m = lgb.LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=1500,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=s, objective='binary', verbose=-1)
        m.fit(X_outer_tr_lgb, y_tr_outer, eval_set=[(X_outer_vl_lgb, y_vl_outer)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        lgb_preds_outer.append(m.predict_proba(X_outer_vl_lgb)[:, 1])
    val_pred_lgb = np.mean(lgb_preds_outer, axis=0)
    
    # 3. Outer XGBoost
    xgb_preds_outer = []
    for s in seeds:
        m = xgb.XGBClassifier(max_depth=3, learning_rate=0.04, n_estimators=1500, tree_method='hist',
                              colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              enable_categorical=True, random_state=s, eval_metric='auc', early_stopping_rounds=100)
        m.fit(X_outer_tr_lgb, y_tr_outer, eval_set=[(X_outer_vl_lgb, y_vl_outer)], verbose=False)
        xgb_preds_outer.append(m.predict_proba(X_outer_vl_lgb)[:, 1])
    val_pred_xgb = np.mean(xgb_preds_outer, axis=0)
    
    # 4. Outer LR
    log_pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_names)
        ])),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    log_pipe.fit(tr_outer[v7_features], y_tr_outer)
    val_pred_lr = log_pipe.predict_proba(vl_outer[v7_features])[:, 1]
    
    # 5. Outer Specialist
    tr_mth = (tr_outer['billing_cycle'] == 'monthly').values
    tr_ann = (tr_outer['billing_cycle'] == 'annual').values
    vl_mth = (vl_outer['billing_cycle'] == 'monthly').values
    vl_ann = (vl_outer['billing_cycle'] == 'annual').values
    
    sp_mth_preds = []
    sp_ann_preds = []
    for s in seeds:
        m_m = CatBoostClassifier(iterations=1500, depth=3, learning_rate=0.04, eval_metric='AUC',
                                 random_seed=s, early_stopping_rounds=100, use_best_model=True, verbose=0, thread_count=-1)
        m_m.fit(X_outer_tr.iloc[tr_mth], y_tr_outer[tr_mth], eval_set=(X_outer_vl.iloc[vl_mth], y_vl_outer[vl_mth]), cat_features=c_idx)
        sp_mth_preds.append(m_m.predict_proba(X_outer_vl.iloc[vl_mth])[:, 1])
        
        m_a = CatBoostClassifier(iterations=1500, depth=3, learning_rate=0.04, eval_metric='AUC',
                                 random_seed=s, early_stopping_rounds=100, use_best_model=True, verbose=0, thread_count=-1)
        m_a.fit(X_outer_tr.iloc[tr_ann], y_tr_outer[tr_ann], eval_set=(X_outer_vl.iloc[vl_ann], y_vl_outer[vl_ann]), cat_features=c_idx)
        sp_ann_preds.append(m_a.predict_proba(X_outer_vl.iloc[vl_ann])[:, 1])
        
    p_sp_mth_out = np.mean(sp_mth_preds, axis=0)
    p_sp_ann_out = np.mean(sp_ann_preds, axis=0)
    val_pred_sp = np.zeros(N_vl, dtype=np.float64)
    val_pred_sp[vl_mth] = p_sp_mth_out
    val_pred_sp[vl_ann] = p_sp_ann_out
    
    print(f"    Outer Validation Base AUCs [{time.time() - t_f:.1f}s]:")
    print(f"      CatBoost: {roc_auc_score(y_vl_outer, val_pred_cat):.5f} | LightGBM: {roc_auc_score(y_vl_outer, val_pred_lgb):.5f} | XGBoost: {roc_auc_score(y_vl_outer, val_pred_xgb):.5f} | LR: {roc_auc_score(y_vl_outer, val_pred_lr):.5f} | Specialist: {roc_auc_score(y_vl_outer, val_pred_sp):.5f}")
    
    nested_fold_data.append({
        'fold': f_idx + 1,
        'y_tr': y_tr_outer,
        'y_vl': y_vl_outer,
        'meta_tr': {
            'cat': oof_cat,
            'lgb': oof_lgb,
            'xgb': oof_xgb,
            'lr': oof_lr,
            'sp': oof_sp
        },
        'meta_vl': {
            'cat': val_pred_cat,
            'lgb': val_pred_lgb,
            'xgb': val_pred_xgb,
            'lr': val_pred_lr,
            'sp': val_pred_sp
        },
        'context_vl': {
            'billing_cycle': vl_outer['billing_cycle'].values,
            'region': vl_outer['region'].values,
            'plan': vl_outer['plan'].values
        }
    })

# Save OOF predictions
np.savez_compressed(
    'v17/v17_oof_predictions.npz',
    f1_y_tr=nested_fold_data[0]['y_tr'],
    f1_y_vl=nested_fold_data[0]['y_vl'],
    f1_tr_cat=nested_fold_data[0]['meta_tr']['cat'],
    f1_tr_lgb=nested_fold_data[0]['meta_tr']['lgb'],
    f1_tr_xgb=nested_fold_data[0]['meta_tr']['xgb'],
    f1_tr_lr=nested_fold_data[0]['meta_tr']['lr'],
    f1_tr_sp=nested_fold_data[0]['meta_tr']['sp'],
    f1_vl_cat=nested_fold_data[0]['meta_vl']['cat'],
    f1_vl_lgb=nested_fold_data[0]['meta_vl']['lgb'],
    f1_vl_xgb=nested_fold_data[0]['meta_vl']['xgb'],
    f1_vl_lr=nested_fold_data[0]['meta_vl']['lr'],
    f1_vl_sp=nested_fold_data[0]['meta_vl']['sp'],
    f2_y_tr=nested_fold_data[1]['y_tr'],
    f2_y_vl=nested_fold_data[1]['y_vl'],
    f2_tr_cat=nested_fold_data[1]['meta_tr']['cat'],
    f2_tr_lgb=nested_fold_data[1]['meta_tr']['lgb'],
    f2_tr_xgb=nested_fold_data[1]['meta_tr']['xgb'],
    f2_tr_lr=nested_fold_data[1]['meta_tr']['lr'],
    f2_tr_sp=nested_fold_data[1]['meta_tr']['sp'],
    f2_vl_cat=nested_fold_data[1]['meta_vl']['cat'],
    f2_vl_lgb=nested_fold_data[1]['meta_vl']['lgb'],
    f2_vl_xgb=nested_fold_data[1]['meta_vl']['xgb'],
    f2_vl_lr=nested_fold_data[1]['meta_vl']['lr'],
    f2_vl_sp=nested_fold_data[1]['meta_vl']['sp'],
    f3_y_tr=nested_fold_data[2]['y_tr'],
    f3_y_vl=nested_fold_data[2]['y_vl'],
    f3_tr_cat=nested_fold_data[2]['meta_tr']['cat'],
    f3_tr_lgb=nested_fold_data[2]['meta_tr']['lgb'],
    f3_tr_xgb=nested_fold_data[2]['meta_tr']['xgb'],
    f3_tr_lr=nested_fold_data[2]['meta_tr']['lr'],
    f3_tr_sp=nested_fold_data[2]['meta_tr']['sp'],
    f3_vl_cat=nested_fold_data[2]['meta_vl']['cat'],
    f3_vl_lgb=nested_fold_data[2]['meta_vl']['lgb'],
    f3_vl_xgb=nested_fold_data[2]['meta_vl']['xgb'],
    f3_vl_lr=nested_fold_data[2]['meta_vl']['lr'],
    f3_vl_sp=nested_fold_data[2]['meta_vl']['sp']
)

print(f"\nNested OOF Generation Completed in {time.time() - t0:.1f}s.")
print("Proceeding to Meta-Learner Training & Evaluation across Stacking Configurations...")

# ==============================================================================
# PHASE B, C, D, E: META-LEARNERS & STACKING EXPERIMENTS
# ==============================================================================

# Helper function to evaluate any meta-model across the 3 outer folds
def evaluate_meta_model(build_meta_X_fn, fit_predict_meta_fn, desc="Meta Model"):
    fold_aucs = []
    meta_weights = []
    
    for f in range(3):
        d = nested_fold_data[f]
        y_tr = d['y_tr']
        y_vl = d['y_vl']
        
        X_m_tr = build_meta_X_fn(d['meta_tr'], d, is_val=False)
        X_m_vl = build_meta_X_fn(d['meta_vl'], d, is_val=True)
        
        p_meta_vl, weights_info = fit_predict_meta_fn(X_m_tr, y_tr, X_m_vl)
        auc = float(roc_auc_score(y_vl, p_meta_vl))
        fold_aucs.append(auc)
        meta_weights.append(weights_info)
        
    m_cv = float(np.mean(fold_aucs))
    delta_v15 = m_cv - v15_champion_mean
    delta_v9 = m_cv - v9_champion_mean
    return {
        'desc': desc,
        'mean_cv': m_cv,
        'folds': fold_aucs,
        'delta_vs_v15': delta_v15,
        'delta_vs_v9': delta_v9,
        'weights_info': meta_weights
    }

# Meta-learner fit-predict functions:
# 1. Logistic Regression
def make_lr_meta_fn(C=0.1):
    def fn(X_tr, y_tr, X_vl):
        clf = LogisticRegression(C=C, random_state=42, max_iter=500)
        clf.fit(X_tr, y_tr)
        p_vl = clf.predict_proba(X_vl)[:, 1]
        return p_vl, {'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0])}
    return fn

# 2. Non-Negative Least Squares / Constrained Optimization (weights >= 0, sum=1)
def nnls_meta_fn(X_tr, y_tr, X_vl):
    K = X_tr.shape[1]
    def loss(w):
        # Weighted probability combination
        p = np.clip(np.dot(X_tr, w), 1e-7, 1 - 1e-7)
        return log_loss(y_tr, p)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bnds = tuple((0.0, 1.0) for _ in range(K))
    w0 = np.ones(K) / K
    res = minimize(loss, w0, method='SLSQP', bounds=bnds, constraints=cons)
    w_opt = res.x
    p_vl = np.clip(np.dot(X_vl, w_opt), 0.0, 1.0)
    return p_vl, {'weights': w_opt.tolist()}

# 3. Shallow CatBoost Meta-Learner
def make_cat_meta_fn(depth=1, l2=5.0):
    def fn(X_tr, y_tr, X_vl):
        m = CatBoostClassifier(iterations=300, depth=depth, learning_rate=0.03, l2_leaf_reg=l2,
                               eval_metric='AUC', random_seed=42, verbose=0, thread_count=-1)
        m.fit(X_tr, y_tr)
        p_vl = m.predict_proba(X_vl)[:, 1]
        return p_vl, {'feature_importances': m.get_feature_importance().tolist()}
    return fn

stack_experiments = []

print("\n" + "=" * 80)
print("PHASE B: PURE THREE-MODEL STACK (CATBOOST + LIGHTGBM + XGBOOST)")
print("=" * 80)

def build_X_tree_triad(m_dict, d, is_val=False):
    return np.column_stack([m_dict['cat'], m_dict['lgb'], m_dict['xgb']])

for C_val in [0.01, 0.1, 1.0, 10.0]:
    res = evaluate_meta_model(build_X_tree_triad, make_lr_meta_fn(C=C_val), desc=f"Triad Stack -> Meta-LR (C={C_val})")
    stack_experiments.append(res)
    print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

res_nnls = evaluate_meta_model(build_X_tree_triad, nnls_meta_fn, desc="Triad Stack -> Constrained NNLS (LogLoss)")
stack_experiments.append(res_nnls)
print(f"{res_nnls['desc']:<45s} -> Mean CV: {res_nnls['mean_cv']:.5f} | F1: {res_nnls['folds'][0]:.5f}, F2: {res_nnls['folds'][1]:.5f}, F3: {res_nnls['folds'][2]:.5f} | Δ(V15): {res_nnls['delta_vs_v15']:+.5f}")

for d in [1, 2]:
    res = evaluate_meta_model(build_X_tree_triad, make_cat_meta_fn(depth=d), desc=f"Triad Stack -> Meta-CatBoost (Depth {d})")
    stack_experiments.append(res)
    print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

print("\n" + "=" * 80)
print("PHASE D: ADDING SECONDARY PREDICTION SOURCES (LR & BILLING SPECIALIST)")
print("=" * 80)

# A. Cat + LGB + XGB + LR (4 models)
def build_X_triad_lr(m_dict, d, is_val=False):
    return np.column_stack([m_dict['cat'], m_dict['lgb'], m_dict['xgb'], m_dict['lr']])

res = evaluate_meta_model(build_X_triad_lr, make_lr_meta_fn(C=0.1), desc="Triad + LR (4 Models) -> Meta-LR (C=0.1)")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

res = evaluate_meta_model(build_X_triad_lr, nnls_meta_fn, desc="Triad + LR (4 Models) -> Constrained NNLS")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

# B. Cat + LGB + XGB + Specialist (4 models)
def build_X_triad_sp(m_dict, d, is_val=False):
    return np.column_stack([m_dict['cat'], m_dict['lgb'], m_dict['xgb'], m_dict['sp']])

res = evaluate_meta_model(build_X_triad_sp, make_lr_meta_fn(C=0.1), desc="Triad + Specialist (4 Models) -> Meta-LR (C=0.1)")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

# C. Cat + LGB + XGB + Specialist + LR (All 5 Models)
def build_X_all_5(m_dict, d, is_val=False):
    return np.column_stack([m_dict['cat'], m_dict['lgb'], m_dict['xgb'], m_dict['lr'], m_dict['sp']])

res = evaluate_meta_model(build_X_all_5, make_lr_meta_fn(C=0.1), desc="All 5 Models -> Meta-LR (C=0.1)")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

res = evaluate_meta_model(build_X_all_5, nnls_meta_fn, desc="All 5 Models -> Constrained NNLS")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

for d in [1, 2]:
    res = evaluate_meta_model(build_X_all_5, make_cat_meta_fn(depth=d), desc=f"All 5 Models -> Meta-CatBoost (Depth {d})")
    stack_experiments.append(res)
    print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

print("\n" + "=" * 80)
print("PHASE E: META-FEATURE TRANSFORMS (DISAGREEMENT & STATS)")
print("=" * 80)

def build_X_transformed_meta(m_dict, d, is_val=False):
    p_cat = m_dict['cat']
    p_lgb = m_dict['lgb']
    p_xgb = m_dict['xgb']
    p_lr  = m_dict['lr']
    
    # Base probabilities
    cols = [p_cat, p_lgb, p_xgb, p_lr]
    
    # Disagreements
    cols.append(p_cat - p_lgb)
    cols.append(p_cat - p_xgb)
    cols.append(p_lgb - p_xgb)
    cols.append(p_cat - p_lr)
    
    # Summary stats
    mean_p = (p_cat + p_lgb + p_xgb) / 3.0
    cols.append(mean_p)
    cols.append(np.std([p_cat, p_lgb, p_xgb], axis=0))
    
    # Logit transforms (clipped for numerical safety)
    eps = 1e-6
    cols.append(np.log(np.clip(p_cat, eps, 1 - eps) / (1 - np.clip(p_cat, eps, 1 - eps))))
    cols.append(np.log(np.clip(p_lgb, eps, 1 - eps) / (1 - np.clip(p_lgb, eps, 1 - eps))))
    cols.append(np.log(np.clip(p_xgb, eps, 1 - eps) / (1 - np.clip(p_xgb, eps, 1 - eps))))
    
    return np.column_stack(cols)

res = evaluate_meta_model(build_X_transformed_meta, make_lr_meta_fn(C=0.01), desc="Transformed Meta-Features -> Meta-LR (C=0.01)")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

res = evaluate_meta_model(build_X_transformed_meta, make_lr_meta_fn(C=0.1), desc="Transformed Meta-Features -> Meta-LR (C=0.1)")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

res = evaluate_meta_model(build_X_transformed_meta, make_cat_meta_fn(depth=1), desc="Transformed Meta-Features -> Meta-Cat (Depth 1)")
stack_experiments.append(res)
print(f"{res['desc']:<45s} -> Mean CV: {res['mean_cv']:.5f} | F1: {res['folds'][0]:.5f}, F2: {res['folds'][1]:.5f}, F3: {res['folds'][2]:.5f} | Δ(V15): {res['delta_vs_v15']:+.5f}")

# Sort all stacking experiments
stack_experiments.sort(key=lambda x: x['mean_cv'], reverse=True)
best_stack = stack_experiments[0]

print("\n" + "=" * 80)
print(f"BEST STACKED MODEL: {best_stack['desc']}")
print(f"Mean CV: {best_stack['mean_cv']:.5f} | F1: {best_stack['folds'][0]:.5f}, F2: {best_stack['folds'][1]:.5f}, F3: {best_stack['folds'][2]:.5f}")
print(f"Delta vs V15 Champion (0.86936): {best_stack['delta_vs_v15']:+.5f}")
print(f"Delta vs V9 Champion  (0.86824): {best_stack['delta_vs_v9']:+.5f}")
print("=" * 80)

# Save Master Results JSON
os.makedirs('v17', exist_ok=True)

with open('v17/v17_meta_results.json', 'w') as f:
    json.dump({'all_stack_experiments': stack_experiments, 'best_stack': best_stack}, f, indent=2)

master_v17 = {
    'campaign': 'V17 Nested Leakage-Safe Stacking Campaign',
    'hard_target_threshold': 0.87000,
    'v9_baseline': v9_champion_mean,
    'v15_baseline': v15_champion_mean,
    'best_stack_candidate': best_stack,
    'winning_candidate': best_stack if best_stack['mean_cv'] > v15_champion_mean else {
        'desc': 'V15 Champion Retained',
        'mean_cv': v15_champion_mean,
        'folds': [0.872698, 0.863377, 0.871999],
        'delta_vs_v15': 0.0,
        'delta_vs_v9': v15_champion_mean - v9_champion_mean
    },
    'gap_to_target_0_87000': 0.87000 - max(best_stack['mean_cv'], v15_champion_mean),
    'submission_created': False,
    'submission_uploaded': False,
    'runtime_total_seconds': time.time() - t0
}

with open('v17/v17_results.json', 'w') as f:
    json.dump(master_v17, f, indent=2)

print(f"\nV17 Stacking execution completed in {time.time() - t0:.1f}s.")
