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
print("V9: TRAINING 5-SEED CATBOOST ENSEMBLES & MULTI-MODEL BLENDS")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v6/train_features_v6.parquet')
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v7_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Add raw columns needed for encodings
train_df['referral_code'] = train_raw['referral_code'].values
train_df['billing_cycle'] = train_raw['billing_cycle'].values

# Leak-safe target encoding helper
def compute_oof_target_encoding(train_s, y_train, val_s, smoothing=20.0, n_splits=5, seed=42):
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

def get_v9_features_for_fold(tr_sub, y_tr, vl_sub):
    # Block 1: referral_code
    tr_ref = tr_sub['referral_code'].fillna('NO_REF').astype(str)
    vl_ref = vl_sub['referral_code'].fillna('NO_REF').astype(str)
    ref_counts = tr_ref.value_counts()
    
    oof_te_ref, val_te_ref = compute_oof_target_encoding(tr_ref, y_tr, vl_ref, smoothing=20.0)
    
    # Block 4: Coarse Groups x Billing
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

seeds = [42, 123, 314, 555, 777]

hp_d4 = {'depth': 4, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0}
hp_d6 = {'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0}

d4_preds = []
d6_preds = []
lr_preds = []
y_trues = []

d4_folds = []
d6_folds = []
lr_folds = []

print("\n--- Training 5-Seed CatBoost Ensembles (Depth 4 & Depth 6) + Logistic Regression ---")

for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    t_f = time.time()
    tr_sub = train_df[tr_mask].copy()
    vl_sub = train_df[vl_mask].copy()
    y_tr = tr_sub['churn'].values
    y_vl = vl_sub['churn'].values
    y_trues.append(y_vl)
    
    # Generate in-fold leak-safe V9 features (86 base + 4 TE = 90 features)
    X_tr, X_vl = get_v9_features_for_fold(tr_sub, y_tr, vl_sub)
    all_feats = list(X_tr.columns)
    c_idx = [all_feats.index(c) for c in cat_names if c in all_feats]
    
    # 1. Depth 4 5-Seed CatBoost
    f_d4_p = []
    for s in seeds:
        m4 = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m4.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_d4_p.append(m4.predict_proba(X_vl)[:, 1])
    ens4 = np.mean(f_d4_p, axis=0)
    d4_preds.append(ens4)
    auc4 = roc_auc_score(y_vl, ens4)
    d4_folds.append(auc4)
    
    # 2. Depth 6 5-Seed CatBoost
    f_d6_p = []
    for s in seeds:
        m6 = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx, early_stopping_rounds=100, use_best_model=True, **hp_d6)
        m6.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        f_d6_p.append(m6.predict_proba(X_vl)[:, 1])
    ens6 = np.mean(f_d6_p, axis=0)
    d6_preds.append(ens6)
    auc6 = roc_auc_score(y_vl, ens6)
    d6_folds.append(auc6)
    
    # 3. Regularized Logistic Regression on base V7 features
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
    auclr = roc_auc_score(y_vl, lp)
    lr_folds.append(auclr)
    
    print(f"{fold_name} [{time.time() - t_f:.1f}s]:")
    print(f"  Depth 4 5-Seed AUC : {auc4:.5f}")
    print(f"  Depth 6 5-Seed AUC : {auc6:.5f}")
    print(f"  Logistic Reg AUC   : {auclr:.5f}")

mean_d4 = np.mean(d4_folds)
mean_d6 = np.mean(d6_folds)
mean_lr = np.mean(lr_folds)

print("\n" + "=" * 80)
print("V9 ENSEMBLE SUMMARY ACROSS 3 FOLDS")
print("=" * 80)
print(f"V7 Baseline Best Blend CV          : 0.86602")
print(f"V9 CatBoost Depth 4 5-Seed Mean CV : {mean_d4:.5f} | F1: {d4_folds[0]:.5f}, F2: {d4_folds[1]:.5f}, F3: {d4_folds[2]:.5f}")
print(f"V9 CatBoost Depth 6 5-Seed Mean CV : {mean_d6:.5f} | F1: {d6_folds[0]:.5f}, F2: {d6_folds[1]:.5f}, F3: {d6_folds[2]:.5f}")

# Multi-depth & Multi-model Blends
print("\n--- Testing V9 Blends ---")
# 1. 50% Depth 4 + 50% Depth 6
b_d4_d6 = [roc_auc_score(y_trues[f], 0.50 * d4_preds[f] + 0.50 * d6_preds[f]) for f in range(3)]
print(f"  Blend (50% D4 + 50% D6)          -> Mean: {np.mean(b_d4_d6):.5f} | F1: {b_d4_d6[0]:.5f}, F2: {b_d4_d6[1]:.5f}, F3: {b_d4_d6[2]:.5f}")

# 2. 60% Depth 4 + 40% Depth 6
b_d4_60 = [roc_auc_score(y_trues[f], 0.60 * d4_preds[f] + 0.40 * d6_preds[f]) for f in range(3)]
print(f"  Blend (60% D4 + 40% D6)          -> Mean: {np.mean(b_d4_60):.5f} | F1: {b_d4_60[0]:.5f}, F2: {b_d4_60[1]:.5f}, F3: {b_d4_60[2]:.5f}")

# 3. Depth 4 CB + Logistic Regression
for w_lr in [0.05, 0.08, 0.10, 0.12]:
    w_cb = 1.0 - w_lr
    b = [roc_auc_score(y_trues[f], w_cb * d4_preds[f] + w_lr * lr_preds[f]) for f in range(3)]
    print(f"  Blend ({int(w_cb*100)}% D4 CB + {int(w_lr*100)}% LR)     -> Mean: {np.mean(b):.5f} | F1: {b[0]:.5f}, F2: {b[1]:.5f}, F3: {b[2]:.5f} (Δ vs V7: {np.mean(b) - 0.86602:+.5f})")

# 4. Multi-Depth + LR
for w_lr in [0.06, 0.08, 0.10]:
    w_tree = 1.0 - w_lr
    w_d4 = w_tree * 0.60
    w_d6 = w_tree * 0.40
    b = [roc_auc_score(y_trues[f], w_d4 * d4_preds[f] + w_d6 * d6_preds[f] + w_lr * lr_preds[f]) for f in range(3)]
    print(f"  Triple ({int(w_d4*100)}% D4 + {int(w_d6*100)}% D6 + {int(w_lr*100)}% LR) -> Mean: {np.mean(b):.5f} | F1: {b[0]:.5f}, F2: {b[1]:.5f}, F3: {b[2]:.5f} (Δ vs V7: {np.mean(b) - 0.86602:+.5f})")

# Save detailed results
import os
os.makedirs('v9', exist_ok=True)
with open('v9/v9_final_summary.json', 'w') as f:
    json.dump({
        'v7_benchmark': 0.86602,
        'v9_d4_5seed_mean': mean_d4,
        'v9_d4_5seed_folds': d4_folds,
        'v9_d6_5seed_mean': mean_d6,
        'v9_d6_5seed_folds': d6_folds,
        'best_d4_lr_blend_mean': float(np.mean([roc_auc_score(y_trues[f], 0.90 * d4_preds[f] + 0.10 * lr_preds[f]) for f in range(3)])),
        'best_d4_lr_blend_folds': [float(roc_auc_score(y_trues[f], 0.90 * d4_preds[f] + 0.10 * lr_preds[f])) for f in range(3)],
        'best_triple_blend_mean': float(np.mean([roc_auc_score(y_trues[f], 0.552 * d4_preds[f] + 0.368 * d6_preds[f] + 0.08 * lr_preds[f]) for f in range(3)])),
        'best_triple_blend_folds': [float(roc_auc_score(y_trues[f], 0.552 * d4_preds[f] + 0.368 * d6_preds[f] + 0.08 * lr_preds[f])) for f in range(3)],
        'net_improvement_over_v7': float(mean_d4 - 0.86602)
    }, f, indent=2)

print("\nV9 training & blending complete. Results saved to v9/v9_final_summary.json")
