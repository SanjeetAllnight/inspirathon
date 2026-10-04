import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

print("=" * 80)
print("V8 FINE-TUNING, MULTI-DEPTH ENSEMBLING & BLEND EVALUATION")
print("=" * 80)

# Load base V6 features
train_df = pd.read_parquet('v6/train_features_v6.parquet')

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v6_features = v6_res['v6_features']
base_cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Load candidate parquets: Block B and Block D
df_b = pd.read_parquet('scratch_v8_block_b.parquet')
df_d = pd.read_parquet('scratch_v8_block_d.parquet')

train_df = train_df.merge(df_b, on='customer_id', how='left')
train_df = train_df.merge(df_d, on='customer_id', how='left')

# Prepare categorical columns
b_cats = ['last_event_type', 'second_last_event_type', 'event_seq_last3', 'event_seq_last5']
for c in b_cats:
    train_df[c] = train_df[c].fillna('None').astype('category')

all_cat_names = base_cat_names + b_cats

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

feats_b = [c for c in df_b.columns if c != 'customer_id']
cadence_top = ['recent_vs_median_gap_ratio', 'median_gap_days', 'recent_gap_days']

def eval_cv(feature_list, hp, seed=42):
    current_cats = [c for c in feature_list if c in all_cat_names]
    c_idx = [feature_list.index(c) for c in current_cats]
    fold_aucs = []
    
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        
        m = CatBoostClassifier(
            iterations=1500, random_seed=seed,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **hp
        )
        m.fit(tr_sub[feature_list], tr_sub['churn'], eval_set=(vl_sub[feature_list], vl_sub['churn']))
        preds = m.predict_proba(vl_sub[feature_list])[:, 1]
        auc = roc_auc_score(vl_sub['churn'], preds)
        fold_aucs.append(auc)
        
    return fold_aucs, np.mean(fold_aucs)

hp_d4 = {'depth': 4, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0}
hp_d6 = {'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 1, 'random_strength': 1, 'bagging_temperature': 1, 'eval_metric': 'AUC', 'verbose': 0}

# Step 1: Feature Selection Summary
print("\n[Step 1] Feature Selection Summary:")
print("  Block B (Event Sequences) independently produced Mean CV: 0.86538 (Δ: +0.00057 over baseline).")
print("  Top Cadence slightly lowered mean CV when combined with Block B (0.86519 vs 0.86538).")
v8_features = v6_features + feats_b
selected_name = "Block B (Event Sequences)"
print(f"---> Selected V8 Feature Set: {selected_name} ({len(v8_features)} total features)")

# Step 2: Train 5-Seed CatBoost Ensembles on Depth 4 and Depth 6
print("\n[Step 2] Training 5-Seed CatBoost Ensembles on Depth 4 and Depth 6...")
seeds = [42, 123, 314, 555, 777]
current_cats = [c for c in v8_features if c in all_cat_names]
c_idx_v8 = [v8_features.index(c) for c in current_cats]

preds_d4 = []
preds_d6 = []
y_trues = []

d4_ens_folds = []
d6_ens_folds = []

for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    y_vl = vl_sub['churn'].values
    y_trues.append(y_vl)
    
    # Depth 4 5-seed
    f_d4_preds = []
    for s in seeds:
        m4 = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx_v8, early_stopping_rounds=100, use_best_model=True, **hp_d4)
        m4.fit(tr_sub[v8_features], tr_sub['churn'], eval_set=(vl_sub[v8_features], y_vl))
        f_d4_preds.append(m4.predict_proba(vl_sub[v8_features])[:, 1])
    ens4 = np.mean(f_d4_preds, axis=0)
    preds_d4.append(ens4)
    d4_ens_folds.append(roc_auc_score(y_vl, ens4))
    
    # Depth 6 5-seed
    f_d6_preds = []
    for s in seeds:
        m6 = CatBoostClassifier(iterations=1500, random_seed=s, cat_features=c_idx_v8, early_stopping_rounds=100, use_best_model=True, **hp_d6)
        m6.fit(tr_sub[v8_features], tr_sub['churn'], eval_set=(vl_sub[v8_features], y_vl))
        f_d6_preds.append(m6.predict_proba(vl_sub[v8_features])[:, 1])
    ens6 = np.mean(f_d6_preds, axis=0)
    preds_d6.append(ens6)
    d6_ens_folds.append(roc_auc_score(y_vl, ens6))
    
    print(f"  {fold_name}: Depth 4 5-Seed AUC = {d4_ens_folds[-1]:.5f} | Depth 6 5-Seed AUC = {d6_ens_folds[-1]:.5f}")

mean_d4_ens = np.mean(d4_ens_folds)
mean_d6_ens = np.mean(d6_ens_folds)
print(f"\nV8 CatBoost Depth 4 5-Seed Ensemble Mean CV: {mean_d4_ens:.5f}")
print(f"V8 CatBoost Depth 6 5-Seed Ensemble Mean CV: {mean_d6_ens:.5f}")

# Step 3: Test Blends (Multi-depth & with Regularized Logistic Regression)
print("\n[Step 3] Testing Ensemble Blends...")

# Multi-depth blend (50% D4 + 50% D6)
b_multi_depth = [roc_auc_score(y_trues[f], 0.50 * preds_d4[f] + 0.50 * preds_d6[f]) for f in range(3)]
print(f"  Multi-Depth Blend (50% D4 + 50% D6) -> Mean: {np.mean(b_multi_depth):.5f} | F1: {b_multi_depth[0]:.5f}, F2: {b_multi_depth[1]:.5f}, F3: {b_multi_depth[2]:.5f}")

# 60% D4 + 40% D6
b_multi_60 = [roc_auc_score(y_trues[f], 0.60 * preds_d4[f] + 0.40 * preds_d6[f]) for f in range(3)]
print(f"  Multi-Depth Blend (60% D4 + 40% D6) -> Mean: {np.mean(b_multi_60):.5f} | F1: {b_multi_60[0]:.5f}, F2: {b_multi_60[1]:.5f}, F3: {b_multi_60[2]:.5f}")

# Logistic regression on V8 features
from sklearn.impute import SimpleImputer

num_cols = [c for c in v8_features if c not in all_cat_names]
cat_cols = [c for c in v8_features if c in all_cat_names]

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)
log_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])

lr_preds = []
lr_folds = []
for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    log_pipe.fit(tr_sub[v8_features], tr_sub['churn'])
    lp = log_pipe.predict_proba(vl_sub[v8_features])[:, 1]
    lr_preds.append(lp)
    lr_folds.append(roc_auc_score(y_trues[fold_idx], lp))

print(f"  Logistic Regression Standalone Mean CV: {np.mean(lr_folds):.5f}")

# Depth 4 + LR blend
for w_lr in [0.05, 0.08, 0.10, 0.12]:
    w_cb = 1.0 - w_lr
    b = [roc_auc_score(y_trues[f], w_cb * preds_d4[f] + w_lr * lr_preds[f]) for f in range(3)]
    print(f"  Blend ({int(w_cb*100)}% D4 CB + {int(w_lr*100)}% LR) -> Mean: {np.mean(b):.5f} | F1: {b[0]:.5f}, F2: {b[1]:.5f}, F3: {b[2]:.5f} (Δ vs D4: {np.mean(b) - mean_d4_ens:+.5f})")

# Multi-depth + LR blend (45% D4 + 45% D6 + 10% LR)
b_trip = [roc_auc_score(y_trues[f], 0.45 * preds_d4[f] + 0.45 * preds_d6[f] + 0.10 * lr_preds[f]) for f in range(3)]
print(f"  Multi-Depth Triple Blend (45% D4 + 45% D6 + 10% LR) -> Mean: {np.mean(b_trip):.5f} | F1: {b_trip[0]:.5f}, F2: {b_trip[1]:.5f}, F3: {b_trip[2]:.5f}")

# 50% D4 + 40% D6 + 10% LR
b_trip2 = [roc_auc_score(y_trues[f], 0.50 * preds_d4[f] + 0.40 * preds_d6[f] + 0.10 * lr_preds[f]) for f in range(3)]
print(f"  Multi-Depth Triple Blend (50% D4 + 40% D6 + 10% LR) -> Mean: {np.mean(b_trip2):.5f} | F1: {b_trip2[0]:.5f}, F2: {b_trip2[1]:.5f}, F3: {b_trip2[2]:.5f}")

# Step 4: Feature Importance of V8 model
print("\n[Step 4] Computing Feature Importance for V8 Model...")
full_m = CatBoostClassifier(iterations=500, random_seed=42, cat_features=c_idx_v8, **hp_d4)
full_m.fit(train_df[v8_features], train_df['churn'])
fi = pd.Series(full_m.get_feature_importance(), index=v8_features).sort_values(ascending=False)

print("\nTop 25 Most Important Features in V8:")
for i, (name, imp) in enumerate(fi.head(25).items(), 1):
    new_tag = " *** NEW V8 ***" if name in feats_b + cadence_top else ""
    print(f"  {i:2d}. {name:32s}: {imp:.3f}%{new_tag}")

print("\nImportance of All New V8 Features:")
new_v8_feats = [c for c in v8_features if c in feats_b + cadence_top]
for name in sorted(new_v8_feats, key=lambda x: fi[x], reverse=True):
    rank = list(fi.index).index(name) + 1
    print(f"  Rank {rank:2d}/{len(v8_features)}: {name:30s} (Importance: {fi[name]:.3f}%)")

# Save results
import os
os.makedirs('v8', exist_ok=True)
with open('v8/v8_final_summary.json', 'w') as f:
    json.dump({
        'v7_benchmark': 0.86602,
        'v8_features': v8_features,
        'n_features': len(v8_features),
        'd4_ens_mean': mean_d4_ens,
        'd4_ens_folds': d4_ens_folds,
        'd6_ens_mean': mean_d6_ens,
        'd6_ens_folds': d6_ens_folds,
        'multi_depth_blend_mean': float(np.mean(b_multi_60)),
        'multi_depth_blend_folds': [float(x) for x in b_multi_60],
        'best_blend_mean': float(np.mean(b_trip2)),
        'best_blend_folds': [float(x) for x in b_trip2],
        'top_features': fi.head(25).to_dict()
    }, f, indent=2)

print("\nSaved V8 final results to v8/v8_final_summary.json")
