import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

print("=" * 80)
print("DEEP INVESTIGATION: SLOPES, ACTIVE DAYS & MODEL DIVERSITY")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v5/train_features_v5.parquet')
test_df = pd.read_parquet('v5/test_features_v5.parquet')

with open('v5/v5_results.json') as f:
    v5_res = json.load(f)

v5_features = v5_res['v5_features']
catboost_hp = v5_res['best_v5_hp']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Load scratch parquets
df_b1 = pd.read_parquet('scratch_b1.parquet')
df_b2 = pd.read_parquet('scratch_b2.parquet')

train_df = train_df.merge(df_b1, on='customer_id', how='left')
train_df = train_df.merge(df_b2, on='customer_id', how='left')
test_df = test_df.merge(df_b1, on='customer_id', how='left')
test_df = test_df.merge(df_b2, on='customer_id', how='left')

# Also compute 12-week metrics from raw data to see if 12w is even better!
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date'])
events_raw = pd.read_csv('events.csv', parse_dates=['timestamp'])

events = events_raw.merge(train_raw[['customer_id', 'snapshot_date']], on='customer_id', how='inner')
events = events[events['timestamp'] <= events['snapshot_date']].copy()
events['days_before'] = (events['snapshot_date'] - events['timestamp']).dt.total_seconds() / 86400.0
events['event_date'] = events['timestamp'].dt.date
events['is_session'] = (events['event_type'] == 'session').astype(np.int8)
events['week_idx'] = np.floor(events['days_before'] / 7.0).astype(int) + 1

# 12 weeks aggregations
events_12w = events[(events['week_idx'] >= 1) & (events['week_idx'] <= 12)]
w_agg_12 = events_12w.groupby(['customer_id', 'week_idx']).agg(
    sess = ('is_session', 'sum'),
    act = ('event_date', 'nunique')
).unstack(level='week_idx').fillna(0)

# Matrix across 12 weeks
all_custs = train_df['customer_id'].values
sess_12_mat = np.zeros((len(all_custs), 12))
act_12_mat = np.zeros((len(all_custs), 12))

c_to_idx = {c: i for i, c in enumerate(all_custs)}
if ('sess', 1) in w_agg_12.columns:
    for w in range(1, 13):
        if ('sess', w) in w_agg_12.columns:
            s_series = w_agg_12[('sess', w)]
            a_series = w_agg_12[('act', w)]
            matched_c = s_series.index.intersection(train_df['customer_id'])
            row_indices = [c_to_idx[c] for c in matched_c]
            sess_12_mat[row_indices, w - 1] = s_series.loc[matched_c].values
            act_12_mat[row_indices, w - 1] = a_series.loc[matched_c].values

# 12w slope
weights_12w = np.array([(i - 6.5) / 143.0 for i in range(1, 13)]) # sum(t - t_bar)^2 = 143
train_df['session_slope_12w'] = np.dot(sess_12_mat[:, ::-1], weights_12w)
train_df['activedays_slope_12w'] = np.dot(act_12_mat[:, ::-1], weights_12w)

# 6w slope
weights_6w = np.array([(i - 3.5) / 17.5 for i in range(1, 7)]) # sum(t - t_bar)^2 = 17.5
train_df['session_slope_6w'] = np.dot(sess_12_mat[:, 5::-1], weights_6w)

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

def eval_cv(feature_list, hp, seed=42):
    c_idx = [feature_list.index(c) for c in cat_names if c in feature_list]
    fold_aucs = []
    
    for fold_name, tr_mask, vl_mask in cv_splits:
        tr_sub = train_df[tr_mask]
        vl_sub = train_df[vl_mask]
        
        m = CatBoostClassifier(
            iterations=1500, eval_metric="AUC", random_seed=seed, verbose=0,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **hp
        )
        m.fit(tr_sub[feature_list], tr_sub['churn'], eval_set=(vl_sub[feature_list], vl_sub['churn']))
        preds = m.predict_proba(vl_sub[feature_list])[:, 1]
        auc = roc_auc_score(vl_sub['churn'], preds)
        fold_aucs.append(auc)
        
    return fold_aucs, np.mean(fold_aucs)

base_mean = v5_res['v5_single_cb_mean_cv']
print(f"V5 Baseline Mean: {base_mean:.5f}")

# Test 1: session_slope_8w alone on V5
f1, m1 = eval_cv(v5_features + ['session_slope_8w'], catboost_hp)
print(f"\n1. V5 + session_slope_8w alone -> Mean: {m1:.5f} (Δ: {m1 - base_mean:+.5f}) | F1: {f1[0]:.5f}, F2: {f1[1]:.5f}, F3: {f1[2]:.5f}")

# Test 2: session_slope_6w alone on V5
f2, m2 = eval_cv(v5_features + ['session_slope_6w'], catboost_hp)
print(f"2. V5 + session_slope_6w alone -> Mean: {m2:.5f} (Δ: {m2 - base_mean:+.5f}) | F1: {f2[0]:.5f}, F2: {f2[1]:.5f}, F3: {f2[2]:.5f}")

# Test 3: session_slope_12w alone on V5
f3, m3 = eval_cv(v5_features + ['session_slope_12w'], catboost_hp)
print(f"3. V5 + session_slope_12w alone-> Mean: {m3:.5f} (Δ: {m3 - base_mean:+.5f}) | F1: {f3[0]:.5f}, F2: {f3[1]:.5f}, F3: {f3[2]:.5f}")

# Test 4: V5 + active_days_w1..w8 + session_slope_8w
act_feats = [f'active_days_w{w}' for w in range(1, 9)]
f4, m4 = eval_cv(v5_features + act_feats + ['session_slope_8w'], catboost_hp)
print(f"4. V5 + active_days(8) + session_slope_8w -> Mean: {m4:.5f} (Δ: {m4 - base_mean:+.5f}) | F1: {f4[0]:.5f}, F2: {f4[1]:.5f}, F3: {f4[2]:.5f}")

# Test 5: What if we also add activedays_slope_8w?
f5, m5 = eval_cv(v5_features + act_feats + ['session_slope_8w', 'activedays_slope_8w'], catboost_hp)
print(f"5. V5 + active_days(8) + both slopes -> Mean: {m5:.5f} (Δ: {m5 - base_mean:+.5f}) | F1: {f5[0]:.5f}, F2: {f5[1]:.5f}, F3: {f5[2]:.5f}")

# Test 6: What if we use both session_slope_8w and session_slope_12w?
f6, m6 = eval_cv(v5_features + act_feats + ['session_slope_8w', 'session_slope_12w'], catboost_hp)
print(f"6. V5 + active_days(8) + slope_8w + slope_12w -> Mean: {m6:.5f} (Δ: {m6 - base_mean:+.5f}) | F1: {f6[0]:.5f}, F2: {f6[1]:.5f}, F3: {f6[2]:.5f}")

# Test 7: What if we add session_slope_8w + session_slope_4w?
f7, m7 = eval_cv(v5_features + act_feats + ['session_slope_8w', 'session_slope_4w'], catboost_hp)
print(f"7. V5 + active_days(8) + slope_8w + slope_4w  -> Mean: {m7:.5f} (Δ: {m7 - base_mean:+.5f}) | F1: {f7[0]:.5f}, F2: {f7[1]:.5f}, F3: {f7[2]:.5f}")

# Select the highest CV feature set
candidate_sets = [
    ("Config 1 (V5 + slope_8w)", v5_features + ['session_slope_8w'], m1, f1),
    ("Config 4 (V5 + act_days_8w + slope_8w)", v5_features + act_feats + ['session_slope_8w'], m4, f4),
    ("Config 5 (V5 + act_days_8w + both slopes)", v5_features + act_feats + ['session_slope_8w', 'activedays_slope_8w'], m5, f5),
    ("Config 6 (V5 + act_days_8w + slope_8w + slope_12w)", v5_features + act_feats + ['session_slope_8w', 'session_slope_12w'], m6, f6),
    ("Config 7 (V5 + act_days_8w + slope_8w + slope_4w)", v5_features + act_feats + ['session_slope_8w', 'session_slope_4w'], m7, f7),
]
candidate_sets.sort(key=lambda x: x[2], reverse=True)
winner_name, winner_features, winner_mean, winner_folds = candidate_sets[0]

print("\n" + "=" * 80)
print(f"WINNING FEATURE CONFIGURATION: {winner_name}")
print(f"Mean CV: {winner_mean:.5f} (Δ: {winner_mean - base_mean:+.5f})")
print(f"Folds: {[round(x, 5) for x in winner_folds]}")
print(f"Total features: {len(winner_features)}")
print("=" * 80)

# Evaluate 5-Seed CatBoost Ensemble on Winner
print("\n--- Evaluating 5-Seed CatBoost Ensemble on Winner ---")
seeds = [42, 123, 314, 555, 777]
c_idx = [winner_features.index(c) for c in cat_names if c in winner_features]

fold_ens_aucs = []
for fold_name, tr_mask, vl_mask in cv_splits:
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    
    seed_preds = []
    for s in seeds:
        m = CatBoostClassifier(
            iterations=1500, eval_metric="AUC", random_seed=s, verbose=0,
            cat_features=c_idx, early_stopping_rounds=100, use_best_model=True,
            **catboost_hp
        )
        m.fit(tr_sub[winner_features], tr_sub['churn'], eval_set=(vl_sub[winner_features], vl_sub['churn']))
        seed_preds.append(m.predict_proba(vl_sub[winner_features])[:, 1])
        
    ens_pred = np.mean(seed_preds, axis=0)
    ens_auc = roc_auc_score(vl_sub['churn'], ens_pred)
    fold_ens_aucs.append(ens_auc)
    print(f"  {fold_name}: 5-Seed Ensemble AUC = {ens_auc:.5f}")

mean_cb_ens = np.mean(fold_ens_aucs)
print(f"\nV6 5-Seed CatBoost Ensemble Mean CV: {mean_cb_ens:.5f} (vs V5 5-seed: {v5_res['v5_5seed_mean_cv']:.5f}, Δ: {mean_cb_ens - v5_res['v5_5seed_mean_cv']:+.5f})")

# Evaluate Alternative Model: LightGBM for Diversity
print("\n--- Testing LightGBM on Winner Features for Diversity ---")
lgb_encoded = train_df[winner_features].copy()
for c in cat_names:
    if c in lgb_encoded.columns:
        lgb_encoded[c] = lgb_encoded[c].cat.codes

lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.04, "num_leaves": 63,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

lgb_folds = []
blend_folds = []

for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    
    # LGB
    lm = lgb.LGBMClassifier(**lgb_params)
    lm.fit(
        lgb_encoded[tr_mask], tr_sub['churn'],
        eval_set=[(lgb_encoded[vl_mask], vl_sub['churn'])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    lp = lm.predict_proba(lgb_encoded[vl_mask])[:, 1]
    lgb_auc = roc_auc_score(vl_sub['churn'], lp)
    lgb_folds.append(lgb_auc)
    
    # Get the CB ensemble pred for this fold
    # Blend 0.90 CB Ensemble + 0.10 LGB
    cb_ep = seed_preds[0] # wait, recompute ens_pred for this fold
    # We already have ens_pred stored from the previous loop if we compute together:
    # Let's compute blend
print(f"LightGBM Single Model Folds: {[round(x, 5) for x in lgb_folds]} | Mean: {np.mean(lgb_folds):.5f}")

# Save detailed results
os.makedirs('v6', exist_ok=True)
with open('v6/v6_deep_results.json', 'w') as f:
    json.dump({
        "winner_name": winner_name,
        "winner_features": winner_features,
        "winner_single_mean": winner_mean,
        "winner_single_folds": winner_folds,
        "winner_5seed_mean": mean_cb_ens,
        "winner_5seed_folds": fold_ens_aucs,
        "v5_5seed_mean": v5_res['v5_5seed_mean_cv'],
        "improvement_over_v5_ensemble": mean_cb_ens - v5_res['v5_5seed_mean_cv'],
        "lgb_single_mean": float(np.mean(lgb_folds)),
        "lgb_single_folds": lgb_folds
    }, f, indent=2)

print("\nResults saved to v6/v6_deep_results.json")
