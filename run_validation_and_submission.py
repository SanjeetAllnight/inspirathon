import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import os

print("=" * 70)
print("INSPIRATHON: TEMPORAL CV VALIDATION & SUBMISSION #1 GENERATION")
print("=" * 70)

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. LOAD FEATURES & CONFIG
# ---------------------------------------------------------------------------
print("\n[1] Loading V4 Feature Matrices and Config...")
train_df = pd.read_parquet('v4/train_features_v4.parquet')
test_df = pd.read_parquet('v4/test_features_v4.parquet')

with open('v4/v4_results.json') as f:
    v4_results = json.load(f)

v4_features = v4_results['v4_features']

with open('v3/best_config.json') as f:
    v3_config = json.load(f)

best_hp = v3_config['best_hp']
cat_names = v3_config['cat_features']
cat_idx = [v4_features.index(c) for c in cat_names if c in v4_features]

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
print(f"Features: {len(v4_features)} (Categoricals: {len(cat_idx)})")

# ---------------------------------------------------------------------------
# 2. EVENT-HISTORY CUTOFF & LOOKBACK CHECKS (Items 4 & 5)
# ---------------------------------------------------------------------------
print("\n[2] Checking Event History & Cutoff Correctness...")
events = pd.read_csv('events.csv', parse_dates=['timestamp'])
snapshots = pd.concat([
    train_df[['customer_id', 'snapshot_date']],
    test_df[['customer_id', 'snapshot_date']]
]).drop_duplicates(subset=['customer_id'])

merged_events = events.merge(snapshots, on='customer_id', how='inner')
leaks = merged_events[merged_events['timestamp'] > merged_events['snapshot_date']]
print(f"  -> Total events checked: {len(merged_events):,}")
print(f"  -> Future events found (timestamp > snapshot_date): {len(leaks)}")
assert len(leaks) == 0, "ERROR: Future events detected!"
print("  -> PASSED: Event-history cutoff is strictly respected for every customer.")

min_event_time = events['timestamp'].min()
max_event_time = events['timestamp'].max()
print(f"\n  -> Global event timestamp min: {min_event_time}")
print(f"  -> Global event timestamp max: {max_event_time}")
print("  -> Lookback window from global event start (2025-10-03) to snapshot date:")

for s_date, grp in snapshots.groupby('snapshot_date'):
    days_back = (s_date - min_event_time).days
    n_cust = len(grp)
    set_type = "TRAIN" if s_date in train_df['snapshot_date'].values else "TEST"
    print(f"     * {set_type} Snapshot {s_date.date()}: {days_back:3d} days lookback ({n_cust:5d} customers)")

del events, merged_events  # free memory

# ---------------------------------------------------------------------------
# 3. LEAKAGE-SAFE TEMPORAL CV HARNESS
# ---------------------------------------------------------------------------
print("\n[3] Running Leakage-Safe Temporal Cross-Validation...")
# Folds:
# Fold 1: Jan -> Feb
# Fold 2: Jan-Feb -> Mar
# Fold 3: Jan-Mar -> Apr
cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

seeds_cb = [42, 123, 314, 555, 777]
seeds_lgb = [42, 123, 314]

lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.04, "num_leaves": 63,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1
}

fold_results = []
fold_best_iters_cb = []
fold_best_iters_lgb = []

# Prepare LightGBM encoded data
X_train_enc = train_df[v4_features].copy()
for c in cat_names:
    if c in X_train_enc.columns:
        X_train_enc[c] = X_train_enc[c].cat.codes

for fold_idx, (fold_name, tr_mask, vl_mask) in enumerate(cv_splits):
    f_t0 = time.time()
    tr_sub = train_df[tr_mask]
    vl_sub = train_df[vl_mask]
    
    y_tr = tr_sub['churn']
    y_vl = vl_sub['churn']
    
    X_tr = tr_sub[v4_features]
    X_vl = vl_sub[v4_features]
    
    X_tr_lgb = X_train_enc[tr_mask]
    X_vl_lgb = X_train_enc[vl_mask]
    
    print(f"\n--- {fold_name} ---")
    print(f"Train: {len(tr_sub)} rows ({y_tr.mean():.4f} churn) | Val: {len(vl_sub)} rows ({y_vl.mean():.4f} churn)")
    
    # 3a. CatBoost 5-seed models
    cb_preds = []
    cb_iters = []
    for s in seeds_cb:
        m = CatBoostClassifier(
            iterations=2000, eval_metric="AUC", random_seed=s, verbose=0,
            cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
            **best_hp
        )
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        p = m.predict_proba(X_vl)[:, 1]
        cb_preds.append(p)
        cb_iters.append(m.get_best_iteration())
    
    fold_best_iters_cb.append(cb_iters)
    cb_single_pred = cb_preds[0]
    cb_ens_pred = np.mean(cb_preds, axis=0)
    
    auc_cb_single = roc_auc_score(y_vl, cb_single_pred)
    auc_cb_ens = roc_auc_score(y_vl, cb_ens_pred)
    
    # 3b. LightGBM 3-seed models
    lgb_preds = []
    lgb_iters = []
    for s in seeds_lgb:
        p_cfg = lgb_params.copy()
        p_cfg["random_state"] = s
        lm = lgb.LGBMClassifier(**p_cfg)
        lm.fit(X_tr_lgb, y_tr, eval_set=[(X_vl_lgb, y_vl)], eval_metric="auc",
               callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        lp = lm.predict_proba(X_vl_lgb)[:, 1]
        lgb_preds.append(lp)
        lgb_iters.append(lm.best_iteration_)
        
    fold_best_iters_lgb.append(lgb_iters)
    lgb_ens_pred = np.mean(lgb_preds, axis=0)
    auc_lgb_ens = roc_auc_score(y_vl, lgb_ens_pred)
    
    # 3c. V4 Blend: 0.45*CB_single + 0.50*CB_ens + 0.05*LGB_ens
    blend_pred = 0.45 * cb_single_pred + 0.50 * cb_ens_pred + 0.05 * lgb_ens_pred
    auc_blend = roc_auc_score(y_vl, blend_pred)
    
    f_elapsed = time.time() - f_t0
    print(f"  * CatBoost (Single Seed 42) AUC : {auc_cb_single:.5f} (iter: {cb_iters[0]})")
    print(f"  * CatBoost (5-Seed Ensemble) AUC: {auc_cb_ens:.5f} (mean iters: {np.mean(cb_iters):.1f})")
    print(f"  * LightGBM (3-Seed Ensemble) AUC: {auc_lgb_ens:.5f} (mean iters: {np.mean(lgb_iters):.1f})")
    print(f"  * V4 Blend AUC                  : {auc_blend:.5f} in {f_elapsed:.1f}s")
    
    fold_results.append({
        "fold": fold_name,
        "auc_cb_single": auc_cb_single,
        "auc_cb_ens": auc_cb_ens,
        "auc_lgb_ens": auc_lgb_ens,
        "auc_blend": auc_blend,
        "cb_iters": cb_iters,
        "lgb_iters": lgb_iters
    })

# CV Summary
print("\n" + "=" * 70)
print("TEMPORAL CV RESULTS SUMMARY")
print("=" * 70)
mean_cb_single = np.mean([r['auc_cb_single'] for r in fold_results])
mean_cb_ens    = np.mean([r['auc_cb_ens'] for r in fold_results])
mean_lgb_ens   = np.mean([r['auc_lgb_ens'] for r in fold_results])
mean_blend     = np.mean([r['auc_blend'] for r in fold_results])

for r in fold_results:
    print(f"  {r['fold']:26s} | CB_Single: {r['auc_cb_single']:.5f} | CB_Ens: {r['auc_cb_ens']:.5f} | LGB_Ens: {r['auc_lgb_ens']:.5f} | Blend: {r['auc_blend']:.5f}")

print("-" * 70)
print(f"  {'MEAN TEMPORAL CV AUC':26s} | CB_Single: {mean_cb_single:.5f} | CB_Ens: {mean_cb_ens:.5f} | LGB_Ens: {mean_lgb_ens:.5f} | Blend: {mean_blend:.5f}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 4. TRAIN FULL MODEL ON ALL 60,000 ROWS FOR SUBMISSION #1
# ---------------------------------------------------------------------------
print("\n[4] Training Final V4 Model on All 60,000 Training Rows (Jan-Apr)...")
# Calculate target iterations from Fold 3 (the largest fold, 45k rows) scaled for 60k rows
# Training on 33% more data: slightly scale iterations by 1.1x to allow learning on new April data
avg_fold3_cb_iter = int(np.mean(fold_best_iters_cb[2]))
target_cb_iters = int(avg_fold3_cb_iter * 1.1)

avg_fold3_lgb_iter = int(np.mean(fold_best_iters_lgb[2]))
target_lgb_iters = int(avg_fold3_lgb_iter * 1.1)

print(f"  -> Target CatBoost iterations: {target_cb_iters} (Fold 3 avg was {avg_fold3_cb_iter})")
print(f"  -> Target LightGBM iterations: {target_lgb_iters} (Fold 3 avg was {avg_fold3_lgb_iter})")

X_full = train_df[v4_features]
y_full = train_df['churn']
X_test = test_df[v4_features]

X_full_lgb = X_train_enc
X_test_lgb = test_df[v4_features].copy()
for c in cat_names:
    if c in X_test_lgb.columns:
        X_test_lgb[c] = X_test_lgb[c].cat.codes

# 4a. Train CatBoost 5 seeds on full data
cb_test_preds = []
for idx, s in enumerate(seeds_cb):
    cb_iter_s = int(fold_best_iters_cb[2][idx] * 1.1)
    print(f"  Training Full CatBoost Seed {s} ({cb_iter_s} iterations)...")
    m_full = CatBoostClassifier(
        iterations=cb_iter_s, eval_metric="AUC", random_seed=s, verbose=0,
        cat_features=cat_idx, **best_hp
    )
    m_full.fit(X_full, y_full)
    tp = m_full.predict_proba(X_test)[:, 1]
    cb_test_preds.append(tp)

cb_test_single = cb_test_preds[0]
cb_test_ens = np.mean(cb_test_preds, axis=0)

# 4b. Train LightGBM 3 seeds on full data
lgb_test_preds = []
for s in seeds_lgb:
    print(f"  Training Full LightGBM Seed {s} ({target_lgb_iters} iterations)...")
    p_cfg = lgb_params.copy()
    p_cfg["random_state"] = s
    p_cfg["n_estimators"] = target_lgb_iters
    lm_full = lgb.LGBMClassifier(**p_cfg)
    lm_full.fit(X_full_lgb, y_full)
    lp = lm_full.predict_proba(X_test_lgb)[:, 1]
    lgb_test_preds.append(lp)

lgb_test_ens = np.mean(lgb_test_preds, axis=0)

# 4c. Compute V4 Blend Test Probabilities
final_test_churn = 0.45 * cb_test_single + 0.50 * cb_test_ens + 0.05 * lgb_test_ens

# ---------------------------------------------------------------------------
# 5. GENERATE & VERIFY SUBMISSION FILE
# ---------------------------------------------------------------------------
print("\n[5] Generating and Verifying submission.csv...")
submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'churn': final_test_churn
})

sub_path = '/home/sanjeet/kaggle/inspirathon/submission.csv'
submission.to_csv(sub_path, index=False)
print(f"Saved submission to: {sub_path}")

# VERIFICATION CHECKS (Requirement 8)
test_raw = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print("\n--- Running Verification Checks ---")
check_rows = (len(submission) == 40000)
print(f"  [Check 1] Exactly 40,000 rows: {check_rows} ({len(submission)} rows)")

check_cols = (list(submission.columns) == ['customer_id', 'churn'])
print(f"  [Check 2] Columns exactly customer_id,churn: {check_cols} ({list(submission.columns)})")

check_ids = (submission['customer_id'].equals(test_raw['customer_id'])) and (submission['customer_id'].equals(sample_sub['customer_id']))
print(f"  [Check 3] Customer IDs match test.csv & sample_submission.csv in exact order: {check_ids}")

check_nan = submission['churn'].isna().sum() == 0
check_inf = np.isinf(submission['churn']).sum() == 0
check_bounds = (submission['churn'] >= 0.0).all() and (submission['churn'] <= 1.0).all()
check_probs = check_nan and check_inf and check_bounds and (submission['churn'].nunique() > 1000)
print(f"  [Check 4] Probabilities valid (not binary, no NaNs, within [0, 1]): {check_probs}")

all_checks_passed = check_rows and check_cols and check_ids and check_probs
print(f"\n---> ALL CHECKS PASSED: {all_checks_passed}")

print("\nSubmission Churn Probability Summary:")
print(submission['churn'].describe().to_string())

# Save metadata
with open('submission_metadata.json', 'w') as f:
    json.dump({
        "all_checks_passed": all_checks_passed,
        "mean_cv_auc": mean_blend,
        "fold_results": fold_results,
        "target_cb_iters": target_cb_iters,
        "target_lgb_iters": target_lgb_iters,
        "sub_path": sub_path,
        "sub_mean_churn": float(submission['churn'].mean()),
        "sub_min_churn": float(submission['churn'].min()),
        "sub_max_churn": float(submission['churn'].max())
    }, f, indent=2)

print(f"\nCompleted in {time.time() - t0:.1f} seconds.")
