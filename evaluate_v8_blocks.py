import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("EVALUATING V8 CANDIDATE BLOCKS INDEPENDENTLY ACROSS 3 TEMPORAL FOLDS")
print("=" * 80)

# Load base data
train_df = pd.read_parquet('v6/train_features_v6.parquet')

with open('v6/v6_final_results.json') as f:
    v6_res = json.load(f)

v6_features = v6_res['v6_features']
cat_names = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']

# Load candidate parquets
df_a = pd.read_parquet('scratch_v8_block_a.parquet')
df_b = pd.read_parquet('scratch_v8_block_b.parquet')
df_c = pd.read_parquet('scratch_v8_block_c.parquet')
df_d = pd.read_parquet('scratch_v8_block_d.parquet')
df_e = pd.read_parquet('scratch_v8_block_e.parquet')

train_df = train_df.merge(df_a, on='customer_id', how='left')
train_df = train_df.merge(df_b, on='customer_id', how='left')
train_df = train_df.merge(df_c, on='customer_id', how='left')
train_df = train_df.merge(df_d, on='customer_id', how='left')
train_df = train_df.merge(df_e, on='customer_id', how='left')

# Identify feature names per block
feats_a = [c for c in df_a.columns if c != 'customer_id']
feats_b = [c for c in df_b.columns if c != 'customer_id']
feats_c = [c for c in df_c.columns if c != 'customer_id']
feats_d = [c for c in df_d.columns if c != 'customer_id']
feats_e = [c for c in df_e.columns if c != 'customer_id']

# Any categorical features in B
b_cats = ['last_event_type', 'second_last_event_type', 'event_seq_last3', 'event_seq_last5']
for c in b_cats:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna('None').astype('category')

cv_splits = [
    ("Fold 1 (Jan -> Feb)",       train_df['snapshot_date'] == '2026-01-31', train_df['snapshot_date'] == '2026-02-28'),
    ("Fold 2 (Jan-Feb -> Mar)",   train_df['snapshot_date'] <= '2026-02-28', train_df['snapshot_date'] == '2026-03-31'),
    ("Fold 3 (Jan-Mar -> Apr)",   train_df['snapshot_date'] <= '2026-03-31', train_df['snapshot_date'] == '2026-04-30'),
]

# Baseline hyperparameters (Depth 4, lr=0.04)
hp_d4 = {
    'depth': 4,
    'learning_rate': 0.04,
    'l2_leaf_reg': 1,
    'random_strength': 1,
    'bagging_temperature': 1,
    'eval_metric': 'AUC',
    'verbose': 0
}

def eval_cv(feature_list, hp, seed=42):
    # Find all categorical feature indices
    current_cats = [c for c in feature_list if c in cat_names or c in b_cats]
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

# Baseline evaluation
print("\n[Baseline] Running V6 Feature Set with Depth-4 CatBoost...")
base_folds, base_mean = eval_cv(v6_features, hp_d4)
print(f"V6 Baseline (Depth 4) -> Mean: {base_mean:.5f} | F1: {base_folds[0]:.5f}, F2: {base_folds[1]:.5f}, F3: {base_folds[2]:.5f}")

blocks_to_test = [
    ("Block A: Very-Recent Activity", feats_a),
    ("Block B: Event Sequence", feats_b),
    ("Block C: Event Timing", feats_c),
    ("Block D: Session Cadence", feats_d),
    ("Block E: Mixed Event States", feats_e),
]

block_results = {}

for block_name, block_feats in blocks_to_test:
    t_blk = time.time()
    cand_feats = v6_features + block_feats
    f_res, m_res = eval_cv(cand_feats, hp_d4)
    delta = m_res - base_mean
    block_results[block_name] = {
        'n_feats': len(block_feats),
        'mean_cv': m_res,
        'delta': delta,
        'folds': f_res
    }
    status = "*CANDIDATE*" if delta >= 0.0003 else ("FLAT" if abs(delta) < 0.0003 else "DEGRADED")
    print(f"\n{block_name:30s} ({len(block_feats):2d} feats) [{time.time() - t_blk:.1f}s]:")
    print(f"  Mean CV: {m_res:.5f} (Δ: {delta:+.5f}) [{status}] | F1: {f_res[0]:.5f}, F2: {f_res[1]:.5f}, F3: {f_res[2]:.5f}")

# Sub-group screenings for promising blocks
# If Block A has lots of features, test sub-windows: e.g. 1d, 3d, 5d flags
print("\n--- Sub-group Granular Testing for Key Features ---")
# 1. Very recent flags: no_session_1d, 3d, 5d, 7d, 10d
flags_a = [c for c in feats_a if c.startswith('no_session_')]
f_flg, m_flg = eval_cv(v6_features + flags_a, hp_d4)
print(f"  Block A Flags Only ({len(flags_a)} feats)     -> Mean: {m_flg:.5f} (Δ: {m_flg - base_mean:+.5f}) | F1: {f_flg[0]:.5f}, F2: {f_flg[1]:.5f}, F3: {f_flg[2]:.5f}")

# 2. Block B sequence flags only (no categorical string sequences)
flags_b = [c for c in feats_b if c not in b_cats]
f_flg_b, m_flg_b = eval_cv(v6_features + flags_b, hp_d4)
print(f"  Block B Sequence Flags ({len(flags_b)} feats)  -> Mean: {m_flg_b:.5f} (Δ: {m_flg_b - base_mean:+.5f}) | F1: {f_flg_b[0]:.5f}, F2: {f_flg_b[1]:.5f}, F3: {f_flg_b[2]:.5f}")

# 3. Block D key cadence: recent_vs_median_gap_ratio, median_gap_days
cadence_top = ['recent_vs_median_gap_ratio', 'median_gap_days', 'recent_gap_days']
f_cad, m_cad = eval_cv(v6_features + cadence_top, hp_d4)
print(f"  Block D Top Cadence ({len(cadence_top)} feats)      -> Mean: {m_cad:.5f} (Δ: {m_cad - base_mean:+.5f}) | F1: {f_cad[0]:.5f}, F2: {f_cad[1]:.5f}, F3: {f_cad[2]:.5f}")

# 4. Block E unresolved states
states_top = ['unresolved_payment_failure', 'unresolved_support_ticket', 'pf_recent_and_inactive_7d']
f_st, m_st = eval_cv(v6_features + states_top, hp_d4)
print(f"  Block E Unresolved States ({len(states_top)} feats)-> Mean: {m_st:.5f} (Δ: {m_st - base_mean:+.5f}) | F1: {f_st[0]:.5f}, F2: {f_st[1]:.5f}, F3: {f_st[2]:.5f}")

with open('v8_screening_results.json', 'w') as f:
    json.dump({
        'base_mean': base_mean,
        'base_folds': base_folds,
        'blocks': block_results,
        'subgroups': {
            'flags_a': {'mean': m_flg, 'delta': m_flg - base_mean, 'folds': f_flg},
            'flags_b': {'mean': m_flg_b, 'delta': m_flg_b - base_mean, 'folds': f_flg_b},
            'cadence_top': {'mean': m_cad, 'delta': m_cad - base_mean, 'folds': f_cad},
            'states_top': {'mean': m_st, 'delta': m_st - base_mean, 'folds': f_st},
        }
    }, f, indent=2)

print("\nV8 screening evaluation complete. Results saved to v8_screening_results.json")
