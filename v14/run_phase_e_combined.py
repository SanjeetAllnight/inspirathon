import os
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score

print("=" * 80)
print("V14 PHASE E: COMBINED CANDIDATE EVALUATION & SELECTION")
print("=" * 80)

# Load predictions and effects
oof_data = np.load('v14/v14_oof_preds.npz', allow_pickle=True)
val_y = [oof_data['y_true_f1'], oof_data['y_true_f2'], oof_data['y_true_f3']]
val_p_glob = [oof_data['p_glob_f1'], oof_data['p_glob_f2'], oof_data['p_glob_f3']]
val_p_exp = [oof_data['p_exp_f1'], oof_data['p_exp_f2'], oof_data['p_exp_f3']]
val_regions = [oof_data['reg_f1'], oof_data['reg_f2'], oof_data['reg_f3']]
val_annual_m = [oof_data['ann_m_f1'], oof_data['ann_m_f2'], oof_data['ann_m_f3']]
val_monthly_m = [oof_data['mth_m_f1'], oof_data['mth_m_f2'], oof_data['mth_m_f3']]

with open('v14/v14_residual_results.json') as f:
    res_data = json.load(f)
effects = res_data['training_effects']

v9_champion_mean = 0.86824
v13_depth3_mean = 0.8683525

# Test a grid of Global/Expert blend weights and residual correction strengths
blend_weights = [1.0, 0.90, 0.85, 0.80, 0.75, 0.70, 0.50]
correction_modes = ['none', 'central_only', 'annual_only', 'joint']
alphas = [0.25, 0.50, 0.75]

candidates = []

for w in blend_weights:
    for mode in correction_modes:
        if mode == 'none':
            strength_list = [0.0]
        else:
            strength_list = alphas
            
        for a in strength_list:
            fold_aucs = []
            ann_f2 = None
            mth_f2 = None
            
            for f in range(3):
                y_v = val_y[f]
                p_base = w * val_p_glob[f] + (1.0 - w) * val_p_exp[f]
                
                is_central = (val_regions[f] == 'central')
                is_annual = val_annual_m[f]
                eff = effects[f]
                
                p_final = p_base.copy()
                if mode in ['central_only', 'joint']:
                    p_final[is_central] += a * eff['delta_central']
                if mode in ['annual_only', 'joint']:
                    p_final[is_annual] += a * eff['delta_annual']
                    
                p_final = np.clip(p_final, 0.0, 1.0)
                score = float(roc_auc_score(y_v, p_final))
                fold_aucs.append(score)
                
                if f == 1:
                    ann_m = val_annual_m[1]
                    mth_m = val_monthly_m[1]
                    ann_f2 = float(roc_auc_score(y_v[ann_m], p_final[ann_m]))
                    mth_f2 = float(roc_auc_score(y_v[mth_m], p_final[mth_m]))
                    
            mean_cv = float(np.mean(fold_aucs))
            cand = {
                'weight_global': w,
                'weight_expert': round(1.0 - w, 2),
                'correction_mode': mode,
                'correction_strength': a,
                'f1': fold_aucs[0],
                'f2': fold_aucs[1],
                'f3': fold_aucs[2],
                'mean_cv': mean_cv,
                'delta_vs_v9': mean_cv - v9_champion_mean,
                'delta_vs_depth3': mean_cv - v13_depth3_mean,
                'annual_f2': ann_f2,
                'monthly_f2': mth_f2
            }
            candidates.append(cand)

# Sort by mean CV descending
candidates.sort(key=lambda x: x['mean_cv'], reverse=True)

print("Top 10 Combined Candidates:")
print(f"{'Rank':4s} | {'Glob/Exp':10s} | {'Correction':12s} | {'Alpha':5s} | {'F1':7s} | {'F2':7s} | {'F3':7s} | {'Mean CV':7s} | {'Δ vs V9':7s} | {'Ann F2':7s}")
print("-" * 95)
for i, c in enumerate(candidates[:10]):
    desc = f"{int(c['weight_global']*100)}/{int(c['weight_expert']*100)}"
    print(f"{i+1:4d} | {desc:10s} | {c['correction_mode']:12s} | {c['correction_strength']:5.2f} | {c['f1']:.5f} | {c['f2']:.5f} | {c['f3']:.5f} | {c['mean_cv']:.5f} | {c['delta_vs_v9']:+.5f} | {c['annual_f2']:.5f}")

best_candidate = candidates[0]

output_payload = {
    'best_candidate': best_candidate,
    'top_10_candidates': candidates[:10],
    'comparison': {
        'v9_champion': {
            'mean_cv': v9_champion_mean,
            'folds': [0.87059, 0.86258, 0.87154]
        },
        'v13_depth3': {
            'mean_cv': v13_depth3_mean,
            'folds': [0.87102, 0.86288, 0.87115]
        },
        'best_v14_candidate': best_candidate
    }
}

with open('v14/v14_results.json', 'w') as f:
    json.dump(output_payload, f, indent=2)

print("\n" + "=" * 80)
print("BEST V14 CANDIDATE SPECIFICATION")
print("=" * 80)
print(f"Architecture         : {int(best_candidate['weight_global']*100)}% Global Depth-3 + {int(best_candidate['weight_expert']*100)}% Billing Expert")
print(f"Correction Mode      : {best_candidate['correction_mode']} (alpha = {best_candidate['correction_strength']})")
print(f"Fold 1 (Feb) AUC     : {best_candidate['f1']:.5f}")
print(f"Fold 2 (Mar) AUC     : {best_candidate['f2']:.5f}")
print(f"Fold 3 (Apr) AUC     : {best_candidate['f3']:.5f}")
print(f"Mean Temporal CV     : {best_candidate['mean_cv']:.5f}")
print(f"Delta vs V9 Champion : {best_candidate['delta_vs_v9']:+.5f}")
print(f"Delta vs V13 Depth-3 : {best_candidate['delta_vs_depth3']:+.5f}")
print(f"Annual F2 (March) AUC: {best_candidate['annual_f2']:.5f}")
print(f"Monthly F2 (March) AUC:{best_candidate['monthly_f2']:.5f}")
print(f"Gap to 0.87000       : {0.87000 - best_candidate['mean_cv']:.5f}")
