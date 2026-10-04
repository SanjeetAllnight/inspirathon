# V17 Campaign Final Report: Nested Leakage-Safe Stacking

**Project**: Kaggle Customer Churn Prediction (`/home/sanjeet/kaggle/inspirathon`)  
**Campaign**: V17 — Nested Leakage-Safe Out-Of-Fold (OOF) Stacking  
**Date**: October 4, 2026  
**Primary Target**: Mean Temporal CV $\ge 0.87000$ (Outer Folds 1, 2, 3)  
**Submission Rule**: STRICT NO-SUBMISSION (Zero submissions used, no `submission.csv` created)

---

## 1. V15 Verified Baseline

Before running any V17 stacking experiments, the V15 verified champion and individual constituent performance were loaded from `v15/v15_results.json` and `v15/v15_model_results.json` and saved to `v17/v17_baseline.json`:

| Model / Configuration | Fold 1 (Feb) | Fold 2 (Mar) | Fold 3 (Apr) | Mean Temporal CV | Delta vs V9 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V9 Champion** | 0.87102 | 0.86249 | 0.87121 | **0.86824** | Baseline |
| **V13 Depth-3 Standalone** | 0.87102 | 0.86288 | 0.87115 | **0.86835** | +0.00011 |
| **V14 Champion** | 0.87247 | 0.86311 | 0.87199 | **0.86919** | +0.00095 |
| **V15 Champion (Fixed Blend)** | **0.87270** | **0.86338** | **0.87200** | **0.86936** | **+0.00112** |

The gap to the hard target of $0.87000$ was **$0.00064$** ($6.4$ bps).

---

## 2. Exact Outer Fold Definitions

The outer temporal validation scheme remains identical to the forward temporal validation setup established in V7–V15:

* **Outer Fold 1**:
  * **Train**: January 2025 signups ($N = 14,993$ rows)
  * **Validation**: February 2025 signups ($N = 15,180$ rows)
* **Outer Fold 2**:
  * **Train**: January + February 2025 signups ($N = 30,173$ rows)
  * **Validation**: March 2025 signups ($N = 15,014$ rows)
* **Outer Fold 3**:
  * **Train**: January + February + March 2025 signups ($N = 45,187$ rows)
  * **Validation**: April 2025 signups ($N = 14,813$ rows)

Total training records across the 4 months: $60,000$.

---

## 3. Exact Inner OOF Stacking Architecture

To guarantee strictly leakage-safe meta-learning, in-sample predictions were forbidden:

```
For EACH OUTER TEMPORAL FOLD:
    OUTER TRAIN PARTITION
        │
        ▼
    5-Fold StratifiedKFold Partitioning of Outer Train
        │
        ├─► For each Inner Fold:
        │     - Fit Preprocessing, Frequency Maps, and Target Encodings on Inner-Train ONLY
        │     - Fit Base Learners (CatBoost, LightGBM, XGBoost, Specialist, LR) on Inner-Train ONLY
        │     - Predict Inner-Validation rows (unseen by transformers and models)
        │
        ▼
    Full Inner-OOF Prediction Matrix on Outer Train (N_train × M_models)
        │
        ▼
    Train Meta-Learner ONLY on Inner-OOF Predictions
        │
    [In Parallel]:
    Retrain Base Learners on 100% of Outer Train (using outer-train transformers only)
        │
        ▼
    Predict Outer Validation Partition (completely unseen)
        │
        ▼
    Feed Outer Validation Base Predictions into Trained Meta-Learner
        │
        ▼
    Evaluate Final Outer Validation AUC
```

Every target-dependent feature (including referral code Bayesian smoothed target encoding and billing cycle interaction target encodings) was strictly computed out-of-fold inside each inner split, preventing any target information from leaking into meta-features.

---

## 4. Individual Base Model Validation

Base models were trained on 100% of outer train and evaluated on outer validation:

| Base Model | Architecture & Config | Fold 1 | Fold 2 | Fold 3 | Mean CV | $\Delta$ vs V15 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | Depth 3, 5 seeds (42, 123, 314, 555, 777) | 0.87102 | 0.86288 | 0.87115 | **0.86835** | -0.00101 |
| **LightGBM** | Depth 3, max_depth 3, num_leaves 7, lr 0.03 | 0.87152 | 0.86210 | 0.87059 | **0.86807** | -0.00129 |
| **XGBoost** | Depth 3, max_depth 3, lr 0.03, colsample 0.8 | 0.87040 | 0.86214 | 0.87078 | **0.86777** | -0.00159 |
| **Specialist**| Monthly vs Annual Billing Segment Model | 0.87054 | 0.86150 | 0.86922 | **0.86709** | -0.00227 |
| **Logistic Reg**| Regularized LogReg ($C=0.1$) on scaled dense | 0.85910 | 0.84706 | 0.85722 | **0.85446** | -0.01490 |

CatBoost Depth-3 remains the highest standalone base learner ($0.86835$), matching the exact standalone benchmark from V13.

---

## 5. Base Model Diversity & Correlation Analysis

Pairwise Pearson correlation of predictions and residuals on outer validation ($N = 45,007$ total validation predictions):

### Prediction Correlations
| Model Pair | Correlation ($r$) | Mean Abs Difference | Max Abs Difference | 95th Pct Difference |
| :--- | :--- | :--- | :--- | :--- |
| **CatBoost vs LightGBM** | 0.9868 | 0.0139 | 0.3655 | 0.0592 |
| **CatBoost vs XGBoost** | 0.9866 | 0.0140 | 0.4079 | 0.0590 |
| **LightGBM vs XGBoost** | 0.9982 | 0.0051 | 0.2088 | 0.0211 |
| **CatBoost vs Specialist** | 0.9857 | 0.0136 | 0.5148 | 0.0552 |
| **CatBoost vs Logistic Reg**| 0.9250 | 0.0346 | 0.9824 | 0.1380 |
| **LightGBM vs Specialist** | 0.9809 | 0.0167 | 0.5934 | 0.0685 |
| **Specialist vs LogReg** | 0.9163 | 0.0369 | 0.9671 | 0.1455 |

### Residual Correlations ($y - \hat{p}$)
* $\text{Corr}(e_{\text{cat}}, e_{\text{lgb}}) = 0.9951$
* $\text{Corr}(e_{\text{cat}}, e_{\text{xgb}}) = 0.9951$
* $\text{Corr}(e_{\text{lgb}}, e_{\text{xgb}}) = 0.9993$ (GBDT architectures make virtually identical residual errors)
* $\text{Corr}(e_{\text{cat}}, e_{\text{sp}}) = 0.9947$
* $\text{Corr}(e_{\text{cat}}, e_{\text{lr}}) = 0.9735$ (Linear model provides meaningful orthogonal residual diversity)

---

## 6. Phase B: Pure Three-Model Stack (CatBoost + LightGBM + XGBoost)

The core three-model stack was evaluated using several meta-learner specifications trained strictly on inner-OOF probability outputs:

| Meta-Learner | Regularization / Config | Fold 1 | Fold 2 | Fold 3 | Mean CV | $\Delta$ vs V15 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta-Logistic Regression** | $C = 0.01$ | 0.87179 | 0.86299 | 0.87140 | **0.86873** | -0.00063 |
| **Meta-Logistic Regression** | $C = 0.1$ | 0.87179 | 0.86299 | 0.87140 | **0.86873** | -0.00063 |
| **Meta-Logistic Regression** | $C = 1.0$ | 0.87179 | 0.86299 | 0.87140 | **0.86873** | -0.00063 |
| **Constrained NNLS** | Non-negative sum-to-1 weights | 0.87180 | 0.86300 | 0.87140 | **0.86873** | -0.00063 |
| **Meta-CatBoost (Depth 1)** | 100 trees, lr 0.03, l2 5.0 | 0.87143 | 0.86292 | 0.87114 | **0.86850** | -0.00086 |
| **Meta-CatBoost (Depth 2)** | 100 trees, lr 0.03, l2 5.0 | 0.87121 | 0.86285 | 0.87126 | **0.86844** | -0.00092 |

**Key Finding**:
The learned three-tree stack reached **$0.86873$**, outperforming all individual standalone base models (+38 bps over CatBoost standalone). However, tree-based meta-learners slightly overfit the 3-dimensional meta-space, performing worse than the regularized linear meta-model.

---

## 7. Phase D: Secondary Prediction Sources (Specialist & Logistic Regression)

Next, the secondary prediction sources identified in V14 were introduced into the meta-feature matrix:

| Model Combination | Meta-Learner | Fold 1 | Fold 2 | Fold 3 | Mean CV | $\Delta$ vs V15 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Triad + Specialist (4 models)** | Meta-LR ($C=0.1$) | 0.87237 | 0.86316 | 0.87138 | **0.86897** | -0.00039 |
| **Triad + LogReg (4 models)** | Meta-LR ($C=0.1$) | 0.87176 | 0.86261 | 0.87186 | **0.86874** | -0.00062 |
| **All 5 Models (Full Zoo)** | **Meta-LR ($C=0.1$)** | **0.87246** | **0.86293** | **0.87187** | **0.86909** | **-0.00027** |
| **All 5 Models (Full Zoo)** | Constrained NNLS | 0.87224 | 0.86304 | 0.87180 | **0.86903** | -0.00033 |
| **All 5 Models (Full Zoo)** | Meta-CatBoost (Depth 1) | 0.87185 | 0.86269 | 0.87137 | **0.86864** | -0.00072 |
| **All 5 Models (Full Zoo)** | Meta-CatBoost (Depth 2) | 0.87186 | 0.86263 | 0.87168 | **0.86872** | -0.00064 |

**Key Finding**:
Adding both the billing specialist and Logistic Regression into the meta-learner provides consistent lifts:
$$\text{Triad (0.86873)} \longrightarrow \text{Triad + Specialist (0.86897)} \longrightarrow \text{All 5 Models (0.86909)}$$
The Meta-LR model with all 5 base inputs achieved Fold 1: **0.87246**, Fold 2: **0.86293**, Fold 3: **0.87187**, Mean CV: **0.86909**.

---

## 8. Phase E: Meta-Feature Transforms

To test whether derived meta-features could provide additional signal, we enriched the 5-model probability vector with:
1. Model disagreements: $p_{\text{cat}} - p_{\text{lgb}}$, $p_{\text{cat}} - p_{\text{xgb}}$, $p_{\text{lgb}} - p_{\text{xgb}}$, $p_{\text{cat}} - p_{\text{sp}}$, $p_{\text{cat}} - p_{\text{lr}}$
2. Ensemble summary statistics: $\text{mean}(p)$, $\text{std}(p)$
3. Numerically clipped logit transforms: $\text{logit}(p_i) = \ln\left(\frac{p_i}{1 - p_i}\right)$

| Feature Space | Meta-Learner | Fold 1 | Fold 2 | Fold 3 | Mean CV | $\Delta$ vs V15 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformed Meta-Features (13 cols)** | Meta-LR ($C=0.1$) | 0.87207 | 0.86283 | 0.87166 | **0.86885** | -0.00050 |
| **Transformed Meta-Features (13 cols)** | Meta-LR ($C=0.01$) | 0.87185 | 0.86296 | 0.87153 | **0.86878** | -0.00058 |
| **Transformed Meta-Features (13 cols)** | Meta-Cat (Depth 1) | 0.87100 | 0.86249 | 0.87183 | **0.86844** | -0.00092 |

**Key Finding**:
Explicit disagreement features and logit transforms did not outperform the clean 5-probability vector. In a linear model, differences ($p_1 - p_2$) are linear combinations of the probabilities and introduce colinearity; in tree meta-learners, the expanded dimensionality led to mild overpartitioning.

---

## 9. Phase F: Optional Context Feature Injection

We tested injecting key customer context variables (`billing_cycle`, `region`, `plan`) into the Meta-LR learner alongside the 5 base predictions:

* **Result**: Mean CV plunged from **$0.86909$** down to **$0.86526$** ($-38.3$ bps).
* **Root Cause Analysis**: The base models (CatBoost, LightGBM, XGBoost) have already processed these categorical variables non-linearly with deep cross-feature interactions. Injecting raw one-hot/ordinal categories into an unregularized linear meta-model distorts the well-calibrated base probability margins across segment slices. Raw context injection was decisively rejected.

---

## 10. Meta-Learner Overfitting & Coefficient Audit

For the best meta-model (**All 5 Models $\rightarrow$ Meta-LR $C=0.1$**), coefficients were inspected across the three outer folds:

| Base Model | Fold 1 Weight | Fold 2 Weight | Fold 3 Weight | Mean Weight |
| :--- | :--- | :--- | :--- | :--- |
| **CatBoost Depth-3** | 1.104 | 1.412 | 1.415 | **1.310** |
| **LightGBM Depth-3** | 1.412 | 1.455 | 1.609 | **1.492** |
| **XGBoost Depth-3** | 1.403 | 1.471 | 1.440 | **1.438** |
| **Billing Specialist**| 1.472 | 1.365 | 1.052 | **1.296** |
| **Logistic Regression**| 1.599 | 1.416 | 1.580 | **1.532** |
| **Intercept** | -3.315 | -3.361 | -3.362 | **-3.346** |

**Observations**:
1. **Remarkable Coefficient Stability**: The weights assigned to all 5 models remain stable and balanced ($1.05 - 1.61$) across all three temporal folds. No single model dominates, and no coefficient exhibits signs of ill-conditioning or extreme oscillations.
2. **Balanced Reliance**: Despite Logistic Regression having lower individual AUC ($0.8545$), the meta-model assigns it substantial weight ($\sim 1.53$) due to its low prediction correlation ($0.916 - 0.931$) and independent linear error structure.

---

## 11. Comparison with V15 and Ensembling Synergy

### Standalone Comparison
* **V15 Champion (Fixed Blend + Segment Calibration)**: Fold 1: **0.87270**, Fold 2: **0.86338**, Fold 3: **0.87200** $\rightarrow$ Mean: **0.86936**
* **V17 Standalone Stack (All 5 $\rightarrow$ Meta-LR)**: Fold 1: **0.87246**, Fold 2: **0.86293**, Fold 3: **0.87187** $\rightarrow$ Mean: **0.86909**

### Why V15 Retains a Small Edge over Standalone V17 Stack:
V15 includes hand-tuned post-hoc segment residual calibrations discovered in V14:
* Central Region: $-0.0117$
* Annual Billing: $-0.0048$
These segment calibrations directly address localized temporal shift between training months and test months. The level-2 meta-learner was trained purely on global inner-OOF probabilities without segment adjustments, leaving a $\sim 2.7$ bps difference.

### Ensembling V15 with V17 Stack:
When the calibrated V15 prediction is blended with the learned V17 stack ($80\%$ V15 + $20\%$ V17 Stack):
* **Fold 1**: **0.87282** (highest Fold 1 score in the entire campaign)
* **Fold 2**: **0.86340**
* **Fold 3**: **0.87195**
* **Mean CV**: **0.86939** (+0.00003 over V15, +0.00115 over V9)

While this achieves the highest mean CV recorded to date, it remains below the **$0.87000$** threshold.

---

## 12. Complete Leakage & Integrity Audit

The complete audit is recorded in `v17/v17_leakage_audit.json`:

* [x] **No outer validation targets used by base models**: Base models trained strictly on outer train rows.
* [x] **No outer validation targets used by meta-learner**: Meta-learners trained strictly on inner-OOF predictions.
* [x] **No inner validation targets used by inner base models**: Inner base models trained strictly on inner train rows.
* [x] **No in-sample meta-training predictions**: 100% of meta-training inputs were generated out-of-fold.
* [x] **Zero target encoding leakage**: Target encodings re-computed on inner train for inner OOF, and outer train for outer validation.
* [x] **Row counts verified**:
  * Outer Train: F1 = 14,993, F2 = 30,173, F3 = 45,187
  * Outer Val: F1 = 15,180, F2 = 15,014, F3 = 14,813 (exact match to monthly counts)
* [x] **Prediction completeness**: Exactly 0 missing values or NaNs in all arrays.
* [x] **No Kaggle submission**: No `submission.csv` created, uploaded, or submitted.

---

## 13. Reproduction

All experiments in V17 were executed using fixed seeds (`42, 123, 314, 555, 777` for CatBoost, `42` for LightGBM/XGBoost/LR, `42` for StratifiedKFold). Re-running `run_v17_stacking.py` generates identical outputs down to machine precision.

---

## 14. Gap to Primary Target ($0.87000$)

* **Hard Target**: $0.87000$
* **Best V17 Standalone Stack**: $0.86909$ (Gap: $0.00091$)
* **Best V17 Composite Blend (V15 + Stack)**: $0.86939$ (Gap: **$0.00061$** / $6.1$ bps)

Because the hard target was not reached, **NO SUBMISSION WAS CREATED OR SPENT**.

---

## 15. Recommendations for V18

1. **Segment-Calibrated Stacking**: The primary reason V15 held an advantage over raw stacking was its manual segment calibration (Central region $-0.0117$, Annual billing $-0.0048$). V18 should formalize segment calibration as a native stage inside the Level-2 meta-learner (e.g. hierarchical isotonic calibration per segment).
2. **Multi-Scale Temporal Feature Windows**: Rather than stacking static representations, create temporal delta features across the event history (e.g., changes in user activity between day 1-14 vs 15-30 prior to snapshot) to capture behavioral acceleration.

---

## V17 STATUS BLOCK

```
V17 STATUS
V9 baseline:          0.86824
V13 baseline:         0.86835
V14 baseline:         0.86919
V15 baseline:         0.86936
Best V17:             0.86939
Delta vs V15:         +0.00003
Delta vs V9:          +0.00115
Gap to 0.87000:       0.00061
Base stack:           CatBoost + LightGBM + XGBoost + Specialist + LogReg
Meta-learner:         Regularized Logistic Regression (C=0.1)
Best meta-features:   5-model calibrated probability vector
Reproduced:           YES
Leakage audit:        PASS
Submission:           NO
Next direction:       Segment-calibrated hierarchical Level-2 stacking with multi-scale event deltas
```
