# Final Submission Report: Frozen Production Pipeline

**Competition**: Kaggle Customer Churn Prediction (`/home/sanjeet/kaggle/inspirathon`)  
**Phase**: Final Submission Checkpoint — Freeze & Submit  
**Date**: October 4, 2026  
**Final Candidate**: **80% V15 Champion + 20% V17 Stacked Model**  
**Validation Benchmark**: **0.86939 Mean Temporal CV** (Highest verified across campaign)  
**Status**: SUBMISSION GENERATED & VALIDATED FROM SELF-CONTAINED KAGGLE NOTEBOOK

---

## 1. Final Model Architecture

The submission model unites the two highest-performing methodologies developed during the campaign into an absolute freeze composite:

$$\hat{y}_{\text{final}} = 0.80 \cdot \hat{y}_{\text{V15\_champion}} + 0.20 \cdot \hat{y}_{\text{V17\_stack}}$$

```
Raw Competition CSVs (train.csv, test.csv, events.csv)
                      │
                      ▼
        90 Curated Features (86 Base + 4 Encodings)
                      │
        ┌─────────────┴─────────────────────────┐
        ▼                                       ▼
 [5-Fold Internal OOF on 60k]       [Retrain on Full 60k Train]
  - CatBoost Depth-3                 - CatBoost Depth-3 (5 seeds)
  - LightGBM Depth-3                 - LightGBM Depth-3 (5 seeds)
  - XGBoost Depth-3                  - XGBoost Depth-3 (5 seeds)
  - Billing Specialist               - Billing Specialist (5 seeds)
  - Logistic Regression              - Logistic Regression (C=0.1)
        │                                       │
        ▼                                       ▼
  5-Col OOF Probability Matrix             5-Col Test Probability Matrix
        │                                       │
        ├─► Fit Meta-LR (C=0.1)                 │
        │         │                             │
        │         └─────────────────────────────┼────────┐
        │                                       │        │
        ▼                                       ▼        ▼
  Compute Residual Calibration:          [V15 Pathway]  [V17 Pathway]
  Δ_central = -0.01334                         │              │
  Δ_annual  = -0.00276                         │        Meta-LR Predict
        │                                      │              │
        └──────────────────────────────────────┤              │
                                               ▼              ▼
                                          p_test_V15     p_test_V17
                                               │              │
                                               ▼              ▼
                                           (80% weight)   (20% weight)
                                               │              │
                                               └──────┬───────┘
                                                      ▼
                                           Final Test Probability
                                                      │
                                                      ▼
                                                submission.csv
```

---

## 2. Exact Component Weights

### Top-Level Composite
* **V15 Champion Weight**: **$0.80$**
* **V17 Stacked Learner Weight**: **$0.20$**

### V15 Champion Breakdown
* **Tree Triad Weight**: **$0.88$**
* **Linear Model (Logistic Regression) Weight**: **$0.12$**
* **Tree Triad Internal Weights**:
  * $80\%$ V14 Candidate ($75\%$ CatBoost Depth-3 + $25\%$ Billing Specialist + Residual Calibration)
  * $20\%$ LightGBM Depth-3
* **Residual Calibration**:
  * $\alpha = 0.50$
  * $\delta_{\text{central}} = -0.01334$ (applied to `region == 'central'`)
  * $\delta_{\text{annual}} = -0.00276$ (applied to `billing_cycle == 'annual'`)

### V17 Stack Breakdown
* **Meta-Learner**: Regularized Logistic Regression ($C=0.1$, intercept enabled)
* **Learned Level-2 Coefficients**:
  * CatBoost Depth-3: **$+1.484$**
  * LightGBM Depth-3: **$+1.660$**
  * XGBoost Depth-3: **$+1.491$**
  * Billing Specialist: **$+1.432$**
  * Logistic Regression: **$+1.068$**
  * Intercept: **$-3.376$**

---

## 3. Feature Count & Specifications

* **Total Feature Count**: Exactly **90 features** (86 base features + 4 leakage-safe encodings).
* **86 Base Features**:
  * Multi-window event counts, session counts, support ticket counts, payment failure counts, plan changes (windows: 7d, 14d, 30d, 60d, 90d, 120d).
  * Recencies: days since last session, support ticket, payment failure, plan change, and $k^{\text{th}}$-last sessions ($k=2, 3, 5, 10$).
  * Weekly activity profiles and slope across the 8-week lookback window.
  * Behavioral ratios: session ratios (7/30, 14/60, 30/60, 60/120), duration ratios, and event acceleration.
  * Customer profile interactions: tenure $\times$ billing cycle, tenure $\times$ plan basic, inactive lifetime fraction.
* **4 Encodings**:
  1. `ref_code_train_freq`: Referral code training frequency.
  2. `ref_code_te`: Bayesian smoothed target encoding for referral codes ($m=20.0$).
  3. `act_bin_billing_te`: Coarse session activity $\times$ billing cycle target encoding ($m=15.0$).
  4. `rec_bin_billing_te`: Coarse session recency $\times$ billing cycle target encoding ($m=15.0$).

---

## 4. Base Model Configurations (Retrained on Full 60,000 Rows)

1. **CatBoost Depth-3**:
   * Hyperparameters: `depth=3, learning_rate=0.04, l2_leaf_reg=1, random_strength=1, bagging_temperature=1, iterations=600`
   * 5 Seeds: `[42, 123, 314, 555, 777]`
2. **LightGBM Depth-3**:
   * Hyperparameters: `max_depth=3, num_leaves=7, learning_rate=0.04, n_estimators=450, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_alpha=0.1, reg_lambda=1.0`
   * 5 Seeds: `[42, 123, 314, 555, 777]`
3. **XGBoost Depth-3**:
   * Hyperparameters: `max_depth=3, learning_rate=0.04, n_estimators=450, tree_method='hist', colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0`
   * 5 Seeds: `[42, 123, 314, 555, 777]`
4. **Billing Specialist**:
   * Two independent segment models (Monthly billing cohort and Annual billing cohort).
   * CatBoost Depth-3, 5 Seeds: `[42, 123, 314, 555, 777]`, `iterations=600`.
5. **Regularized Logistic Regression**:
   * Standard scaled numerical features + One-Hot Encoded categorical variables.
   * `C=0.1, max_iter=1000`.

---

## 5. Internal OOF Stacking Procedure

To strictly avoid in-sample bias during meta-model training:
1. 5-Fold `KFold(n_splits=5, shuffle=True, random_state=42)` partitioned the 60,000 training rows into 5 folds (48,000 train / 12,000 validation).
2. Inside each fold, base models were fitted strictly on the 48,000 partition and predicted the 12,000 held-out rows.
3. Target encodings within each fold were computed out-of-fold.
4. The assembled $60,000 \times 5$ probability matrix served as the training set for the Level-2 Logistic Regression meta-learner.
5. All 5 base models were subsequently refit on 100% of the training data to produce the final test features.

---

## 6. Submission Structure Validation & Integrity Checks

All 6 automated checks passed cleanly during notebook execution:

| Check Description | Required Specification | Verified Output | Status |
| :--- | :--- | :--- | :--- |
| **Row Count** | Exactly 40,000 | 40,000 rows | **PASS** |
| **Column Names** | `['customer_id', 'churn']` | `['customer_id', 'churn']` | **PASS** |
| **Customer ID Ordering** | Exact match to `test.csv` & `sample_submission.csv` | 100% identical row-for-row | **PASS** |
| **NaN / Inf Values** | Exactly 0 | 0 nulls, 0 infs | **PASS** |
| **Probability Bounds** | Continuous within $[0, 1]$ | Min: 0.00692, Max: 0.96005 | **PASS** |
| **Distribution Sanity** | Non-trivial distribution | Mean: 0.12757, Std: 0.17054 | **PASS** |

### Decile Distribution
* **10th percentile**: $0.0155$
* **50th percentile (Median)**: $0.0490$
* **90th percentile**: $0.4170$

---

## 7. Kaggle Notebook Artifacts

The final pipeline is fully self-contained in two interchangeable formats:
* **Standalone Python Script**: [`kaggle_notebook_final.py`](file:///home/sanjeet/kaggle/inspirathon/kaggle_notebook_final.py)
* **Kaggle Jupyter Notebook**: [`kaggle_notebook_final.ipynb`](file:///home/sanjeet/kaggle/inspirathon/kaggle_notebook_final.ipynb)
* **Notebook Builder Script**: [`build_kaggle_notebook_final.py`](file:///home/sanjeet/kaggle/inspirathon/build_kaggle_notebook_final.py)
* **Output File**: [`submission.csv`](file:///home/sanjeet/kaggle/inspirathon/submission.csv) (1,203,665 bytes)

---

## 8. Validation Reference Score

* **Fold 1 (Jan $\rightarrow$ Feb)**: **0.87282**
* **Fold 2 (Jan-Feb $\rightarrow$ Mar)**: **0.86340**
* **Fold 3 (Jan-Mar $\rightarrow$ Apr)**: **0.87195**
* **Mean Temporal CV**: **0.86939** (Benchmark: $+0.00003$ over V15, $+0.00115$ over V9)

---

## 9. Final Decision & Status

* **Status**: Complete freeze. No further experimental versions (V18+) will be run.
* **Submission Candidate**: $80\%$ V15 Champion + $20\%$ V17 Stack.
* **Single Submission Spent**: Yes. Exactly one submission checkpoint executed.
