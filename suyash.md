# Inspirathon Kaggle — V12 Experiment Documentation

## 1. Purpose

V12 was a focused investigation into **train/test distribution shift** and whether adapting the V9 pipeline to the test distribution could improve temporal validation and, ultimately, leaderboard performance.

V12 tested four main ideas:

1. Quantify train-vs-test covariate shift with adversarial validation.
2. Use adversarial density-ratio sample weighting to make training more representative of the test distribution.
3. Introduce CatBoost architectural diversity.
4. Test alternative ensemble/blending strategies.

**Final conclusion:** V12 did **not** improve on V9 and should **not** be submitted.

---

## 2. Current Champion Before V12

The project entered V12 with V9 as the strongest pipeline.

### V9 Champion

- 86 curated V7/base features
- Referral-code frequency + leakage-safe target encoding
- Activity-bin × billing target encoding
- Recency-bin × billing target encoding
- 5-seed CatBoost Depth 4 ensemble
- 12% regularized Logistic Regression blend
- Probability averaging

### V9 temporal CV

| Fold | AUC |
|---|---:|
| Feb | 0.87059 |
| Mar | 0.86258 |
| Apr | 0.87154 |
| **Mean** | **0.86824** |

### Actual Kaggle submission

The V9-based submission achieved:

- **Public LB: 0.868203**
- **Rank: 20**

This is the benchmark V12 had to beat.

---

# 3. Dataset / Validation Setup

## Dataset

- Train: **60,000 rows**
- Test: **40,000 rows**
- Approximately **2.6M event records**
- Train snapshots: **Jan–Apr 2026**
- Test snapshots: **May–Jun 2026**
- Target: churn within 30 days after snapshot

There is no customer overlap between train and test.

The event data is timestamped and is constructed so that events after a customer's snapshot date are not available as features.

## Temporal validation

The project uses three forward temporal folds:

1. **Fold 1:** Jan → Feb
2. **Fold 2:** Jan–Feb → Mar
3. **Fold 3:** Jan–Mar → Apr

This is intended to approximate deployment better than random cross-validation.

A recurring pattern is that **March is substantially harder** than February and April.

V9:

- Feb: 0.87059
- Mar: 0.86258
- Apr: 0.87154

---

# 4. V12 Part A — Adversarial Validation

## Goal

Determine whether the training distribution is substantially different from the eventual test distribution.

Instead of predicting churn, an adversarial classifier predicts:

- `0` = train row
- `1` = test row

Features contain no churn labels.

If the adversarial model obtains AUC significantly above 0.5, train and test have detectable covariate differences.

## Full Train vs Test Result

**Adversarial AUC: `0.78581`**

This is substantial distribution shift.

The important observation is that the shift is strongly related to **time / customer age progression**.

## Strongest shifted features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `signup_year` | 51.41% |
| 2 | `account_age_months` | 18.60% |
| 3 | `account_age_days` | 18.50% |
| 4 | `signup_dow` | 5.73% |
| 5 | `days_since_last_plan_change` | 1.41% |
| 6 | `acquisition_channel` | 1.09% |

### Interpretation

The strongest distinction between train and test is chronological progression.

Test customers are systematically further forward in time than the training snapshots.

In particular:

- signup-year composition changes
- account tenure advances
- elapsed time features naturally increase
- acquisition mix changes over time

This established that **dataset shift is real**, rather than merely being a modeling artifact.

---

# 5. Forward-Temporal Adversarial Validation

To determine whether the same phenomenon exists inside the temporal validation scheme, adversarial models were also evaluated on each forward fold.

| Fold | Train → Validation | Adversarial AUC |
|---|---|---:|
| Fold 1 | Jan → Feb | 0.57791 |
| Fold 2 | Jan–Feb → Mar | 0.70062 |
| Fold 3 | Jan–Mar → Apr | 0.74014 |

### Important observation

The distribution shift becomes progressively stronger:

**0.57791 → 0.70062 → 0.74014**

This supports the idea that later snapshots become increasingly different from the historical training population.

It also provides additional context for why the March fold is difficult.

---

# 6. V12 Part B — Covariate-Shift Sample Weighting

## Hypothesis

If the test population differs from train, perhaps the model should give greater training weight to rows that resemble the test population.

An adversarial classifier estimates the probability that a row belongs to the test distribution.

The density-ratio-style weight used was:

`w(x) = p(x) / (1 - p(x)) × N_train / N_val`

Weights were normalized to mean 1.

The experiments were performed using the V9-style feature set and evaluated with the normal forward temporal folds.

## Weighting strategies

### 1. Uniform baseline

No weighting:

`w(x) = 1.0`

Mean CV:

**0.86722**

### 2. Adversarial weights

Weights clipped to:

`0.2–5.0`

Mean CV:

**0.86725**

Lift:

**+0.00003**

### 3. Mild clipped weights

Weights clipped to:

`0.5–2.0`

Mean CV:

**0.86756**

Lift:

**+0.00034**

This was the best weighting strategy.

### 4. Damped square-root weights

Used:

`sqrt(w(x))`

followed by normalization.

Mean CV:

**0.86734**

Lift:

**+0.00012**

---

## Detailed weighting results

| Strategy | Fold 1 | Fold 2 | Fold 3 | Mean CV | Δ |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.86944 | 0.86209 | 0.87015 | 0.86722 | — |
| Clip 0.2–5.0 | 0.86931 | 0.86251 | 0.86993 | 0.86725 | +0.00003 |
| **Clip 0.5–2.0** | **0.86933** | **0.86268** | **0.87068** | **0.86756** | **+0.00034** |
| √weight | 0.86902 | 0.86240 | 0.87061 | 0.86734 | +0.00012 |

The mild clipped weighting improved the later folds:

- Fold 2: `0.86209 → 0.86268` = **+0.00059**
- Fold 3: `0.87015 → 0.87068` = **+0.00053**

However, the overall improvement was too small to change the champion.

### Why aggressive weighting failed

Wide/unclipped propensity weights increased gradient variance.

The tree model became overly influenced by a relatively small set of highly test-like observations.

This produced instability rather than useful generalization.

---

# 7. V12 Part C — CatBoost Architectural Diversity

The next experiment tested whether the remaining signal could be captured by changing CatBoost's tree construction.

The baseline architecture was:

- SymmetricTree
- Plain boosting
- Depth 4

Alternative architectures tested:

- Lossguide growth
- Ordered boosting
- Depth 6

## Results

| Architecture | Grow Policy | Boosting | Depth / Leaves | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---:|---:|---:|---:|
| **Baseline** | SymmetricTree | Plain | Depth 4 | 0.86944 | 0.86209 | 0.87015 | **0.86722** |
| Lossguide | Lossguide | Plain | max leaves 16 | 0.86807 | 0.86194 | 0.87090 | 0.86697 |
| Ordered | SymmetricTree | Ordered | Depth 4 | 0.86823 | 0.86081 | 0.87039 | 0.86647 |
| Depth 6 | SymmetricTree | Plain | Depth 6 | 0.86771 | 0.86157 | 0.87018 | 0.86649 |
| Lossguide Depth 6 | Lossguide | Plain | max leaves 32 | 0.86808 | 0.86095 | 0.86930 | 0.86611 |

### Conclusion

The original CatBoost configuration remains better.

- Lossguide: worse
- Ordered boosting: worse
- Depth 6: worse
- Lossguide Depth 6: substantially worse

The evidence favors:

**SymmetricTree + Plain boosting + Depth 4**

for this dataset.

---

# 8. V12 Part D — Ensemble and Blend Analysis

V12 also tested whether a different combination of model predictions could outperform the established V9 blend.

## Models

### CatBoost

5-seed Depth-4 ensemble:

Seeds:

- 42
- 123
- 314
- 555
- 777

### Logistic Regression

Regularized Logistic Regression using the V7/base feature set.

## Results

| Configuration | Method | Fold 1 | Fold 2 | Fold 3 | Mean CV |
|---|---|---:|---:|---:|---:|
| CatBoost D4 5-seed | Tree ensemble | 0.87023 | 0.86258 | 0.87095 | 0.86792 |
| Logistic Regression | Standalone | 0.85910 | 0.84706 | 0.85722 | 0.85446 |
| **88/12 CB + LR** | **Probability average** | **0.87059** | **0.86258** | **0.87154** | **0.86824** |
| 90/10 CB + LR | Probability average | 0.87056 | 0.86260 | 0.87148 | 0.86821 |
| 90/10 CB + LR | Rank average | 0.87079 | 0.86256 | 0.87121 | 0.86819 |
| 95/5 CB + LR | Probability average | 0.87042 | 0.86263 | 0.87127 | 0.86810 |
| 95/5 CB + LR | Rank average | 0.87057 | 0.86262 | 0.87114 | 0.86811 |

### Conclusion

The original:

**88% CatBoost + 12% Logistic Regression**

with **probability averaging** remains the best blend.

Rank averaging did not help.

The 88/12 blend also remains the overall V9 benchmark:

**0.86824**

---

# 9. Final V12 Result

## Champion comparison

| Pipeline | Mean CV |
|---|---:|
| V7 | 0.86602 |
| V9 CatBoost ensemble | 0.86792 |
| **V9 Champion Blend** | **0.86824** |
| V12 mild-weight single model | 0.86756 |
| **V12 final best blend** | **0.86824** |

### V12 best folds

The best final blend remained:

- Fold 1: **0.87059**
- Fold 2: **0.86258**
- Fold 3: **0.87154**
- Mean: **0.86824**

This is effectively identical to V9.

---

# 10. Submission Decision

**V12 should NOT be submitted.**

Reason:

- V9 already has a public leaderboard score of **0.868203**
- V12's best CV is **0.86824**
- V12 does not provide evidence of a meaningful improvement
- A submission would consume one of the remaining competition attempts without a strong reason

At this stage, V9 remains the production/submission champion.

---

# 11. What V12 Taught Us

V12 was still valuable because it answered several important questions.

## Finding 1 — Train/test shift is real

Adversarial AUC:

**0.78581**

The model can strongly distinguish train from test.

Therefore, blindly assuming train and test are identically distributed is incorrect.

## Finding 2 — Shift is mostly temporal

The largest shift features are:

- `signup_year`
- `account_age_months`
- `account_age_days`

This points toward chronological / lifecycle distribution shift.

## Finding 3 — Simple covariate weighting is not enough

Mild weighting helped:

**+0.00034**

but this did not translate into a better final ensemble.

Therefore, the missing leaderboard performance is probably not solved simply by reweighting training rows.

## Finding 4 — CatBoost architecture is already well matched

Depth 4 SymmetricTree + Plain boosting remains superior to the tested alternatives.

## Finding 5 — Probability blending is still best

The 88/12 CatBoost + Logistic Regression blend remains the strongest tested combination.

## Finding 6 — March is an important unresolved problem

March remains substantially weaker than February and April.

This may indicate a different temporal regime, label-generation behavior, or distribution shift that deserves dedicated investigation.

---

# 12. Current Project Baseline for Future Work

Any new experiment should be compared against:

### Model baseline

**V9 Champion**

- 90 features
- 5-seed CatBoost Depth 4
- Logistic Regression
- 88/12 probability blend

### Validation baseline

**Mean temporal CV: 0.86824**

Fold scores:

- Feb: 0.87059
- Mar: 0.86258
- Apr: 0.87154

### Real-world leaderboard baseline

**Public LB: 0.868203**

**Rank: 20**

Any new approach should ideally demonstrate a meaningful improvement over this before consuming a Kaggle submission.

---

# 13. Experiments That Should Not Be Repeated Without a New Hypothesis

The following directions have already shown weak or negative results in the project:

- Simply increasing referral-code smoothing
- Referral × engagement multiplicative features
- Referral × billing/profile interactions
- Additional referral frequency flags
- Deeper CatBoost
- Lossguide CatBoost
- Ordered CatBoost
- Rank averaging
- Aggressive adversarial weighting
- Generic handcrafted feature multiplication
- Large numbers of arbitrary temporal/activity ratios

A new experiment should have a clear reason why it can capture signal that these approaches cannot.

---

# 14. V13 Direction

The strongest clue coming out of V12 is:

> **The train/test distribution is substantially different, and the difference is strongly temporal, but simple test-distribution reweighting does not solve it.**

Therefore, the next investigation should focus on **how the churn relationship itself changes over time**, rather than only matching feature distributions.

High-value hypotheses for V13+ include:

1. Concept drift / changing feature-target relationships.
2. Snapshot-specific churn regimes.
3. Why March behaves differently from February and April.
4. Better future-like validation approximating May/June.
5. Customer lifecycle / state-transition representations from event sequences.
6. More principled temporal target encoding.
7. A genuinely different model family that can complement CatBoost.
8. Hidden structure in the event-generation process.
9. Better use of referral networks without noisy explicit interactions.

The goal should be **meaningful AUC improvement per unit of experimentation time**, not simply increasing model complexity.

---

# 15. Team Handoff Checklist

Before another contributor starts experimentation, they should understand:

- **V9 is the current champion.**
- Do not overwrite or invalidate V9 artifacts.
- V12 did not improve the leaderboard baseline.
- `0.86824` mean temporal CV is the key offline benchmark.
- `0.868203` public LB is the key real-world benchmark.
- March is the weakest temporal fold.
- Train/test adversarial AUC is `0.78581`.
- The strongest distribution shift is chronological/lifecycle related.
- Mild adversarial weighting gave only `+0.00034` and was not enough.
- CatBoost Depth 4 SymmetricTree + Plain remains preferred.
- 88/12 CatBoost + Logistic probability blending remains preferred.
- Do not spend a submission on V12.
- New work should target a genuinely new hypothesis.

---

## Quick Reference

| Item | Current Best |
|---|---|
| Best pipeline | V9 Champion |
| Mean temporal CV | **0.86824** |
| Feb CV | **0.87059** |
| Mar CV | **0.86258** |
| Apr CV | **0.87154** |
| Public LB | **0.868203** |
| Current rank | **20** |
| Best CatBoost | 5-seed Depth 4 |
| Best blend | 88% CB + 12% LR |
| Best blend method | Probability averaging |
| Train/Test adversarial AUC | **0.78581** |
| Best V12 weighting | Clip weights 0.5–2.0 |
| Best V12 weighted single model | **0.86756** |
| V12 final best CV | **0.86824** |
| V12 submission? | **NO** |
