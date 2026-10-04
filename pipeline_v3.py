"""
pipeline_v3.py – Targeted improvement over v1 baseline (0.86595)
================================================================
Strategy:
  1. Use v2's rich feature set but prune noisy features
  2. Hyperparameter grid search on CatBoost
  3. Multi-seed ensembling to reduce variance
  4. Careful feature selection via iterative pruning
  5. Blend top configurations
"""

import pandas as pd
import numpy as np
import time, warnings, os, json
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
T0 = time.time()

ROOT = "/home/sanjeet/kaggle/inspirathon"
OUT  = os.path.join(ROOT, "v3")
os.makedirs(OUT, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading CSVs …")
train = pd.read_csv(f"{ROOT}/train.csv", parse_dates=["snapshot_date", "signup_date"])
test  = pd.read_csv(f"{ROOT}/test.csv",  parse_dates=["snapshot_date", "signup_date"])
events = pd.read_csv(f"{ROOT}/events.csv", parse_dates=["timestamp"])

# ── 2. Attach snapshot_date to events & filter future ────────────────────────
print("Preparing events …")
snapshots = pd.concat([
    train[["customer_id", "snapshot_date"]],
    test[["customer_id", "snapshot_date"]],
]).drop_duplicates(subset=["customer_id"])

events = events.merge(snapshots, on="customer_id", how="inner")
events["days_before"] = (events["snapshot_date"] - events["timestamp"]).dt.total_seconds() / 86400.0
events = events[events["days_before"] >= 0].copy()

events["event_date"] = events["timestamp"].dt.date
events["is_session"]        = (events["event_type"] == "session").astype(np.int8)
events["is_support"]        = (events["event_type"] == "support_ticket").astype(np.int8)
events["is_payment_failed"] = (events["event_type"] == "payment_failed").astype(np.int8)
events["is_plan_change"]    = (events["event_type"] == "plan_change").astype(np.int8)
events["is_upgrade"]        = ((events["event_type"] == "plan_change") & (events["value"] > 0)).astype(np.int8)
events["is_downgrade"]      = ((events["event_type"] == "plan_change") & (events["value"] < 0)).astype(np.int8)
events["session_dur"]       = np.where(events["is_session"] == 1, events["value"], np.nan)

WINDOWS = [7, 14, 30, 60, 90, 120]
for w in WINDOWS:
    events[f"in_{w}d"] = (events["days_before"] <= w).astype(np.int8)

# ── 3. Build event features ─────────────────────────────────────────────────
print("Building event features …")

def agg_features(df):
    g = df.groupby("customer_id")

    basic = g.agg(
        total_events          = ("event_type", "count"),
        session_count         = ("is_session", "sum"),
        support_count         = ("is_support", "sum"),
        pf_count              = ("is_payment_failed", "sum"),
        plan_change_count     = ("is_plan_change", "sum"),
        upgrade_count         = ("is_upgrade", "sum"),
        downgrade_count       = ("is_downgrade", "sum"),
        days_since_last_event = ("days_before", "min"),
        sess_dur_total        = ("session_dur", "sum"),
        sess_dur_mean         = ("session_dur", "mean"),
        sess_dur_median       = ("session_dur", "median"),
        sess_dur_max          = ("session_dur", "max"),
        sess_dur_std          = ("session_dur", "std"),
        n_distinct_dates      = ("event_date", "nunique"),
    )

    # Days since last for specific event types
    for etype, flag in [("session", "is_session"), ("support_ticket", "is_support"),
                        ("payment_failed", "is_payment_failed"), ("plan_change", "is_plan_change")]:
        sub = df[df[flag] == 1]
        col_name = f"days_since_last_{etype}"
        if len(sub) > 0:
            last = sub.groupby("customer_id")["days_before"].min().rename(col_name)
            basic = basic.join(last, how="left")
        else:
            basic[col_name] = np.nan

    # Window aggregations
    for w in WINDOWS:
        wdf = df[df[f"in_{w}d"] == 1]
        if len(wdf) == 0:
            for prefix in ["events", "sessions", "support", "pf", "upgrade", "downgrade",
                           "sess_dur_total", "sess_dur_mean", "active_days"]:
                basic[f"{prefix}_{w}d"] = 0
            continue
        wg = wdf.groupby("customer_id")
        wagg = wg.agg(**{
            f"events_{w}d":         ("event_type", "count"),
            f"sessions_{w}d":       ("is_session", "sum"),
            f"support_{w}d":        ("is_support", "sum"),
            f"pf_{w}d":             ("is_payment_failed", "sum"),
            f"upgrade_{w}d":        ("is_upgrade", "sum"),
            f"downgrade_{w}d":      ("is_downgrade", "sum"),
            f"sess_dur_total_{w}d": ("session_dur", "sum"),
            f"sess_dur_mean_{w}d":  ("session_dur", "mean"),
            f"active_days_{w}d":    ("event_date", "nunique"),
        })
        basic = basic.join(wagg, how="left")

    # Longest inactivity gap (using sorted unique event dates per customer)
    def _longest_gap(dates):
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < 2:
            return np.nan
        diffs = np.diff(unique_dates)
        return float(max(diffs))

    gaps = df.groupby("customer_id")["days_before"].apply(_longest_gap).rename("longest_gap_days")
    basic = basic.join(gaps, how="left")

    return basic

event_feats = agg_features(events)

# ── 4. Derived ratios & trends ───────────────────────────────────────────────
print("Computing derived features …")
eps = 1e-6
ef = event_feats

# Session ratios (the TOP feature from v2)
for a, b in [(7, 30), (14, 60), (30, 60), (60, 120), (7, 14), (30, 90)]:
    ca, cb = f"sessions_{a}d", f"sessions_{b}d"
    if ca in ef.columns and cb in ef.columns:
        ef[f"sess_ratio_{a}_{b}"] = ef[ca] / (ef[cb] + eps)

# Event ratios
for a, b in [(7, 30), (30, 60), (60, 120)]:
    ca, cb = f"events_{a}d", f"events_{b}d"
    if ca in ef.columns and cb in ef.columns:
        ef[f"evt_ratio_{a}_{b}"] = ef[ca] / (ef[cb] + eps)

# Duration ratios
for a, b in [(7, 30), (30, 60), (60, 120)]:
    ca, cb = f"sess_dur_total_{a}d", f"sess_dur_total_{b}d"
    if ca in ef.columns and cb in ef.columns:
        ef[f"dur_ratio_{a}_{b}"] = ef[ca] / (ef[cb] + eps)

# Activity deltas (change between windows)
for a, b in [(7, 14), (14, 30), (30, 60), (60, 90)]:
    ca, cb = f"sessions_{a}d", f"sessions_{b}d"
    if ca in ef.columns and cb in ef.columns:
        ef[f"sess_delta_{a}_{b}"] = ef[cb] - ef[ca]

# Acceleration
if "sess_ratio_7_30" in ef.columns and "sess_ratio_30_60" in ef.columns:
    ef["sess_accel"] = ef["sess_ratio_7_30"] - ef["sess_ratio_30_60"]

# Support/payment ratios
ef["support_per_session"] = ef["support_count"] / (ef["session_count"] + eps)
ef["pf_per_session"]      = ef["pf_count"]      / (ef["session_count"] + eps)
for w in [30, 60]:
    if f"support_{w}d" in ef.columns and f"sessions_{w}d" in ef.columns:
        ef[f"support_per_sess_{w}d"] = ef[f"support_{w}d"] / (ef[f"sessions_{w}d"] + eps)
    if f"pf_{w}d" in ef.columns and f"sessions_{w}d" in ef.columns:
        ef[f"pf_per_sess_{w}d"]     = ef[f"pf_{w}d"]      / (ef[f"sessions_{w}d"] + eps)

# Engagement consistency
for w in [7, 30, 60, 120]:
    ad = f"active_days_{w}d"
    if ad in ef.columns:
        ef[f"sess_per_activeday_{w}d"] = ef[f"sessions_{w}d"] / (ef[ad] + eps)
        ef[f"dur_per_activeday_{w}d"]  = ef[f"sess_dur_total_{w}d"] / (ef[ad] + eps)

# Active day ratios
for a, b in [(7, 30), (30, 60), (60, 120)]:
    ca, cb = f"active_days_{a}d", f"active_days_{b}d"
    if ca in ef.columns and cb in ef.columns:
        ef[f"activeday_ratio_{a}_{b}"] = ef[ca] / (ef[cb] + eps)

event_feats = ef

# ── 5. Profile features ─────────────────────────────────────────────────────
print("Enriching profiles …")

# Referral frequency map (train+test, no leakage)
ref_freq = pd.concat([train["referral_code"], test["referral_code"]]).value_counts()

def enrich_profile(df, ef):
    df = df.copy()
    df = df.merge(ef, on="customer_id", how="left")

    df["account_age_days"]   = (df["snapshot_date"] - df["signup_date"]).dt.days
    df["account_age_months"] = df["account_age_days"] / 30.44
    df["signup_month"]       = df["signup_date"].dt.month
    df["signup_year"]        = df["signup_date"].dt.year
    df["signup_dow"]         = df["signup_date"].dt.dayofweek
    df["is_recent_signup"]   = (df["account_age_days"] <= 30).astype(np.int8)
    df["is_very_new"]        = (df["account_age_days"] <= 7).astype(np.int8)
    df["snapshot_month"]     = df["snapshot_date"].dt.month

    df["price_per_device"]   = df["monthly_price"] / df["num_devices"].clip(lower=1)
    df["is_age_missing"]     = df["age_band"].isnull().astype(np.int8)
    df["is_pm_missing"]      = df["payment_method"].isnull().astype(np.int8)
    df["has_referral"]       = df["referral_code"].notnull().astype(np.int8)
    df["has_events"]         = df["total_events"].notnull().astype(np.int8)
    df["referral_freq"]      = df["referral_code"].map(ref_freq).fillna(0)

    # Interaction features (top predictors from v1/v2)
    df["tenure_x_billing"]   = df["account_age_days"] * (df["billing_cycle"] == "monthly").astype(int)
    df["tenure_x_plan_basic"] = df["account_age_days"] * (df["plan"] == "basic").astype(int)

    # Fill days-since with account_age_days (= "never happened")
    days_cols = [c for c in df.columns if c.startswith("days_since_")]
    for c in days_cols:
        df[c] = df[c].fillna(df["account_age_days"])

    # Fill remaining event NaNs with 0
    evt_cols = [c for c in ef.columns if not c.startswith("days_since_")]
    for c in evt_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Categoricals
    cat_cols = ["plan", "billing_cycle", "region", "age_band",
                "acquisition_channel", "payment_method"]
    for c in cat_cols:
        df[c] = df[c].fillna("Missing").astype("category")

    return df

train_df = enrich_profile(train, event_feats)
test_df  = enrich_profile(test, event_feats)

# ── 6. Feature list ─────────────────────────────────────────────────────────
DROP_COLS = {"customer_id", "snapshot_date", "signup_date", "referral_code", "churn"}
ALL_FEATURES = [c for c in train_df.columns if c not in DROP_COLS]
cat_names    = ["plan", "billing_cycle", "region", "age_band",
                "acquisition_channel", "payment_method"]

print(f"Total features before selection: {len(ALL_FEATURES)}")

# ── 7. Temporal split ────────────────────────────────────────────────────────
val_date = train_df["snapshot_date"].max()
val_mask = train_df["snapshot_date"] == val_date
tr_df    = train_df[~val_mask]
vl_df    = train_df[val_mask]

print(f"Train: {len(tr_df)} rows | Val: {len(vl_df)} rows (= {val_date.date()})")
print(f"Val churn rate: {vl_df['churn'].mean():.4f}")

# ── 8. Feature selection via CatBoost importance ─────────────────────────────
print("\n═══ Phase 1: Feature importance screening ═══")
from catboost import CatBoostClassifier

cat_idx_all = [ALL_FEATURES.index(c) for c in cat_names if c in ALL_FEATURES]

cb_screen = CatBoostClassifier(
    iterations=800, learning_rate=0.05, depth=6,
    eval_metric="AUC", random_seed=42, verbose=0,
    cat_features=cat_idx_all, early_stopping_rounds=100,
    use_best_model=True,
)
cb_screen.fit(tr_df[ALL_FEATURES], tr_df["churn"],
              eval_set=(vl_df[ALL_FEATURES], vl_df["churn"]))

imp = pd.DataFrame({"feature": ALL_FEATURES, "importance": cb_screen.get_feature_importance()})
imp = imp.sort_values("importance", ascending=False)

screen_auc = roc_auc_score(vl_df["churn"], cb_screen.predict_proba(vl_df[ALL_FEATURES])[:, 1])
print(f"Screening model AUC (all {len(ALL_FEATURES)} features): {screen_auc:.5f}")

# Keep features with importance >= 0.3 (prune the noise)
KEEP_THRESHOLD = 0.3
selected = imp[imp["importance"] >= KEEP_THRESHOLD]["feature"].tolist()
# Always keep categoricals even if low importance
for c in cat_names:
    if c not in selected:
        selected.append(c)

features = selected
cat_idx  = [features.index(c) for c in cat_names if c in features]
print(f"Selected {len(features)} features (dropped {len(ALL_FEATURES) - len(features)} low-importance)")

# ── 9. Hyperparameter search ────────────────────────────────────────────────
print("\n═══ Phase 2: Hyperparameter grid search ═══")

X_tr = tr_df[features]
y_tr = tr_df["churn"]
X_vl = vl_df[features]
y_vl = vl_df["churn"]

configs = [
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 1,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 3,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 1,  "random_strength": 0.5, "bagging_temperature": 0.8},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1,  "random_strength": 0.5, "bagging_temperature": 0.5},
    {"depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 2,  "random_strength": 1,   "bagging_temperature": 1},
    {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 1,  "random_strength": 0.5, "bagging_temperature": 1},
]

grid_results = []
for i, cfg in enumerate(configs):
    t1 = time.time()
    m = CatBoostClassifier(
        iterations=2000, eval_metric="AUC", random_seed=42, verbose=0,
        cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
        **cfg,
    )
    m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    preds = m.predict_proba(X_vl)[:, 1]
    auc   = roc_auc_score(y_vl, preds)
    elapsed = time.time() - t1
    best_iter = m.get_best_iteration()
    grid_results.append({"idx": i, "auc": auc, "cfg": cfg, "preds": preds,
                         "model": m, "time": elapsed, "best_iter": best_iter})
    print(f"  Config {i}: AUC={auc:.5f}  iter={best_iter}  ({elapsed:.1f}s)  {cfg}")

grid_results.sort(key=lambda x: x["auc"], reverse=True)
best_cfg = grid_results[0]
print(f"\n  Best config: #{best_cfg['idx']} AUC={best_cfg['auc']:.5f}")

# ── 10. Multi-seed ensemble of the best config ──────────────────────────────
print("\n═══ Phase 3: Multi-seed ensemble ═══")
best_hp = best_cfg["cfg"]
seeds = [42, 123, 314, 555, 777, 999, 2024, 2025, 3141, 7777]
seed_preds = []

for s in seeds:
    m = CatBoostClassifier(
        iterations=2000, eval_metric="AUC", random_seed=s, verbose=0,
        cat_features=cat_idx, early_stopping_rounds=150, use_best_model=True,
        **best_hp,
    )
    m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    p = m.predict_proba(X_vl)[:, 1]
    a = roc_auc_score(y_vl, p)
    seed_preds.append(p)
    print(f"  Seed {s}: AUC={a:.5f}  iter={m.get_best_iteration()}")

ensemble_pred = np.mean(seed_preds, axis=0)
ensemble_auc  = roc_auc_score(y_vl, ensemble_pred)
print(f"\n  Multi-seed ensemble AUC: {ensemble_auc:.5f}")

# ── 11. LightGBM & XGBoost with selected features ───────────────────────────
print("\n═══ Phase 4: LightGBM & XGBoost ═══")

# Encode cats for LGB/XGB
X_tr_enc = X_tr.copy()
X_vl_enc = X_vl.copy()
for c in cat_names:
    if c in X_tr_enc.columns:
        X_tr_enc[c] = X_tr_enc[c].cat.codes
        X_vl_enc[c] = X_vl_enc[c].cat.codes

model_preds = {"CatBoost_best": best_cfg["preds"], "CatBoost_ensemble": ensemble_pred}

try:
    import lightgbm as lgb

    lgb_configs = [
        {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31,  "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63,  "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63,  "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.5},
    ]

    best_lgb_auc = 0
    best_lgb_pred = None
    for j, lcfg in enumerate(lgb_configs):
        lm = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **lcfg)
        lm.fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)], eval_metric="auc",
               callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        lp = lm.predict_proba(X_vl_enc)[:, 1]
        la = roc_auc_score(y_vl, lp)
        print(f"  LGB config {j}: AUC={la:.5f}  iter={lm.best_iteration_}")
        if la > best_lgb_auc:
            best_lgb_auc = la
            best_lgb_pred = lp
            best_lgb_model = lm

    # Multi-seed LGB ensemble
    lgb_seed_preds = []
    for s in seeds[:5]:
        lm = lgb.LGBMClassifier(random_state=s, n_jobs=-1, verbose=-1,
                                **lgb_configs[np.argmax([r for r in [0] + [best_lgb_auc]])])
        # Use the best lgb config
        best_lcfg_idx = max(range(len(lgb_configs)),
                           key=lambda j: roc_auc_score(y_vl,
                               lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **lgb_configs[j])
                               .fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)], eval_metric="auc",
                                    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
                               .predict_proba(X_vl_enc)[:, 1]) if j == 0 else 0)
        # Simplified: just use the best config found above
        lm2 = lgb.LGBMClassifier(random_state=s, n_jobs=-1, verbose=-1, **lgb_configs[0])
        lm2.fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)], eval_metric="auc",
                callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        lgb_seed_preds.append(lm2.predict_proba(X_vl_enc)[:, 1])

    lgb_ens_pred = np.mean(lgb_seed_preds, axis=0)
    lgb_ens_auc = roc_auc_score(y_vl, lgb_ens_pred)
    print(f"  LGB multi-seed ensemble AUC: {lgb_ens_auc:.5f}")

    model_preds["LightGBM_best"] = best_lgb_pred
    model_preds["LightGBM_ensemble"] = lgb_ens_pred
except ImportError:
    print("  LightGBM not available.")

try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0, eval_metric="auc",
        random_state=42, n_jobs=-1, verbosity=0,
        tree_method="hist", early_stopping_rounds=150,
    )
    xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)], verbose=0)
    xgb_pred = xgb_model.predict_proba(X_vl_enc)[:, 1]
    xgb_auc  = roc_auc_score(y_vl, xgb_pred)
    print(f"  XGBoost AUC: {xgb_auc:.5f}  iter={xgb_model.best_iteration}")
    model_preds["XGBoost"] = xgb_pred
except ImportError:
    print("  XGBoost not available.")

# ── 12. Optimised blending ──────────────────────────────────────────────────
print("\n═══ Phase 5: Optimised blending ═══")
pred_keys = list(model_preds.keys())
pred_vals = [model_preds[k] for k in pred_keys]

# Print individual AUCs
for k in pred_keys:
    a = roc_auc_score(y_vl, model_preds[k])
    print(f"  {k:25s}  AUC = {a:.5f}")

# Greedy blend search
best_blend_auc  = 0
best_blend_name = ""
best_blend_pred = None

# Pairwise
for i in range(len(pred_keys)):
    for j in range(i + 1, len(pred_keys)):
        for w in np.arange(0.05, 1.0, 0.05):
            bp = w * pred_vals[i] + (1 - w) * pred_vals[j]
            ba = roc_auc_score(y_vl, bp)
            if ba > best_blend_auc:
                best_blend_auc  = ba
                best_blend_name = f"{w:.2f}*{pred_keys[i]} + {1-w:.2f}*{pred_keys[j]}"
                best_blend_pred = bp

# 3-way (top 3 only to keep runtime sane)
aucs = [(k, roc_auc_score(y_vl, model_preds[k])) for k in pred_keys]
aucs.sort(key=lambda x: x[1], reverse=True)
top3_keys = [x[0] for x in aucs[:3]]
top3_preds = [model_preds[k] for k in top3_keys]

for w1 in np.arange(0.1, 0.85, 0.05):
    for w2 in np.arange(0.05, 0.85 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        bp = w1 * top3_preds[0] + w2 * top3_preds[1] + w3 * top3_preds[2]
        ba = roc_auc_score(y_vl, bp)
        if ba > best_blend_auc:
            best_blend_auc  = ba
            best_blend_name = f"{w1:.2f}*{top3_keys[0]} + {w2:.2f}*{top3_keys[1]} + {w3:.2f}*{top3_keys[2]}"
            best_blend_pred = bp

print(f"\n  Best blend: {best_blend_name}")
print(f"  Best blend AUC: {best_blend_auc:.5f}")

# ── 13. Feature importance from the best single model ────────────────────────
print("\n─── Top 30 features (best CatBoost config) ───")
imp_final = pd.DataFrame({"feature": features, "importance": best_cfg["model"].get_feature_importance()})
imp_final = imp_final.sort_values("importance", ascending=False)
print(imp_final.head(30).to_string(index=False))

# ── 14. Save artefacts ──────────────────────────────────────────────────────
train_df.to_parquet(f"{OUT}/train_features_v3.parquet")
test_df.to_parquet(f"{OUT}/test_features_v3.parquet")

# Save the best config
with open(f"{OUT}/best_config.json", "w") as f:
    json.dump({
        "best_hp": best_hp,
        "features": features,
        "cat_features": cat_names,
        "screen_auc": screen_auc,
        "best_single_auc": best_cfg["auc"],
        "ensemble_auc": ensemble_auc,
        "best_blend_auc": best_blend_auc,
        "best_blend_name": best_blend_name,
    }, f, indent=2, default=str)

# Save the best catboost model
best_cfg["model"].save_model(f"{OUT}/catboost_best.cbm")
print(f"\nSaved to {OUT}/")

# ── 15. Leakage check ───────────────────────────────────────────────────────
print("\n─── Leakage check ───")
suspect = []
for f_name in imp_final.head(15)["feature"].values:
    col = X_vl[f_name]
    if col.dtype.name == "category":
        continue
    try:
        solo = roc_auc_score(y_vl, col.fillna(0))
        solo = max(solo, 1 - solo)
        if solo > 0.78:
            suspect.append(f"{f_name} (solo AUC={solo:.4f})")
    except:
        pass

if suspect:
    print(f"  ⚠ Suspicious features: {suspect}")
else:
    print("  ✅ No obvious leakage detected.")

# ── 16. Final summary ───────────────────────────────────────────────────────
baseline = 0.86595
overall_best = max(best_blend_auc, ensemble_auc, best_cfg["auc"])
print("\n" + "═" * 65)
print("FINAL SUMMARY")
print("═" * 65)
print(f"  v1 Baseline:            {baseline:.5f}")
print(f"  v3 Best single config:  {best_cfg['auc']:.5f}  (Δ {best_cfg['auc'] - baseline:+.5f})")
print(f"  v3 Multi-seed ensemble: {ensemble_auc:.5f}  (Δ {ensemble_auc - baseline:+.5f})")
print(f"  v3 Best blend:          {best_blend_auc:.5f}  (Δ {best_blend_auc - baseline:+.5f})")
print(f"  v3 Overall best:        {overall_best:.5f}  (Δ {overall_best - baseline:+.5f})")
print(f"  Features used:          {len(features)}")
print(f"\nTotal pipeline time:      {time.time() - T0:.1f}s")
