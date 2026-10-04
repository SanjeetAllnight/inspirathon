"""
pipeline_v2.py – Improved churn-prediction pipeline
====================================================
Builds on the v1 baseline (0.86595) with richer feature engineering,
multi-model comparison (CatBoost / LightGBM / XGBoost), and blending.

Strict temporal validation: train Jan–Mar 2026, validate Apr 2026.
"""

import pandas as pd
import numpy as np
import time, warnings, os
warnings.filterwarnings("ignore")

T0 = time.time()

# ── 0. Paths ─────────────────────────────────────────────────────────────────
ROOT = "/home/sanjeet/kaggle/inspirathon"
OUT  = os.path.join(ROOT, "v2")
os.makedirs(OUT, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading CSVs …")
train = pd.read_csv(f"{ROOT}/train.csv", parse_dates=["snapshot_date", "signup_date"])
test  = pd.read_csv(f"{ROOT}/test.csv",  parse_dates=["snapshot_date", "signup_date"])
events = pd.read_csv(f"{ROOT}/events.csv", parse_dates=["timestamp"])

# ── 2. Churn-rate analysis (printed, not used as features) ───────────────────
print("\n═══ Churn-rate analysis ═══")
for col in ["snapshot_date", "plan", "billing_cycle", "region", "age_band", "payment_method"]:
    cr = train.groupby(col)["churn"].mean().sort_values(ascending=False)
    print(f"\n{col}:\n{cr.to_string()}")

# ── 3. Attach snapshot_date to events & filter future ────────────────────────
print("\nPreparing events …")
snapshots = pd.concat([
    train[["customer_id", "snapshot_date"]],
    test[["customer_id", "snapshot_date"]],
]).drop_duplicates(subset=["customer_id"])

events = events.merge(snapshots, on="customer_id", how="inner")
events["days_before"] = (events["snapshot_date"] - events["timestamp"]).dt.total_seconds() / 86400.0
events = events[events["days_before"] >= 0].copy()

# Convenience columns
events["event_date"] = events["timestamp"].dt.date
events["is_session"]        = (events["event_type"] == "session").astype(np.int8)
events["is_support"]        = (events["event_type"] == "support_ticket").astype(np.int8)
events["is_payment_failed"] = (events["event_type"] == "payment_failed").astype(np.int8)
events["is_plan_change"]    = (events["event_type"] == "plan_change").astype(np.int8)
events["is_upgrade"]        = ((events["event_type"] == "plan_change") & (events["value"] > 0)).astype(np.int8)
events["is_downgrade"]      = ((events["event_type"] == "plan_change") & (events["value"] < 0)).astype(np.int8)
events["session_dur"]       = np.where(events["is_session"] == 1, events["value"], np.nan)

# Pre-compute window flags
WINDOWS = [7, 14, 30, 60, 90, 120]
for w in WINDOWS:
    events[f"in_{w}d"] = (events["days_before"] <= w).astype(np.int8)

# ── 4. Build event features ─────────────────────────────────────────────────
print("Building event features …")

def agg_features(df):
    """Aggregate event-level data into customer-level features."""
    g = df.groupby("customer_id")

    # --- BASIC COUNTS ---
    basic = g.agg(
        total_events       = ("event_type", "count"),
        session_count      = ("is_session", "sum"),
        support_count      = ("is_support", "sum"),
        pf_count           = ("is_payment_failed", "sum"),
        plan_change_count  = ("is_plan_change", "sum"),
        upgrade_count      = ("is_upgrade", "sum"),
        downgrade_count    = ("is_downgrade", "sum"),
        days_since_last_event = ("days_before", "min"),
        # session duration stats
        sess_dur_total     = ("session_dur", "sum"),
        sess_dur_mean      = ("session_dur", "mean"),
        sess_dur_median    = ("session_dur", "median"),
        sess_dur_max       = ("session_dur", "max"),
        sess_dur_std       = ("session_dur", "std"),
        # distinct event dates
        n_distinct_dates   = ("event_date", "nunique"),
    )

    # --- DAYS SINCE LAST for each event type ---
    for etype, flag in [("session", "is_session"), ("support_ticket", "is_support"),
                        ("payment_failed", "is_payment_failed"), ("plan_change", "is_plan_change")]:
        sub = df[df[flag] == 1]
        col_name = f"days_since_last_{etype.replace('_', '')}"
        if len(sub) > 0:
            last = sub.groupby("customer_id")["days_before"].min().rename(col_name)
            basic = basic.join(last, how="left")
        else:
            basic[col_name] = np.nan

    # --- WINDOW AGGREGATIONS ---
    for w in WINDOWS:
        wdf = df[df[f"in_{w}d"] == 1]
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

    # --- LONGEST INACTIVITY GAP ---
    def _longest_gap(g_series):
        sorted_days = np.sort(g_series.values)
        if len(sorted_days) < 2:
            return np.nan
        diffs = np.diff(sorted_days)
        return diffs.max()

    # We approximate by using days_before (continuous), sorted desc → compute diff
    gaps = df.groupby("customer_id")["days_before"].apply(_longest_gap).rename("longest_gap_days")
    basic = basic.join(gaps, how="left")

    return basic

event_feats = agg_features(events)

# ── 5. Derived ratios & trends (vectorised on the feature df) ────────────────
print("Computing derived features …")

def add_derived(df):
    eps = 1e-6

    # --- Activity change between consecutive windows ---
    for a, b in [(7, 14), (14, 30), (30, 60), (60, 90), (90, 120)]:
        if f"sessions_{a}d" in df.columns and f"sessions_{b}d" in df.columns:
            df[f"sess_delta_{a}_{b}"] = df[f"sessions_{b}d"] - df[f"sessions_{a}d"]
        if f"events_{a}d" in df.columns and f"events_{b}d" in df.columns:
            df[f"evt_delta_{a}_{b}"] = df[f"events_{b}d"] - df[f"events_{a}d"]

    # --- Ratios: sessions ---
    for a, b in [(7, 30), (30, 60), (60, 120)]:
        if f"sessions_{a}d" in df.columns and f"sessions_{b}d" in df.columns:
            df[f"sess_ratio_{a}_{b}"] = df[f"sessions_{a}d"] / (df[f"sessions_{b}d"] + eps)
    
    # --- Ratios: events ---
    for a, b in [(7, 30), (30, 60), (60, 120)]:
        if f"events_{a}d" in df.columns and f"events_{b}d" in df.columns:
            df[f"evt_ratio_{a}_{b}"] = df[f"events_{a}d"] / (df[f"events_{b}d"] + eps)

    # --- Ratios: session duration trends ---
    for a, b in [(7, 30), (30, 60), (60, 120)]:
        ca, cb = f"sess_dur_total_{a}d", f"sess_dur_total_{b}d"
        if ca in df.columns and cb in df.columns:
            df[f"dur_ratio_{a}_{b}"] = df[ca] / (df[cb] + eps)

    # --- Acceleration: (ratio_7_30 - ratio_30_60) ---
    if "sess_ratio_7_30" in df.columns and "sess_ratio_30_60" in df.columns:
        df["sess_accel"] = df["sess_ratio_7_30"] - df["sess_ratio_30_60"]
    if "evt_ratio_7_30" in df.columns and "evt_ratio_30_60" in df.columns:
        df["evt_accel"] = df["evt_ratio_7_30"] - df["evt_ratio_30_60"]

    # --- support / session ratios ---
    df["support_per_session"] = df["support_count"] / (df["session_count"] + eps)
    df["pf_per_session"]      = df["pf_count"]      / (df["session_count"] + eps)
    for w in [30, 60]:
        if f"support_{w}d" in df.columns and f"sessions_{w}d" in df.columns:
            df[f"support_per_sess_{w}d"] = df[f"support_{w}d"] / (df[f"sessions_{w}d"] + eps)
        if f"pf_{w}d" in df.columns and f"sessions_{w}d" in df.columns:
            df[f"pf_per_sess_{w}d"] = df[f"pf_{w}d"] / (df[f"sessions_{w}d"] + eps)

    # --- Engagement consistency ---
    for w in [7, 30, 60, 120]:
        ad = f"active_days_{w}d"
        if ad in df.columns:
            df[f"sess_per_activeday_{w}d"]  = df[f"sessions_{w}d"] / (df[ad] + eps)
            df[f"dur_per_activeday_{w}d"]   = df[f"sess_dur_total_{w}d"] / (df[ad] + eps)

    return df

event_feats = add_derived(event_feats)

# ── 6. Profile features ─────────────────────────────────────────────────────
print("Enriching profiles …")

def enrich_profile(df, ef):
    df = df.copy()
    df = df.merge(ef, on="customer_id", how="left")

    # Tenure
    df["account_age_days"]   = (df["snapshot_date"] - df["signup_date"]).dt.days
    df["account_age_months"] = df["account_age_days"] / 30.44
    df["signup_month"]       = df["signup_date"].dt.month
    df["signup_year"]        = df["signup_date"].dt.year
    df["signup_dow"]         = df["signup_date"].dt.dayofweek
    df["is_recent_signup"]   = (df["account_age_days"] <= 30).astype(np.int8)
    df["is_very_new"]        = (df["account_age_days"] <= 7).astype(np.int8)

    # Price
    df["price_per_device"]   = df["monthly_price"] / df["num_devices"].clip(lower=1)

    # Missing value indicators
    df["is_age_missing"]     = df["age_band"].isnull().astype(np.int8)
    df["is_pm_missing"]      = df["payment_method"].isnull().astype(np.int8)
    df["has_referral"]       = df["referral_code"].notnull().astype(np.int8)
    df["has_events"]         = df["total_events"].notnull().astype(np.int8)

    # Referral code frequency encoding (no target leakage – using full train+test referral freqs)
    ref_freq = pd.concat([train["referral_code"], test["referral_code"]]).value_counts()
    df["referral_freq"] = df["referral_code"].map(ref_freq).fillna(0)

    # Fill NaN in days-since columns with account_age_days (= "never happened")
    days_cols = [c for c in df.columns if c.startswith("days_since_")]
    for c in days_cols:
        df[c] = df[c].fillna(df["account_age_days"])

    # Fill remaining event NaNs with 0
    evt_cols = [c for c in ef.columns if c != "customer_id" and c not in days_cols]
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

# ── 7. Feature list ─────────────────────────────────────────────────────────
DROP_COLS = {"customer_id", "snapshot_date", "signup_date", "referral_code", "churn"}
features  = [c for c in train_df.columns if c not in DROP_COLS]
cat_features_names = [c for c in ["plan", "billing_cycle", "region", "age_band",
                                   "acquisition_channel", "payment_method"]
                      if c in features]

print(f"\nFeature matrix: train {train_df.shape}, test {test_df.shape}")
print(f"Features: {len(features)}")

# Save
train_df.to_parquet(f"{OUT}/train_features_v2.parquet")
test_df.to_parquet(f"{OUT}/test_features_v2.parquet")
print(f"Saved to {OUT}/")

# ── 8. Temporal split ────────────────────────────────────────────────────────
val_date  = train_df["snapshot_date"].max()  # 2026-04-30
val_mask  = train_df["snapshot_date"] == val_date
tr = train_df[~val_mask]
vl = train_df[val_mask]

X_tr, y_tr = tr[features], tr["churn"]
X_vl, y_vl = vl[features], vl["churn"]

print(f"\nTrain: {len(X_tr)} rows  (→ {val_date.date()})")
print(f"Val  : {len(X_vl)} rows  (= {val_date.date()})")
print(f"Val churn rate: {y_vl.mean():.4f}")

# ── 9. Models ────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score
results = {}

# --- 9a. CatBoost ---
print("\n─── CatBoost ───")
from catboost import CatBoostClassifier, Pool

cat_idx = [features.index(c) for c in cat_features_names]

cb = CatBoostClassifier(
    iterations=2000, learning_rate=0.04, depth=6,
    l2_leaf_reg=3, random_strength=1.5,
    eval_metric="AUC", random_seed=42, verbose=200,
    cat_features=cat_idx, early_stopping_rounds=150,
    use_best_model=True,
)
t1 = time.time()
cb.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
cb_time = time.time() - t1

cb_pred = cb.predict_proba(X_vl)[:, 1]
cb_auc  = roc_auc_score(y_vl, cb_pred)
results["CatBoost"] = {"auc": cb_auc, "preds": cb_pred, "time": cb_time}
print(f"CatBoost AUC: {cb_auc:.5f}  ({cb_time:.1f}s)")

# --- 9b. LightGBM ---
print("\n─── LightGBM ───")
try:
    import lightgbm as lgb

    # LightGBM needs integer-coded categoricals
    X_tr_lgb = X_tr.copy()
    X_vl_lgb = X_vl.copy()
    for c in cat_features_names:
        X_tr_lgb[c] = X_tr_lgb[c].cat.codes
        X_vl_lgb[c] = X_vl_lgb[c].cat.codes

    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.04, num_leaves=63,
        max_depth=-1, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    t1 = time.time()
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_vl_lgb, y_vl)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],
    )
    lgb_time = time.time() - t1

    lgb_pred = lgb_model.predict_proba(X_vl_lgb)[:, 1]
    lgb_auc  = roc_auc_score(y_vl, lgb_pred)
    results["LightGBM"] = {"auc": lgb_auc, "preds": lgb_pred, "time": lgb_time}
    print(f"LightGBM AUC: {lgb_auc:.5f}  ({lgb_time:.1f}s)")
except ImportError:
    print("LightGBM not available, skipping.")

# --- 9c. XGBoost ---
print("\n─── XGBoost ───")
try:
    import xgboost as xgb

    X_tr_xgb = X_tr.copy()
    X_vl_xgb = X_vl.copy()
    for c in cat_features_names:
        X_tr_xgb[c] = X_tr_xgb[c].cat.codes
        X_vl_xgb[c] = X_vl_xgb[c].cat.codes

    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.04, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
        random_state=42, n_jobs=-1, verbosity=0,
        tree_method="hist", enable_categorical=False,
        early_stopping_rounds=150,
    )
    t1 = time.time()
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_vl_xgb, y_vl)], verbose=200)
    xgb_time = time.time() - t1

    xgb_pred = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    xgb_auc  = roc_auc_score(y_vl, xgb_pred)
    results["XGBoost"] = {"auc": xgb_auc, "preds": xgb_pred, "time": xgb_time}
    print(f"XGBoost AUC: {xgb_auc:.5f}  ({xgb_time:.1f}s)")
except ImportError:
    print("XGBoost not available, skipping.")

# ── 10. Blend ────────────────────────────────────────────────────────────────
print("\n─── Blending ───")
avail = list(results.keys())
if len(avail) >= 2:
    # Try equal-weight blend of all available
    blend_preds = np.mean([results[k]["preds"] for k in avail], axis=0)
    blend_auc   = roc_auc_score(y_vl, blend_preds)
    print(f"Equal blend ({'+'.join(avail)}): {blend_auc:.5f}")

    # Try optimised 2-way blends
    best_blend_auc = blend_auc
    best_blend_name = f"Equal({'+'.join(avail)})"
    best_blend_preds = blend_preds

    if len(avail) >= 2:
        for i, a in enumerate(avail):
            for b in avail[i+1:]:
                for w in np.arange(0.1, 1.0, 0.05):
                    bp = w * results[a]["preds"] + (1 - w) * results[b]["preds"]
                    ba = roc_auc_score(y_vl, bp)
                    if ba > best_blend_auc:
                        best_blend_auc = ba
                        best_blend_name = f"{w:.2f}*{a} + {1-w:.2f}*{b}"
                        best_blend_preds = bp

    if len(avail) == 3:
        # 3-way grid
        for w1 in np.arange(0.1, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.8 - w1 + 0.01, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                bp = (w1 * results[avail[0]]["preds"] +
                      w2 * results[avail[1]]["preds"] +
                      w3 * results[avail[2]]["preds"])
                ba = roc_auc_score(y_vl, bp)
                if ba > best_blend_auc:
                    best_blend_auc = ba
                    best_blend_name = f"{w1:.1f}*{avail[0]}+{w2:.1f}*{avail[1]}+{w3:.1f}*{avail[2]}"
                    best_blend_preds = bp

    print(f"Best blend: {best_blend_name}  →  AUC {best_blend_auc:.5f}")
else:
    best_blend_auc = results[avail[0]]["auc"]
    best_blend_preds = results[avail[0]]["preds"]
    print("Only one model available, no blending.")

# ── 11. Feature importance (CatBoost) ────────────────────────────────────────
print("\n─── Top 30 features (CatBoost importance) ───")
imp = pd.DataFrame({"feature": features, "importance": cb.get_feature_importance()})
imp = imp.sort_values("importance", ascending=False)
print(imp.head(30).to_string(index=False))

# ── 12. Summary ──────────────────────────────────────────────────────────────
baseline = 0.86595
best_single = max(results.values(), key=lambda x: x["auc"])
best_single_name = [k for k, v in results.items() if v is best_single][0]

print("\n" + "═" * 60)
print("SUMMARY")
print("═" * 60)
for k, v in results.items():
    delta = v["auc"] - baseline
    print(f"  {k:12s}  AUC = {v['auc']:.5f}  (Δ {delta:+.5f})  {v['time']:.1f}s")
print(f"\n  Best blend   AUC = {best_blend_auc:.5f}  (Δ {best_blend_auc - baseline:+.5f})")
print(f"  Best single: {best_single_name} = {best_single['auc']:.5f}")
print(f"\n  Baseline v1: {baseline:.5f}")
print(f"  Best v2:     {max(best_blend_auc, best_single['auc']):.5f}")
print(f"  Improvement: {max(best_blend_auc, best_single['auc']) - baseline:+.5f}")
print(f"\nTotal time: {time.time() - T0:.1f}s")

# Check for suspicious/leaky features
print("\n─── Leakage check ───")
suspect = []
for f in features:
    if f in ["churn"]:
        suspect.append(f)
# Any feature with AUC > 0.85 alone?
from sklearn.metrics import roc_auc_score as auc_fn
for f in imp.head(10)["feature"].values:
    col = X_vl[f]
    if col.dtype.name == "category":
        continue
    try:
        solo = auc_fn(y_vl, col.fillna(0))
        solo = max(solo, 1 - solo)  # in case negative correlation
        if solo > 0.78:
            suspect.append(f"{f} (solo AUC={solo:.4f})")
    except:
        pass

if suspect:
    print(f"  ⚠ Suspicious features: {suspect}")
else:
    print("  ✅ No obvious leakage detected.")
