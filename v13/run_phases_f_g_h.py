import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

print("=" * 80)
print("V13 PHASES F, G, H: ARTIFACT, LIFECYCLE & RENEWAL PHASE INVESTIGATION")
print("=" * 80)

# Load data
train_raw = pd.read_csv('train.csv', parse_dates=['snapshot_date', 'signup_date'])
test_raw = pd.read_csv('test.csv', parse_dates=['snapshot_date', 'signup_date'])
events_raw = pd.read_csv('events.csv', parse_dates=['timestamp'])

# ===========================================================================
# PHASE F: ARTIFACT / DATA-GENERATION AUDIT
# ===========================================================================
print("\n" + "=" * 80)
print("PHASE F: ARTIFACT / DATA-GENERATION AUDIT")
print("=" * 80)

audit_results = {}

# 1. AUC of customer_id alone
# Check if customer_id has numeric/hex order that predicts churn
cid_num = train_raw['customer_id'].str.replace('C', '', regex=False)
try:
    cid_int = cid_num.apply(lambda x: int(x, 16))
    auc_cid = roc_auc_score(train_raw['churn'], cid_int)
except Exception:
    auc_cid = 0.5

print(f"1. AUC of customer_id hex integer alone: {auc_cid:.5f}")
audit_results['customer_id_auc'] = float(auc_cid)

# 2. customer_id correlation with dates, region, channel, churn
cid_corr = {}
cid_corr['corr_with_churn'] = float(train_raw['churn'].corr(cid_int))
cid_corr['corr_with_signup_date'] = float(train_raw['signup_date'].astype(int).corr(cid_int))
cid_corr['corr_with_snapshot_date'] = float(train_raw['snapshot_date'].astype(int).corr(cid_int))
print(f"2. customer_id correlations: churn={cid_corr['corr_with_churn']:.4f}, signup={cid_corr['corr_with_signup_date']:.4f}, snapshot={cid_corr['corr_with_snapshot_date']:.4f}")
audit_results['customer_id_correlations'] = cid_corr

# 3. events.csv row-order patterns
# Check if events are sorted strictly by customer_id or timestamp
is_ts_sorted = events_raw['timestamp'].is_monotonic_increasing
print(f"3. events.csv timestamp is monotonic increasing: {is_ts_sorted}")
audit_results['events_timestamp_sorted'] = bool(is_ts_sorted)

# 4 & 5. Missingness patterns across profile columns and churn prediction
print("\n4 & 5. Missingness across profile columns vs Churn:")
missing_dict = {}
for col in ['age_band', 'payment_method', 'referral_code', 'monthly_price', 'num_devices', 'family_sharing']:
    n_miss = train_raw[col].isnull().sum()
    pct_miss = n_miss / len(train_raw) * 100.0
    if n_miss > 0:
        churn_when_miss = train_raw[train_raw[col].isnull()]['churn'].mean()
        churn_when_pres = train_raw[train_raw[col].notnull()]['churn'].mean()
        diff = churn_when_miss - churn_when_pres
        print(f"  {col:20s}: {n_miss:5d} nulls ({pct_miss:4.1f}%) | Churn Null: {churn_when_miss:.4f} vs Pres: {churn_when_pres:.4f} (Diff: {diff:+.4f})")
        missing_dict[col] = {'null_count': int(n_miss), 'pct_null': pct_miss, 'churn_when_null': churn_when_miss, 'churn_when_present': churn_when_pres, 'diff': diff}
    else:
        print(f"  {col:20s}: 0 nulls")
        missing_dict[col] = {'null_count': 0, 'pct_null': 0.0}

audit_results['missingness_audit'] = missing_dict

# 6. monthly_price relative to plan x billing expectations
print("\n6. monthly_price distribution by plan x billing:")
price_summary = train_raw.groupby(['plan', 'billing_cycle'])['monthly_price'].agg(['count', 'min', 'mean', 'max', 'std']).reset_index()
print(price_summary.to_string())
audit_results['price_by_plan_billing'] = price_summary.to_dict(orient='records')

# 7. joint structure of snapshot_date - signup_date
account_age_days = (train_raw['snapshot_date'] - train_raw['signup_date']).dt.days
print(f"\n7. Account age days: min={account_age_days.min()}, median={account_age_days.median()}, max={account_age_days.max()}")
negative_ages = (account_age_days < 0).sum()
print(f"  Negative account ages: {negative_ages}")
audit_results['negative_account_ages'] = int(negative_ages)

# 8. referral_code string structure
print("\n8. referral_code string structure:")
refs = train_raw['referral_code'].dropna().unique()
ref_lengths = set(len(r) for r in refs)
ref_prefixes = set(r[:2] for r in refs)
print(f"  Total unique codes: {len(refs)}")
print(f"  Length of codes: {ref_lengths}")
print(f"  First 2 characters set: {ref_prefixes}")
audit_results['referral_code_structure'] = {'unique_codes': len(refs), 'lengths': list(ref_lengths), 'prefixes': list(ref_prefixes)}

# ===========================================================================
# PHASE G: LIFECYCLE / WINDOW-CENSORING AUDIT
# ===========================================================================
print("\n" + "=" * 80)
print("PHASE G: LIFECYCLE / WINDOW-CENSORING AUDIT")
print("=" * 80)

# Check effective observation window:
# Find global earliest event in events.csv
global_min_event = events_raw['timestamp'].min()
global_max_event = events_raw['timestamp'].max()
print(f"Global event timestamp range: {global_min_event} to {global_max_event}")

# For each snapshot month, calculate days of available event history
print("\nEffective Event Lookback Window by Snapshot Month:")
lookback_audit = {}
for dt in sorted(train_raw['snapshot_date'].unique()):
    dt_ts = pd.to_datetime(dt)
    days_back = (dt_ts - global_min_event).total_seconds() / 86400.0
    # Average total events per customer for this snapshot
    sub_custs = train_raw[train_raw['snapshot_date'] == dt]['customer_id']
    sub_events = events_raw[(events_raw['customer_id'].isin(sub_custs)) & (events_raw['timestamp'] <= dt_ts)]
    mean_evts = len(sub_events) / len(sub_custs)
    print(f"  Snapshot {str(dt)[:10]}: Max observable history = {days_back:.1f} days | Mean events/cust = {mean_evts:.1f}")
    lookback_audit[str(dt)[:10]] = {'max_days_history': days_back, 'mean_events_per_cust': mean_evts}

for dt in sorted(test_raw['snapshot_date'].unique()):
    dt_ts = pd.to_datetime(dt)
    days_back = (dt_ts - global_min_event).total_seconds() / 86400.0
    sub_custs = test_raw[test_raw['snapshot_date'] == dt]['customer_id']
    sub_events = events_raw[(events_raw['customer_id'].isin(sub_custs)) & (events_raw['timestamp'] <= dt_ts)]
    mean_evts = len(sub_events) / len(sub_custs)
    print(f"  Test Snapshot {str(dt)[:10]}: Max observable history = {days_back:.1f} days | Mean events/cust = {mean_evts:.1f}")
    lookback_audit[str(dt)[:10]] = {'max_days_history': days_back, 'mean_events_per_cust': mean_evts}

# Check fixed 120-day window:
# Since global_min_event is ~2025-10-03, Jan 31 2026 is exactly 120 days from Oct 3 2025!
# That means 14d, 30d, 60d, 90d, 120d windows are FULLY covered for ALL snapshot months (Jan, Feb, Mar, Apr, May, Jun)!
audit_results['lookback_audit'] = lookback_audit

# ===========================================================================
# PHASE H: RENEWAL / BILLING PHASE EDA
# ===========================================================================
print("\n" + "=" * 80)
print("PHASE H: RENEWAL / BILLING PHASE INVESTIGATION")
print("=" * 80)

# Calculate Renewal Timing variables for train_raw
# 1. Annual days to anniversary (modulo 365.25)
train_raw['tenure_days'] = (train_raw['snapshot_date'] - train_raw['signup_date']).dt.days
train_raw['annual_days_since_anniversary'] = train_raw['tenure_days'] % 365
train_raw['annual_days_to_anniversary'] = 365 - train_raw['annual_days_since_anniversary']

# 2. Monthly days to renewal (modulo 30.44 or 30 days)
train_raw['monthly_days_since_renewal'] = train_raw['tenure_days'] % 30
train_raw['monthly_days_to_renewal'] = 30 - train_raw['monthly_days_since_renewal']

# Tabulate Churn Rate by Annual Days to Renewal in Annual Customers
ann_sub = train_raw[train_raw['billing_cycle'] == 'annual'].copy()
ann_sub['renewal_bucket'] = pd.cut(ann_sub['annual_days_to_anniversary'], bins=[0, 15, 30, 60, 90, 180, 270, 366], labels=['0-15d', '16-30d', '31-60d', '61-90d', '91-180d', '181-270d', '270d+'])

print("\n--- Annual Billing: Churn Rate by Days to Anniversary ---")
ann_churn_by_phase = ann_sub.groupby('renewal_bucket', observed=False).agg(
    count=('churn', 'count'),
    mean_churn=('churn', 'mean')
).reset_index()
print(ann_churn_by_phase.to_string())

# Tabulate Monthly Billing: Churn Rate by Monthly Cycle Phase
mon_sub = train_raw[train_raw['billing_cycle'] == 'monthly'].copy()
mon_sub['cycle_bucket'] = pd.cut(mon_sub['monthly_days_to_renewal'], bins=[0, 3, 7, 14, 21, 31], labels=['0-3d', '4-7d', '8-14d', '15-21d', '22-30d'])

print("\n--- Monthly Billing: Churn Rate by Days to Monthly Renewal ---")
mon_churn_by_phase = mon_sub.groupby('cycle_bucket', observed=False).agg(
    count=('churn', 'count'),
    mean_churn=('churn', 'mean')
).reset_index()
print(mon_churn_by_phase.to_string())

# Check statistical significance via Chi-square test
contingency_ann = pd.crosstab(ann_sub['renewal_bucket'], ann_sub['churn'])
chi2_ann, p_ann, _, _ = stats.chi2_contingency(contingency_ann)
print(f"\nChi-square test for Annual Renewal Phase: Chi2 = {chi2_ann:.2f}, p-value = {p_ann:.4e}")

contingency_mon = pd.crosstab(mon_sub['cycle_bucket'], mon_sub['churn'])
chi2_mon, p_mon, _, _ = stats.chi2_contingency(contingency_mon)
print(f"Chi-square test for Monthly Cycle Phase:  Chi2 = {chi2_mon:.2f}, p-value = {p_mon:.4e}")

audit_results['annual_renewal_phase'] = ann_churn_by_phase.to_dict(orient='records')
audit_results['monthly_cycle_phase'] = mon_churn_by_phase.to_dict(orient='records')
audit_results['chi2_annual'] = {'chi2': chi2_ann, 'p_value': p_ann}
audit_results['chi2_monthly'] = {'chi2': chi2_mon, 'p_value': p_mon}

# Save Phase F, G, H findings
os.makedirs('v13', exist_ok=True)
with open('v13/v13_phases_f_g_h.json', 'w') as f:
    json.dump(audit_results, f, indent=2, default=str)

print("\nPhases F, G, H investigation complete! Saved to v13/v13_phases_f_g_h.json")
