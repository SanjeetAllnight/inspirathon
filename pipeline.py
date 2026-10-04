import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import os

start_time = time.time()

# 1. Load data
print("Loading CSV files...")
train = pd.read_csv('/home/sanjeet/kaggle/inspirathon/train.csv')
test = pd.read_csv('/home/sanjeet/kaggle/inspirathon/test.csv')
events = pd.read_csv('/home/sanjeet/kaggle/inspirathon/events.csv')

# 2. Parse dates
print("Parsing dates...")
train['snapshot_date'] = pd.to_datetime(train['snapshot_date'])
train['signup_date'] = pd.to_datetime(train['signup_date'])
test['snapshot_date'] = pd.to_datetime(test['snapshot_date'])
test['signup_date'] = pd.to_datetime(test['signup_date'])
events['timestamp'] = pd.to_datetime(events['timestamp'])

# Combine train/test snapshots for event joining
snapshots = pd.concat([
    train[['customer_id', 'snapshot_date']],
    test[['customer_id', 'snapshot_date']]
]).drop_duplicates(subset=['customer_id'])

# Join snapshot_date to events
events = events.merge(snapshots, on='customer_id', how='inner')

# Filter out future events
events['days_before_snapshot'] = (events['snapshot_date'] - events['timestamp']).dt.total_seconds() / 86400.0
events = events[events['days_before_snapshot'] >= 0].copy()

# Add columns for event types to make aggregation easier
events['is_session'] = (events['event_type'] == 'session').astype(int)
events['is_support'] = (events['event_type'] == 'support_ticket').astype(int)
events['is_payment_failed'] = (events['event_type'] == 'payment_failed').astype(int)
events['is_upgrade'] = ((events['event_type'] == 'plan_change') & (events['value'] > 0)).astype(int)
events['is_downgrade'] = ((events['event_type'] == 'plan_change') & (events['value'] < 0)).astype(int)

events['session_duration'] = np.where(events['event_type'] == 'session', events['value'], np.nan)

# Create boolean masks for time windows
windows = [7, 14, 30, 60, 120]
for w in windows:
    events[f'in_{w}d'] = (events['days_before_snapshot'] <= w).astype(int)

def build_customer_features(events_df):
    grouped = events_df.groupby('customer_id')
    
    feats = grouped.agg(
        total_event_count=('event_type', 'count'),
        session_count=('is_session', 'sum'),
        support_ticket_count=('is_support', 'sum'),
        payment_failed_count=('is_payment_failed', 'sum'),
        plan_change_count=('event_type', lambda x: (x == 'plan_change').sum()),
        upgrade_count=('is_upgrade', 'sum'),
        downgrade_count=('is_downgrade', 'sum'),
        days_since_last_event=('days_before_snapshot', 'min'),
        
        session_duration_total=('session_duration', 'sum'),
        session_duration_mean=('session_duration', 'mean'),
        session_duration_median=('session_duration', 'median'),
        session_duration_max=('session_duration', 'max'),
    )
    
    # Days since last session
    session_events = events_df[events_df['is_session'] == 1]
    last_session = session_events.groupby('customer_id')['days_before_snapshot'].min().rename('days_since_last_session')
    feats = feats.join(last_session, how='left')
    
    # Time window features
    for w in windows:
        w_mask = events_df[f'in_{w}d'] == 1
        w_events = events_df[w_mask]
        w_grouped = w_events.groupby('customer_id')
        
        w_feats = w_grouped.agg(**{
            f'events_in_{w}d': ('event_type', 'count'),
            f'session_count_in_{w}d': ('is_session', 'sum'),
            f'session_duration_in_{w}d': ('session_duration', 'sum'),
            f'payment_failed_in_{w}d': ('is_payment_failed', 'sum'),
            f'support_ticket_in_{w}d': ('is_support', 'sum'),
            f'upgrade_in_{w}d': ('is_upgrade', 'sum'),
            f'downgrade_in_{w}d': ('is_downgrade', 'sum'),
        })
        feats = feats.join(w_feats, how='left')
        
    return feats

print("Building event features (this may take a moment)...")
event_feats = build_customer_features(events)

def enrich_profile(df, event_feats):
    df = df.copy()
    
    # Join event features
    df = df.merge(event_feats, on='customer_id', how='left')
    
    # Profile features
    df['account_age_days'] = (df['snapshot_date'] - df['signup_date']).dt.days
    df['price_per_device'] = df['monthly_price'] / df['num_devices'].clip(lower=1)
    
    # Missing value indicators
    df['is_age_missing'] = df['age_band'].isnull().astype(int)
    df['is_payment_method_missing'] = df['payment_method'].isnull().astype(int)
    df['has_referral_code'] = df['referral_code'].notnull().astype(int)
    df['has_events'] = df['total_event_count'].notnull().astype(int)
    
    # Impute missing days since last event/session with account_age_days
    days_cols = ['days_since_last_event', 'days_since_last_session']
    for c in days_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df['account_age_days'])
            
    # Impute other missing event features with 0
    event_cols = event_feats.columns
    event_cols_to_fill = [c for c in event_cols if c not in days_cols]
    df[event_cols_to_fill] = df[event_cols_to_fill].fillna(0)
    
    # Activity ratios
    df['ratio_events_7d_30d'] = df['events_in_7d'] / (df['events_in_30d'] + 1e-5)
    df['ratio_events_30d_60d'] = df['events_in_30d'] / (df['events_in_60d'] + 1e-5)
    df['ratio_sessions_7d_30d'] = df['session_count_in_7d'] / (df['session_count_in_30d'] + 1e-5)
    df['ratio_sessions_30d_60d'] = df['session_count_in_30d'] / (df['session_count_in_60d'] + 1e-5)
    
    # Convert categoricals to dummy variables or use native categorical types
    cat_cols = ['plan', 'billing_cycle', 'region', 'age_band', 'acquisition_channel', 'payment_method']
    for c in cat_cols:
        # Fill missing values in categoricals with a string so CatBoost is happy
        df[c] = df[c].fillna('Missing').astype(str)
        df[c] = df[c].astype('category')
        
    return df

print("Enriching profiles...")
train_df = enrich_profile(train, event_feats)
test_df = enrich_profile(test, event_feats)

print("Train shape after merge:", train_df.shape)
print("Test shape after merge:", test_df.shape)

cols_to_drop = ['customer_id', 'snapshot_date', 'signup_date', 'referral_code']
if 'churn' in cols_to_drop: cols_to_drop.remove('churn')
features = [c for c in train_df.columns if c not in cols_to_drop and c != 'churn']

# Save matrices
train_df.to_parquet('/home/sanjeet/kaggle/inspirathon/train_features.parquet')
test_df.to_parquet('/home/sanjeet/kaggle/inspirathon/test_features.parquet')
print("Saved train_features.parquet and test_features.parquet")

# Temporal Validation Split
max_train_date = train_df['snapshot_date'].max()
val_mask = train_df['snapshot_date'] == max_train_date
train_split = train_df[~val_mask]
val_split = train_df[val_mask]

X_train = train_split[features]
y_train = train_split['churn']
X_val = val_split[features]
y_val = val_split['churn']

print(f"Train split: {X_train.shape[0]} rows (Dates: {train_split['snapshot_date'].min().date()} to {train_split['snapshot_date'].max().date()})")
print(f"Val split: {X_val.shape[0]} rows (Date: {val_split['snapshot_date'].min().date()})")
print(f"Number of features: {len(features)}")

# Model Training
model_type = "Unknown"
cat_features = [features.index(c) for c in X_train.select_dtypes(include='category').columns]

try:
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=1000, 
                               learning_rate=0.05, 
                               eval_metric='AUC', 
                               random_seed=42, 
                               verbose=100, 
                               cat_features=cat_features)
    model_type = "CatBoost"
except ImportError:
    try:
        from lightgbm import LGBMClassifier
        # convert cat columns to int or use lightgbm category type properly
        model = LGBMClassifier(n_estimators=1000, 
                               learning_rate=0.05, 
                               random_state=42, 
                               n_jobs=-1)
        model_type = "LightGBM"
    except ImportError:
        pass

if model_type == "CatBoost":
    print(f"Training {model_type}...")
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
elif model_type == "LightGBM":
    print(f"Training {model_type}...")
    # LGBM can handle 'category' dtype directly
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[])
else:
    print("Neither CatBoost nor LightGBM is available. Please install one of them.")
    import sys; sys.exit(1)

# Evaluation
val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print(f"\n--- Validation ROC-AUC: {auc:.5f} ---")

# Feature Importance
if model_type == "CatBoost":
    importances = model.get_feature_importance()
else:
    importances = model.feature_importances_

importance_df = pd.DataFrame({'feature': features, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
print("\nTop 25 Features:")
print(importance_df.head(25).to_string(index=False))

end_time = time.time()
print(f"\nTotal Pipeline Time: {end_time - start_time:.2f} seconds")
