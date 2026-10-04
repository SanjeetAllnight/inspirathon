import pandas as pd
import numpy as np

print("--- 1. Shape of each file ---")
train = pd.read_csv('/home/sanjeet/kaggle/inspirathon/train.csv')
test = pd.read_csv('/home/sanjeet/kaggle/inspirathon/test.csv')
events = pd.read_csv('/home/sanjeet/kaggle/inspirathon/events.csv')
sub = pd.read_csv('/home/sanjeet/kaggle/inspirathon/sample_submission.csv')

print(f"train.csv shape: {train.shape}")
print(f"test.csv shape: {test.shape}")
print(f"events.csv shape: {events.shape}")
print(f"sample_submission.csv shape: {sub.shape}\n")

print("--- 2. Every column, dtype, missing-value count and cardinality ---")
def print_df_info(name, df):
    print(f"*** {name} ***")
    info_df = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_%': (df.isnull().sum() / len(df) * 100).round(2),
        'cardinality': df.nunique()
    })
    print(info_df)
    print()

print_df_info("train.csv", train)
print_df_info("test.csv", test)
print_df_info("events.csv", events)

print("--- 3. Target distribution in train.csv ---")
print(train['churn'].value_counts(normalize=True))
print(train['churn'].value_counts())
print()

print("--- 4. Date ranges in train.csv/test.csv ---")
for col in ['snapshot_date', 'signup_date']:
    train[col] = pd.to_datetime(train[col])
    test[col] = pd.to_datetime(test[col])
    print(f"train {col}: min {train[col].min()}, max {train[col].max()}")
    print(f"test {col}: min {test[col].min()}, max {test[col].max()}")
print()

print("--- 5. Events distribution by event_type ---")
print(events['event_type'].value_counts(normalize=True))
print(events['event_type'].value_counts())
print()

print("--- 6. Number of unique customers in train/test/events ---")
train_cust = set(train['customer_id'])
test_cust = set(test['customer_id'])
event_cust = set(events['customer_id'])
print(f"train: {len(train_cust)}")
print(f"test: {len(test_cust)}")
print(f"events: {len(event_cust)}")
print()

print("--- 7. Whether every train/test customer has events ---")
print(f"Train customers without events: {len(train_cust - event_cust)}")
print(f"Test customers without events: {len(test_cust - event_cust)}")
print()

print("--- 8. Event frequency per customer: min/median/mean/max ---")
event_counts = events.groupby('customer_id').size()
print(f"min: {event_counts.min()}")
print(f"median: {event_counts.median()}")
print(f"mean: {event_counts.mean():.2f}")
print(f"max: {event_counts.max()}")
print()

print("--- 9. Duplicate rows and suspicious values ---")
print(f"train duplicate rows: {train.duplicated().sum()}")
print(f"test duplicate rows: {test.duplicated().sum()}")
print(f"events duplicate rows: {events.duplicated().sum()}")

# Checking suspicious values in train/test numericals
print("\nTrain numerical descriptions:")
print(train.describe())

print("--- 10. Any train/test leakage risks ---")
overlap_cust = train_cust.intersection(test_cust)
print(f"Customers in both train and test: {len(overlap_cust)}")

if len(overlap_cust) > 0:
    print(f"Overlap example: {list(overlap_cust)[:5]}")

print("\n--- 11. Important patterns ---")
# Let's check event values
print("Event value descriptions by event_type:")
for etype in events['event_type'].unique():
    print(f"\n{etype}:")
    mask = events['event_type'] == etype
    print(pd.to_numeric(events.loc[mask, 'value'], errors='coerce').describe())
