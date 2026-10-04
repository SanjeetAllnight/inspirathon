import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

print("Testing leak-safe target encoder...")

def compute_oof_target_encoding(train_s, y_train, val_s, smoothing=10.0, n_splits=5, seed=42):
    """
    Computes strictly leak-safe smoothed target encoding:
    - 5-fold OOF encoding for train
    - smoothed mapping for val
    """
    global_mean = y_train.mean()
    
    # 1. Train OOF encoding
    oof_te = np.full(len(train_s), np.nan, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    train_df = pd.DataFrame({'cat': train_s, 'y': y_train})
    
    for tr_idx, val_idx in kf.split(train_df):
        tr_part = train_df.iloc[tr_idx]
        val_part = train_df.iloc[val_idx]
        
        inner_prior = tr_part['y'].mean()
        stats = tr_part.groupby('cat')['y'].agg(['count', 'mean'])
        
        smoothed = (stats['count'] * stats['mean'] + smoothing * inner_prior) / (stats['count'] + smoothing)
        
        encoded_val = val_part['cat'].map(smoothed).fillna(inner_prior).values
        oof_te[val_idx] = encoded_val
        
    # 2. Val encoding from full train
    full_stats = train_df.groupby('cat')['y'].agg(['count', 'mean'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
    
    val_te = val_s.map(full_smoothed).fillna(global_mean).values.astype(np.float32)
    
    return oof_te, val_te

# Quick sanity check
cats_tr = pd.Series(['A', 'A', 'B', 'B', 'B', 'C', 'C', 'A'])
y_tr = pd.Series([1, 1, 0, 0, 0, 1, 0, 1])
cats_vl = pd.Series(['A', 'B', 'C', 'D'])

oof_tr, val_encoded = compute_oof_target_encoding(cats_tr, y_tr, cats_vl, smoothing=2.0, n_splits=3)
print("OOF Train TE:", oof_tr)
print("Val TE      :", val_encoded)
print("Sanity check passed!")
