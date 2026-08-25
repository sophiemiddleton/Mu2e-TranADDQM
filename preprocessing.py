"""
Phase 2: Data Preprocessing

Responsibilities:
  - Identify active (non-flat) features
  - Apply outlier detection (IQR method)
  - Standardize feature scaling
  - Create sliding window tensors for time-series models
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler


def identify_active_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Filter to dynamic, non-zero features (exclude flat or constant columns).
    
    A feature is considered "active" if:
      - Has non-zero values (not all 0.0)
      - Not all 1.0
      - Has actual variance (nunique > 1)
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        List of active feature column names
    """
    active_features = []
    
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        
        # Check criteria
        has_nonzero = (vals != 0.0).any()
        not_all_ones = not (vals == 1.0).all()
        has_variance = vals.nunique() > 1
        
        if not vals.empty and has_nonzero and not_all_ones and has_variance:
            active_features.append(col)
    
    print(f"\nFeature Selection:")
    print(f"  Total features:       {len(feature_cols)}")
    print(f"  Active features:      {len(active_features)}")
    print(f"  Removed (flat):       {len(feature_cols) - len(active_features)}")
    
    return sorted(active_features)


def apply_iqr_outlier_filtering(df: pd.DataFrame, 
                               feature_cols: List[str],
                               iqr_multiplier: float = 1.5) -> Tuple[pd.DataFrame, dict]:
    """
    Remove outlier records using Interquartile Range (IQR) method.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns to check
        iqr_multiplier: Multiplier for IQR bounds (1.5 = aggressive, 3.0 = conservative)
        
    Returns:
        Tuple of (cleaned_df, filter_stats)
    """
    df_clean = df.copy()
    outlier_mask = pd.Series(False, index=df_clean.index)
    
    outliers_per_feature = {}
    
    for col in feature_cols:
        vals = pd.to_numeric(df_clean[col], errors='coerce')
        
        Q1 = vals.quantile(0.25)
        Q3 = vals.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            
            col_outliers = (vals < lower) | (vals > upper)
            outlier_mask |= col_outliers
            outliers_per_feature[col] = col_outliers.sum()
    
    df_clean = df_clean[~outlier_mask].reset_index(drop=True)
    
    stats = {
        'input_records': len(df),
        'output_records': len(df_clean),
        'removed_records': outlier_mask.sum(),
        'removal_percentage': (outlier_mask.sum() / len(df) * 100) if len(df) > 0 else 0,
        'outliers_per_feature': outliers_per_feature,
    }
    
    print(f"\nIQR Outlier Filtering (multiplier={iqr_multiplier}x):")
    print(f"  Input records:        {stats['input_records']}")
    print(f"  Output records:       {stats['output_records']}")
    print(f"  Removed:              {stats['removed_records']} ({stats['removal_percentage']:.1f}%)")
    
    return df_clean, stats


def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to mean=0, std=1.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        
    Returns:
        Tuple of (scaled_matrix, scaler_object)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeature Scaling:")
    print(f"  Input shape:          {X.shape}")
    print(f"  Output shape:         {X_scaled.shape}")
    print(f"  Mean (after):         {X_scaled.mean(axis=0)[:3]}... (first 3)")
    print(f"  Std (after):          {X_scaled.std(axis=0)[:3]}... (first 3)")
    
    return X_scaled, scaler


def create_sliding_windows(X: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Create sliding window tensors for time-series models (e.g., TranAD).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        window_size: Length of each window
        stride: Step between windows
        
    Returns:
        Tensor of shape (n_windows, window_size, n_features)
    """
    if X.shape[0] < window_size:
        raise ValueError(f"Data has {X.shape[0]} samples, need at least {window_size} for window_size={window_size}")
    
    windows = []
    for i in range(0, X.shape[0] - window_size + 1, stride):
        windows.append(X[i:i + window_size])
    
    tensor = np.array(windows)
    print(f"\nSliding Window Tensors:")
    print(f"  Window size:          {window_size}")
    print(f"  Stride:               {stride}")
    print(f"  Input shape:          {X.shape}")
    print(f"  Output tensor shape:  {tensor.shape}")
    
    return tensor


def preprocess_pipeline(df: pd.DataFrame, 
                       feature_cols: List[str],
                       iqr_multiplier: float = 1.5) -> Tuple[pd.DataFrame, List[str], np.ndarray, StandardScaler]:
    """
    Complete preprocessing pipeline: feature selection -> outlier removal -> scaling.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        iqr_multiplier: IQR multiplier for outlier filtering
        
    Returns:
        Tuple of (cleaned_df, active_features, scaled_matrix, scaler)
    """
    # Step 1: Feature selection
    active_features = identify_active_features(df, feature_cols)
    
    # Step 2: Outlier filtering
    df_clean, filter_stats = apply_iqr_outlier_filtering(df, active_features, iqr_multiplier)
    
    # Step 3: Extract and scale features
    X_raw = df_clean[active_features].values
    X_scaled, scaler = scale_features(X_raw)
    
    print(f"\n✓ Preprocessing complete")
    print(f"  Input:  {len(df)} records × {len(feature_cols)} features")
    print(f"  Output: {len(df_clean)} records × {len(active_features)} features (scaled)")
    
    return df_clean, active_features, X_scaled, scaler
