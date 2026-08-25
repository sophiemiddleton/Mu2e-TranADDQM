"""
Phase 1: Data Ingestion & Validation

Responsibilities:
  - Load filtered DQM CSV dataset
  - Parse and validate timestamps (handle timezone info)
  - Perform basic data quality checks
  - Prepare metadata and feature columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Set


def load_filtered_csv(csv_path: str) -> pd.DataFrame:
    """
    Load filtered DQM anomaly dataset from CSV.
    
    Args:
        csv_path: Path to dqm_crv_anomaly_dataset_*_filtered.csv
        
    Returns:
        DataFrame with columns: run, subrun, timestamp, CRV_* metrics
        
    Raises:
        FileNotFoundError: If CSV does not exist
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"Loading: {csv_file.name}")
    df = pd.read_csv(csv_file)
    
    print(f"✓ Loaded {df.shape[0]} records, {df.shape[1]} columns")
    return df


def validate_and_parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamp column and handle timezone-aware ISO 8601 format.
    
    Args:
        df: Input DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with added 'datetime' column (timezone-naive UTC)
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    # Parse with UTC handling
    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    
    # Convert to tz-naive for clean plotting
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    valid_count = df['datetime'].notna().sum()
    invalid_count = df['datetime'].isna().sum()
    
    print(f"Timestamp Parsing:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    
    if invalid_count > 0:
        print(f"  ⚠️  {invalid_count} records with unparseable timestamps")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract feature columns (exclude metadata).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature column names
    """
    meta_cols = {'run', 'subrun', 'timestamp', 'datetime', 'date_only'}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return sorted(feature_cols)


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    feature_cols = get_feature_columns(df)
    
    quality = {
        'total_records': len(df),
        'total_features': len(feature_cols),
        'valid_timestamps': df['datetime'].notna().sum(),
        'missing_timestamps': df['datetime'].isna().sum(),
        'unique_runs': df['run'].nunique(),
        'unique_subruns': df['subrun'].nunique(),
    }
    
    # Check for completely empty columns
    empty_cols = [c for c in feature_cols if df[c].isnull().all()]
    quality['empty_feature_columns'] = empty_cols
    
    # Check for completely flat columns (all same value)
    flat_cols = []
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) > 0 and vals.nunique() == 1:
            flat_cols.append(col)
    quality['flat_feature_columns'] = flat_cols
    
    return quality


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print human-readable data summary.
    
    Args:
        df: Input DataFrame
    """
    quality = validate_data_quality(df)
    
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    print(f"Total Records:        {quality['total_records']}")
    print(f"Total Features:       {quality['total_features']}")
    print(f"Valid Timestamps:     {quality['valid_timestamps']}")
    print(f"Missing Timestamps:   {quality['missing_timestamps']}")
    print(f"Unique Runs:          {quality['unique_runs']}")
    print(f"Unique Subruns:       {quality['unique_subruns']}")
    
    if quality['empty_feature_columns']:
        print(f"⚠️  Empty Columns:     {len(quality['empty_feature_columns'])}")
    else:
        print(f"✓ No completely empty columns")
    
    if quality['flat_feature_columns']:
        print(f"⚠️  Flat Columns:      {len(quality['flat_feature_columns'])}")
    else:
        print(f"✓ No completely flat columns")
    print("=" * 70 + "\n")


def ingest_data(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete ingestion pipeline: load, validate, parse.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (validated_df, feature_columns)
    """
    df = load_filtered_csv(csv_path)
    df = validate_and_parse_timestamps(df)
    feature_cols = get_feature_columns(df)
    
    print_data_summary(df)
    
    return df, feature_cols
