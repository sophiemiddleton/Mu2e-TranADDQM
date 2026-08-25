# DQM Anomaly Detection Pipeline

A modular, production-ready pipeline for detecting anomalies in CRV (Cosmic Ray Veto) detector data from the Mu2e experiment.

## Overview

The pipeline processes filtered DQM (Data Quality Monitoring) datasets through five phases:

### Phase 1: Ingestion
- Load pre-filtered CSV dataset (`dqm_crv_anomaly_dataset_*_filtered.csv`)
- Parse and validate ISO 8601 timestamps (handles timezone info)
- Perform data quality checks (missing values, data types, etc.)

### Phase 2: Preprocessing
- Identify active (non-flat) features
- Apply IQR-based outlier filtering
- Standardize feature scaling (mean=0, std=1)
- Ready data for anomaly detection models

### Phase 3: Models (Ensemble Anomaly Detection)
- **Z-Score Detector** - Statistical anomaly detection (simple, interpretable)
- **PCA Detector** - Multivariate outlier detection via reconstruction error
- **TranAD Detector** - Transformer-based deep learning model (requires PyTorch)
- **Ensemble Voting** - Combine detectors for robust predictions (majority vote)

### Phase 4: Export
- Generate detailed anomaly reports (CSV)
- Include individual detector scores and voting results
- Separate anomalies-only report for quick review

### Phase 5: Visualization
- Time series analysis (4-panel overview)
- Detector score comparisons (Z-Score, PCA, TranAD)
- Anomaly score distributions
- Feature correlations (heatmap)
- Anomaly timeline highlighting
- Feature distributions (top 6)

## Installation

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn (required)
- matplotlib, seaborn (for visualization)
- torch (optional, for TranAD)

### Quick Setup

```bash
cd /exp/mu2e/app/users/sophie/CrossExperimentalAIDQM
pip install -e .
```

Or install with TranAD support:
```bash
pip install -e ".[tranad]"
```

Or install dependencies manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch  # Optional, for TranAD
```

## Usage

### Command Line (Simplest)

Run with default parameters and visualization:
```bash
python -m dqm_pipeline --csv dqm_crv_anomaly_dataset_004-000_filtered.csv
```

With custom parameters:
```bash
python -m dqm_pipeline \
  --csv data.csv \
  --output ./results \
  --iqr 1.5 \
  --voting-threshold 2.0 \
  --plots
```

Skip visualization (faster):
```bash
python -m dqm_pipeline --csv data.csv --no-plots
```

### Quick Start Script

```bash
python example_usage.py
```

### Python API

```python
from dqm_pipeline.run_pipeline import main

# Run full pipeline with plots
main(
    csv_path="data.csv",
    output_dir="./results",
    iqr_multiplier=1.5,
    voting_threshold=2.0,
    generate_plots=True
)

# Or run individual phases
from dqm_pipeline import ingestion, preprocessing, models

df, features = ingestion.ingest_data("data.csv")
df_clean, active_features, X_scaled, scaler = preprocessing.preprocess_pipeline(
    df, features, iqr_multiplier=1.5
)
predictions, results = models.run_ensemble_detection(X_scaled, df_clean, active_features)
```

## Configuration

### IQR Multiplier (Outlier Sensitivity)
- **1.5** (aggressive) - Removes ~7% of data, catches subtle anomalies
- **3.0** (conservative) - Removes ~0.3% of data, catches only extreme outliers
- Default: **1.5**

### Voting Threshold (Ensemble Confidence)
- **1.0** - Any single detector can flag (low confidence)
- **2.0** - At least 2 of 3 detectors must agree (medium confidence)
- **3.0** - All 3 detectors must agree (high confidence, few anomalies)
- Default: **2.0**

## Output Files

The pipeline generates:

```
results/
├── anomaly_report_YYYYMMDD_HHMMSS.csv           # Full results with all scores
├── anomaly_report_YYYYMMDD_HHMMSS_anomalies_only.csv  # Anomalies only
└── plots/
    ├── timeseries_overview.png                 # 4-panel time series
    ├── anomaly_scores.png                      # TranAD scores over time
    ├── detector_comparison.png                 # All 3 detectors side-by-side
    ├── anomaly_distribution.png                # Pie/bar charts of anomalies
    ├── feature_distributions.png               # Distribution of top 6 features
    ├── correlation_heatmap.png                 # Feature correlations
    └── anomaly_timeline.png                    # Anomalies highlighted on timeline
```

### Report Columns
- `run, subrun, timestamp, datetime` - Record identifiers
- `anomaly` - Binary ensemble prediction (1 = anomaly)
- `z_score_anomaly, pca_anomaly, tranad_anomaly` - Individual detector predictions
- `voting_score` - Sum of detector votes (0-3)
- `z_score_max` - Max Z-score (anomaly severity)
- `pca_reconstruction_error` - PCA reconstruction error
- `tranad_reconstruction_error` - TranAD reconstruction error
- All original feature columns

## Dataset Format

Expected input: Pre-filtered CSV with structure:
```
run, subrun, timestamp, CRV_crvPEsMPV_CRVsector0_LeftEdge, CRV_crvPEsMPV_CRVsector0_RightEdge, ...
118, 1, 2026-02-13 11:07:29-06:00, 1.0, 1.0, ...
118, 2, 2026-02-13 11:07:30-06:00, 1.0, 1.0, ...
```

## Project Structure

```
dqm_pipeline/
├── __init__.py              # Package initialization
├── __main__.py              # CLI entry point
├── ingestion.py             # Phase 1: Data loading & validation
├── preprocessing.py         # Phase 2: Feature scaling & selection
├── models.py                # Phase 3: Anomaly detectors (Z-Score, PCA, TranAD)
├── tranad_model.py          # TranAD transformer implementation
├── visualization.py         # Phase 5: Plot generation
├── run_pipeline.py          # Master orchestrator & CLI
└── README.md                # This file
```

## Performance Notes

- **Dataset**: 1114 records × 37 features (wide-format)
- **Processing Time**: ~5-10 seconds (with TranAD), ~1-2 seconds (without)
- **Memory**: ~150-200 MB with TranAD, ~100 MB without
- **GPU Support**: TranAD can use CUDA if available (automatic detection)

## Examples

### Example 1: Basic Pipeline
```bash
python -m dqm_pipeline --csv Genesis/dqm_crv_anomaly_dataset_004-000_filtered.csv
```

### Example 2: Conservative Anomaly Detection
```bash
python -m dqm_pipeline \
  --csv data.csv \
  --iqr 3.0 \
  --voting-threshold 3.0
```

### Example 3: Aggressive Anomaly Detection with Lower Voting
```bash
python -m dqm_pipeline \
  --csv data.csv \
  --iqr 1.5 \
  --voting-threshold 1.0
```

### Example 4: Skip Visualization (Faster)
```bash
python -m dqm_pipeline --csv data.csv --no-plots
```

## Troubleshooting

### ImportError: No module named 'torch'
TranAD requires PyTorch. Install with:
```bash
pip install torch
# Or as part of setup:
pip install -e ".[tranad]"
```

### ImportError: No module named 'matplotlib'
Visualization requires matplotlib and seaborn. Install with:
```bash
pip install matplotlib seaborn
```

### Memory Error with Large Datasets
Reduce batch size or skip visualization:
```bash
python -m dqm_pipeline --csv data.csv --no-plots
```

## Future Enhancements

- [ ] Real-time streaming support
- [ ] Database export (MongoDB, PostgreSQL)
- [ ] Hyperparameter optimization
- [ ] Explainability features (SHAP, attention visualization)
- [ ] Web dashboard for results
- [ ] Parallel processing for large datasets

## Contact

For questions or issues, contact Sophie (mu2e DQM team).
