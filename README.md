# DQM Anomaly Detection Pipeline

A modular, production-ready pipeline for detecting anomalies in CRV (Cosmic Ray Veto) detector data from the Mu2e experiment. **Version 0.2.0**

## Overview

The pipeline processes filtered DQM (Data Quality Monitoring) datasets through six phases. Before starting the pipeline, data must be exported from DQM and filtered.

### Step 0: Data Export & Filtering (Pre-processing)

This step extracts raw DQM metrics and removes uninformative features:

#### 0a. Export DQM Metrics
```bash
python export_from_dqm.py
```
- Queries DQM database for CRV detector metrics
- Creates `dqm_crv_anomaly_dataset_004-000.csv` (raw data with all features)
- Runs DQM queries in isolated subprocesses to prevent C++ segfaults
- Returns run, subrun, timestamp, and all registered CRV metrics

#### 0b. Filter Static Columns
```bash
python filter_from_dqm.py
```
- Removes uninformative columns (constant values, all 1.0, etc.)
- Creates `dqm_crv_anomaly_dataset_004-000_filtered.csv` (clean input for pipeline)
- Preserves metadata columns (run, subrun, timestamp)
- Keeps only features with actual variance

**Output**: `dqm_crv_anomaly_dataset_004-000_filtered.csv` → Ready for Phase 1 (Ingestion)

### Phase 1: Ingestion
- Load filtered CSV from Step 0b (`dqm_crv_anomaly_dataset_*_filtered.csv`)
- Parse and validate ISO 8601 timestamps (handles timezone info)
- Perform data quality checks (missing values, data types, etc.)

### Phase 2: Preprocessing
- Identify active (non-flat) features
- ✅ **IQR-based outlier filtering is now DISABLED by default** to preserve potential anomalies
- Standardize feature scaling (mean=0, std=1)
- Ready data for anomaly detection models

### Phase 3: Anomaly Detection (Ensemble & TranAD Primary)
- **Z-Score Detector** - Statistical anomaly detection (simple, interpretable)
- **PCA Detector** - Multivariate outlier detection via reconstruction error
- **TranAD Detector** - Transformer-based deep learning model (requires PyTorch) - **Primary detector**
- **Ensemble Voting** - Combine all three detectors for cross-validation and agreement analysis

### Phase 4: Reporting
- Generate detailed anomaly reports (CSV) with TranAD as primary predictions
- Include individual detector scores and ensemble voting results
- Separate anomalies-only report for quick review
- Export comprehensive model results with reconstruction errors and anomaly scores

### Phase 5: Diagnostics
- Feature contribution analysis (normal vs anomalous records)
- Detector agreement/disagreement pattern analysis
- Statistical comparisons between anomaly and normal records
- Temporal pattern identification
- Ensemble voting breakdown

### Phase 6: Visualization
- Time series analysis (4-panel overview)
- TranAD training curve with epoch and batch-level losses
- Individual detector score timelines with anomaly highlights
- Ensemble voting score timeline
- Detector comparison (all 3 detectors side-by-side)
- Anomaly score distributions
- Feature correlations (heatmap)
- Anomaly timeline highlighting
- Feature distributions (top 6 features)

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

### Complete Workflow (Step 0 → Phases 1-6)

```bash
# Step 0a: Extract DQM metrics from database
cd dqm_pipeline
python export_from_dqm.py

# Step 0b: Filter out static/uninformative columns
python filter_from_dqm.py

# Phase 1-6: Run anomaly detection pipeline on filtered data
python -m dqm_pipeline --csv dqm_crv_anomaly_dataset_004-000_filtered.csv
```

### Command Line (Simplest - Phase 1-6 only)

Run with default parameters and visualization:
```bash
python -m dqm_pipeline --csv dqm_crv_anomaly_dataset_004-000_filtered.csv
```

With custom parameters:
```bash
python -m dqm_pipeline \
  --csv data.csv \
  --output ./results \
  --iqr 3.0 \
  --voting-threshold 2.0 \
  --threshold-percentile 95 \
  --apply-iqr \
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

# Run full pipeline with plots (IQR filtering disabled by default)
main(
    csv_path="data.csv",
    output_dir="./results",
    iqr_multiplier=1.5,
    voting_threshold=2.0,
    threshold_percentile=95,
    generate_plots=True,
    apply_iqr_filtering=False  # Keep disabled to preserve anomalies
)

# Or enable IQR filtering only for training data cleanup
main(
    csv_path="training_data.csv",
    output_dir="./results",
    iqr_multiplier=3.0,  # Use conservative setting if filtering
    apply_iqr_filtering=True  # Only for training data, not test/detection
)

# Or run individual phases
from dqm_pipeline import ingestion, preprocessing, models

df, features = ingestion.ingest_data("data.csv")
df_clean, active_features, X_scaled, scaler = preprocessing.preprocess_pipeline(
    df, features, iqr_multiplier=1.5, apply_iqr_filtering=False
)
predictions, results = models.run_ensemble_detection(
    X_scaled, df_clean, active_features, 
    threshold_percentile=95
)
```

## Model Selection & TranAD

**TranAD is the primary anomaly detector** in the pipeline. Here's why:

- **Transformer Architecture**: Captures temporal patterns and dependencies in detector data
- **Reconstruction Error**: Uses reconstruction errors as anomaly scores (unsupervised)
- **Deep Learning**: More expressive than statistical methods for complex patterns
- **Threshold Percentile**: Dynamically sets thresholds based on reconstruction error distribution
- **Cross-validation**: Z-Score and PCA detectors provide ensemble voting for validation

### Detector Roles:
| Detector | Role | Strength | Use Case |
|----------|------|----------|----------|
| **TranAD** | Primary | Deep learning, temporal patterns | Main anomaly detection |
| Z-Score | Validation | Simple, interpretable | Statistical confirmation |
| PCA | Validation | Multivariate, linear relationships | Feature correlation check |
| Ensemble | Cross-check | Majority voting | Confidence assessment |

The anomaly reports and visualizations use TranAD predictions as the primary `anomaly` column, with ensemble voting available for validation and threshold analysis.

## Configuration

### IQR Multiplier & Filtering
- **Disabled by default** to preserve potential anomalies for detection
- Enable with `--apply-iqr` flag if needed for data quality cleaning (training data only)
- When enabled:
  - **1.5** (aggressive) - Removes ~7% of data, may eliminate true anomalies
  - **3.0** (conservative) - Removes ~0.3% of data, safer
- Default: **Not applied** (to avoid removing anomalies)
- ⚠️ **Best Practice**: Do NOT apply IQR filtering to data intended for anomaly detection; only use for training set cleanup if needed

### TranAD Threshold Percentile
- Percentile of reconstruction errors used to determine anomaly threshold
- **95** (default) - Top 5% of reconstruction errors flagged as anomalies
- **90** - More conservative, top 10% flagged
- Higher percentile = fewer anomalies detected
- Note: TranAD serves as the primary anomaly detector; ensemble voting is for validation

### Voting Threshold (Ensemble Agreement for Cross-validation)
- Used for diagnostics and validation (TranAD is primary detector)
- **1.0** - Any single detector can flag (low confidence)
- **2.0** - At least 2 of 3 detectors must agree (medium confidence)
- **3.0** - All 3 detectors must agree (high confidence, most conservative)
- Default: **2.0**

## Diagnostics Module

The diagnostics module analyzes detected anomalies to understand why they were flagged:

### Diagnostic Outputs:
- **Feature Statistics**: Compare mean, std, min, max of each feature between normal and anomalous records
- **Detector Agreement**: Show voting patterns and disagreement between the three detectors
- **Temporal Patterns**: Analyze how anomalies are distributed over time
- **Feature Contributions**: Identify which features most distinguish anomalies from normal data

These help validate anomaly detection results and understand detector behavior.

## Output Files

The pipeline generates:

```
results/
├── anomaly_report_YYYYMMDD_HHMMSS.csv                    # Full results with all scores
├── anomaly_report_YYYYMMDD_HHMMSS_anomalies_only.csv     # Anomalies only (TranAD-based)
├── diagnostics/
│   ├── feature_statistics.csv                            # Normal vs anomalous feature stats
│   ├── detector_agreement.csv                            # Detector agreement analysis
│   └── temporal_patterns.csv                             # Temporal pattern analysis
└── plots/
    ├── tranad_training_curve.png                         # TranAD training losses (epoch + batch)
    ├── timeseries_overview.png                           # 4-panel time series overview
    ├── z_score_scores_highlighted.png                    # Z-Score detector with anomalies
    ├── pca_scores_highlighted.png                        # PCA detector with anomalies
    ├── tranad_scores_highlighted.png                     # TranAD detector with anomalies (primary)
    ├── ensemble_scores_highlighted.png                   # Ensemble voting results
    ├── detector_comparison.png                           # All 3 detectors side-by-side
    ├── anomaly_distribution.png                          # Pie/bar charts of anomalies
    ├── anomaly_timeline.png                              # Anomalies highlighted on timeline
    ├── feature_distributions.png                         # Distribution of top 6 features
    └── correlation_heatmap.png                           # Feature correlations (heatmap)
```

### Report Columns
- `run, subrun, timestamp, datetime` - Record identifiers
- `anomaly` - **Primary anomaly prediction (1 = anomaly)** - from TranAD detector
- `z_score_anomaly, pca_anomaly, tranad_anomaly` - Individual detector predictions (0/1)
- `ensemble_voting_score` - Sum of detector votes (0-3, for validation)
- `z_score_max` - Max Z-score value (anomaly severity measure)
- `pca_reconstruction_error` - PCA reconstruction error value
- `tranad_reconstruction_error` - TranAD reconstruction error value (primary scorer)
- All original feature columns

### Diagnostics Files
- `feature_statistics.csv` - Mean, std, min, max of each feature for normal vs anomalous records
- `detector_agreement.csv` - Voting patterns and agreement between detectors
- `temporal_patterns.csv` - Anomaly distribution over time

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
├── __main__.py              # CLI entry point (python -m dqm_pipeline)
├── export_from_dqm.py       # Step 0a: Extract DQM metrics from database
├── filter_from_dqm.py       # Step 0b: Filter static/uninformative columns
├── run_pipeline.py          # Master orchestrator & CLI interface
├── ingestion.py             # Phase 1: Data loading & validation
├── preprocessing.py         # Phase 2: Feature scaling & selection
├── models.py                # Phase 3: Anomaly detectors (Z-Score, PCA, TranAD ensemble)
├── tranad_model.py          # TranAD transformer implementation (optional)
├── diagnostics.py           # Phase 5: Anomaly analysis & diagnostics
├── visualization.py         # Phase 6: Plot generation
├── setup.py                 # Package installation
└── README.md                # This file
```

## Performance Notes (crv-004)

- **Dataset**: ~1100+ records × 37 features (wide-format)
- **Processing Time**: 
  - 5-10 seconds total with TranAD (including training)
  - 1-2 seconds without TranAD
  - 2-3 seconds for visualization plots (if enabled)
- **Memory**: 
  - ~150-200 MB with TranAD training
  - ~100 MB for inference only
  - ~50-100 MB without TranAD
- **GPU Support**: TranAD can use CUDA if available (automatic detection)
- **Scalability**: Tested up to 1000+ records; design supports larger datasets

## Examples

### Example 0: Full Workflow (Export → Filter → Detect)
```bash
cd dqm_pipeline
# Export metrics from DQM database
python export_from_dqm.py

# Filter static columns
python filter_from_dqm.py

# Run pipeline on filtered data
python -m dqm_pipeline --csv dqm_crv_anomaly_dataset_004-000_filtered.csv
```

### Example 1: Basic Pipeline (Recommended for Anomaly Detection)
```bash
python -m dqm_pipeline --csv Genesis/dqm_crv_anomaly_dataset_004-000_filtered.csv
```

### Example 2: With Conservative IQR Filtering (Training Data Cleanup)
```bash
python -m dqm_pipeline \
  --csv training_data.csv \
  --iqr 3.0 \
  --apply-iqr
```

### Example 3: Conservative Anomaly Detection
```bash
python -m dqm_pipeline \
  --csv data.csv \
  --voting-threshold 3.0 \
  --threshold-percentile 90
```

### Example 4: Aggressive Anomaly Detection with Lower Voting
```bash
python -m dqm_pipeline \
  --csv data.csv \
  --voting-threshold 1.0 \
  --threshold-percentile 97
```

### Example 5: Skip Visualization (Faster)
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
- [ ] Hyperparameter optimization (automated threshold tuning)
- [ ] Explainability features (SHAP values, attention visualization)
- [ ] Web dashboard for interactive results exploration
- [ ] Parallel processing for very large datasets (>10k records)
- [ ] Multi-dataset analysis and comparison
- [ ] Integration with Jupyter for exploratory analysis
- [ ] Train/test split mode for proper cross-validation

## Contact

For questions or issues, contact Sophie.
