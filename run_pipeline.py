"""
DQM Pipeline Runner

Master orchestrator for the complete anomaly detection pipeline:
  Phase 1: Ingestion   - Load and validate data
  Phase 2: Preprocessing - Feature scaling and selection
  Phase 3: Models      - Run anomaly detection (Z-Score, PCA, TranAD)
  Phase 4: Export      - Generate reports and visualizations
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from . import ingestion, preprocessing, models
from .visualization import DQMVisualizer
from .diagnostics import AnomalyDiagnostics


def create_report(df: pd.DataFrame,
                 active_features: list,
                 model_results: dict,
                 output_path: str = "anomaly_report.csv") -> None:
    """
    Generate anomaly detection report.
    Uses TranAD as primary anomaly predictions, includes all three methods for evaluation.
    
    Args:
        df: Original DataFrame
        active_features: List of features used
        model_results: Results from ensemble detection (contains all predictions)
        output_path: Path to save CSV report
    """
    report_df = df.copy()
    # Primary anomaly column: TranAD predictions
    report_df['anomaly'] = model_results['tranad_predictions']
    # Individual detector predictions for evaluation
    report_df['z_score_anomaly'] = model_results['z_score_predictions']
    report_df['pca_anomaly'] = model_results['pca_predictions']
    report_df['tranad_anomaly'] = model_results['tranad_predictions']
    report_df['ensemble_voting_score'] = model_results['voting_scores']
    # Anomaly scores from each detector
    report_df['z_score_max'] = model_results['z_score_scores']
    report_df['pca_reconstruction_error'] = model_results['pca_scores']
    report_df['tranad_reconstruction_error'] = model_results['tranad_scores']
    
    # Save full report
    report_df.to_csv(output_path, index=False)
    print(f"\n✓ Report saved: {output_path}")
    
    # Summarize anomalies (TranAD-based)
    anomalies_df = report_df[report_df['anomaly'] == 1].copy()
    
    if len(anomalies_df) > 0:
        print(f"\nAnomalies ({len(anomalies_df)} records):")
        print("-" * 70)
        
        anomaly_path = output_path.replace('.csv', '_anomalies_only.csv')
        anomalies_df.to_csv(anomaly_path, index=False)
        print(f"\n✓ Anomalies report: {anomaly_path}")
    else:
        print("\n✓ No anomalies detected!")


def main(csv_path: str,
         output_dir: str = "./results",
         iqr_multiplier: float = 1.5,
         voting_threshold: float = 2.0,
         threshold_percentile: float = 95,
         generate_plots: bool = True,
         apply_iqr_filtering: bool = False) -> None:
    """
    Run complete pipeline.
    
    Args:
        csv_path: Path to filtered CSV
        output_dir: Directory for output reports
        iqr_multiplier: IQR multiplier for outlier detection (only used if apply_iqr_filtering=True)
        voting_threshold: Voting threshold for ensemble
        threshold_percentile: TranAD threshold percentile (default: 95)
        generate_plots: Whether to generate visualization plots
        apply_iqr_filtering: Whether to apply IQR-based outlier removal (default: False to preserve anomalies)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dir = output_dir / "plots" if generate_plots else None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("DQM ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    print(f"Start time: {timestamp}")
    print(f"CSV input:  {csv_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 70 + "\n")
    
    # =========================================================================
    # PHASE 1: INGESTION
    # =========================================================================
    print("\n[PHASE 1] INGESTION")
    print("-" * 70)
    df, feature_cols = ingestion.ingest_data(csv_path)
    
    # =========================================================================
    # PHASE 2: PREPROCESSING
    # =========================================================================
    print("\n[PHASE 2] PREPROCESSING")
    print("-" * 70)
    df_clean, active_features, X_scaled, scaler = preprocessing.preprocess_pipeline(
        df, feature_cols, iqr_multiplier=iqr_multiplier, apply_iqr_filtering=apply_iqr_filtering
    )
    
    # =========================================================================
    # PHASE 3: MODELS (ANOMALY DETECTION)
    # =========================================================================
    print("\n[PHASE 3] ANOMALY DETECTION")
    print("-" * 70)
    anomaly_pred, model_results = models.run_ensemble_detection(
        X_scaled, df_clean, active_features, voting_threshold=voting_threshold,
        threshold_percentile=threshold_percentile
    )
    
    # =========================================================================
    # PHASE 4: REPORTING
    # =========================================================================
    print("\n[PHASE 4] REPORTING")
    print("-" * 70)
    report_path = output_dir / f"anomaly_report_{timestamp}.csv"
    create_report(df_clean, active_features, model_results, str(report_path))
    
    # Add TranAD anomaly column to df_clean for diagnostics and visualization
    df_clean = df_clean.copy()
    df_clean['anomaly'] = model_results['tranad_predictions']
    
    # =========================================================================
    # PHASE 5: DIAGNOSTICS
    # =========================================================================
    print("\n[PHASE 5] DIAGNOSTICS")
    print("-" * 70)
    diagnostics = AnomalyDiagnostics(df_clean, active_features)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics.export_diagnostics(
        model_results['z_score_predictions'],
        model_results['pca_predictions'],
        model_results['tranad_predictions'],
        anomaly_pred,
        diagnostics_dir
    )
    
    # =========================================================================
    # PHASE 6: VISUALIZATION (OPTIONAL)
    # =========================================================================
    if generate_plots:
        print("\n[PHASE 6] VISUALIZATION")
        print("-" * 70)
        visualizer = DQMVisualizer(output_dir=str(plot_dir))
        
        # Plot training curve if available
        if 'tranad_losses' in model_results and len(model_results.get('tranad_losses', [])) > 0:
            visualizer.plot_tranad_training(
                model_results['tranad_losses'],
                detector_name='TranAD'
            )
        
        # Generate all other plots using TranAD predictions
        visualizer.generate_all_plots(
            df_clean,
            active_features,
            model_results['z_score_scores'],
            model_results['pca_scores'],
            model_results['tranad_scores'],
            anomaly_col='anomaly',
            tranad_losses=None,  # Already plotted above
            anomaly_predictions=model_results['tranad_predictions'],
            model_results=model_results  # Pass model results to use actual thresholds
        )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Records processed:   {len(df)}")
    print(f"Records after clean: {len(df_clean)}")
    print(f"Features analyzed:   {len(active_features)}")
    tranad_anomalies = int(model_results['tranad_predictions'].sum())
    ensemble_anomalies = int(anomaly_pred.sum())
    print(f"TranAD Anomalies:    {tranad_anomalies} ({100*tranad_anomalies/len(df_clean):.2f}%)")
    print(f"Ensemble Anomalies:  {ensemble_anomalies} (for reference, see diagnostics)")
    print(f"Output dir:          {output_dir}")
    if generate_plots:
        print(f"Plot dir:            {plot_dir}")
    print("=" * 70 + "\n")


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="DQM Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dqm_pipeline --csv data.csv
  python -m dqm_pipeline --csv data.csv --output ./results --iqr 1.5 --plots
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to filtered DQM CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for reports (default: ./results)"
    )
    parser.add_argument(
        "--iqr",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)"
    )
    parser.add_argument(
        "--voting-threshold",
        type=float,
        default=2.0,
        help="Voting threshold for ensemble (default: 2.0)"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95,
        help="TranAD threshold percentile (default: 95, try 90 or 97 to adjust sensitivity)"
    )
    parser.add_argument(
        "--apply-iqr",
        action="store_true",
        help="Apply IQR-based outlier filtering in preprocessing (default: False to preserve anomalies)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate visualization plots (default: True)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization plots"
    )
    
    args = parser.parse_args()
    generate_plots = not args.no_plots
    
    try:
        main(
            csv_path=args.csv,
            output_dir=args.output,
            iqr_multiplier=args.iqr,
            voting_threshold=args.voting_threshold,
            threshold_percentile=args.threshold_percentile,
            generate_plots=generate_plots,
            apply_iqr_filtering=args.apply_iqr
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
