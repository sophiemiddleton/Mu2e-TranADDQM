"""
Diagnostic Module for Anomaly Analysis

Provides utilities to understand detected anomalies:
- Feature contribution analysis
- Detector agreement/disagreement patterns
- Statistical comparisons (anomaly vs normal)
- Temporal patterns
- Voting ensemble breakdown
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class AnomalyDiagnostics:
    """Tools for understanding and analyzing detected anomalies."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Initialize diagnostics.
        
        Args:
            df: DataFrame with all data
            feature_cols: List of feature column names
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
    
    def feature_statistics(self, anomaly_col: str = 'anomaly') -> pd.DataFrame:
        """
        Compare feature statistics between normal and anomalous records.
        
        Args:
            anomaly_col: Column with binary anomaly labels
            
        Returns:
            DataFrame with statistics comparison
        """
        normal_mask = self.df[anomaly_col] == 0
        anomaly_mask = self.df[anomaly_col] == 1
        
        results = []
        
        for feat in self.feature_cols:
            normal_vals = pd.to_numeric(self.df.loc[normal_mask, feat], errors='coerce')
            anomaly_vals = pd.to_numeric(self.df.loc[anomaly_mask, feat], errors='coerce')
            
            results.append({
                'feature': feat,
                'normal_mean': normal_vals.mean(),
                'normal_std': normal_vals.std(),
                'anomaly_mean': anomaly_vals.mean(),
                'anomaly_std': anomaly_vals.std(),
                'mean_diff': abs(anomaly_vals.mean() - normal_vals.mean()),
                'mean_diff_pct': 100 * abs(anomaly_vals.mean() - normal_vals.mean()) / abs(normal_vals.mean() + 1e-10),
            })
        
        stats_df = pd.DataFrame(results)
        stats_df = stats_df.sort_values('mean_diff_pct', ascending=False)
        
        return stats_df
    
    def detector_agreement(self, z_score_pred: np.ndarray, pca_pred: np.ndarray, 
                          tranad_pred: np.ndarray) -> Dict:
        """
        Analyze agreement/disagreement between detectors.
        
        Args:
            z_score_pred: Z-Score binary predictions
            pca_pred: PCA binary predictions
            tranad_pred: TranAD binary predictions
            
        Returns:
            Dictionary with agreement statistics
        """
        # All three agree on anomaly
        all_agree_anomaly = (z_score_pred == 1) & (pca_pred == 1) & (tranad_pred == 1)
        
        # All three agree on normal
        all_agree_normal = (z_score_pred == 0) & (pca_pred == 0) & (tranad_pred == 0)
        
        # Only one detector flags anomaly
        only_z_score = (z_score_pred == 1) & (pca_pred == 0) & (tranad_pred == 0)
        only_pca = (z_score_pred == 0) & (pca_pred == 1) & (tranad_pred == 0)
        only_tranad = (z_score_pred == 0) & (pca_pred == 0) & (tranad_pred == 1)
        
        # Exactly two detectors flag anomaly
        z_score_pca = (z_score_pred == 1) & (pca_pred == 1) & (tranad_pred == 0)
        z_score_tranad = (z_score_pred == 1) & (pca_pred == 0) & (tranad_pred == 1)
        pca_tranad = (z_score_pred == 0) & (pca_pred == 1) & (tranad_pred == 1)
        
        return {
            'all_agree_anomaly': int(all_agree_anomaly.sum()),
            'all_agree_normal': int(all_agree_normal.sum()),
            'only_z_score': int(only_z_score.sum()),
            'only_pca': int(only_pca.sum()),
            'only_tranad': int(only_tranad.sum()),
            'z_score_pca': int(z_score_pca.sum()),
            'z_score_tranad': int(z_score_tranad.sum()),
            'pca_tranad': int(pca_tranad.sum()),
            'pct_all_agree_anomaly': 100 * all_agree_anomaly.sum() / (all_agree_anomaly.sum() + 1e-10),
        }
    
    def voting_breakdown(self, z_score_pred: np.ndarray, pca_pred: np.ndarray, 
                        tranad_pred: np.ndarray, ensemble_pred: np.ndarray) -> Dict:
        """
        Analyze voting patterns in ensemble.
        
        Args:
            z_score_pred: Z-Score binary predictions
            pca_pred: PCA binary predictions
            tranad_pred: TranAD binary predictions
            ensemble_pred: Final ensemble predictions (majority voting)
            
        Returns:
            Dictionary with voting statistics
        """
        votes = z_score_pred + pca_pred + tranad_pred
        
        # Count by number of votes
        zero_votes = (votes == 0).sum()
        one_vote = (votes == 1).sum()
        two_votes = (votes == 2).sum()
        three_votes = (votes == 3).sum()
        
        # Voting errors (ensemble gets it wrong if 2 detectors agree but shouldn't)
        # Unanimous anomaly detection
        unanimous_anomaly = three_votes
        # Unanimous normal detection
        unanimous_normal = zero_votes
        # Split decisions (harder cases)
        split_decisions = one_vote + two_votes
        
        return {
            'zero_votes': int(zero_votes),
            'one_vote': int(one_vote),
            'two_votes': int(two_votes),
            'three_votes': int(three_votes),
            'unanimous_anomaly': int(unanimous_anomaly),
            'unanimous_normal': int(unanimous_normal),
            'split_decisions': int(split_decisions),
            'pct_split': 100 * split_decisions / len(ensemble_pred),
        }
    
    def temporal_patterns(self, anomaly_col: str = 'anomaly') -> Dict:
        """
        Analyze temporal distribution of anomalies.
        
        Args:
            anomaly_col: Column with binary anomaly labels
            
        Returns:
            Dictionary with temporal statistics
        """
        if 'datetime' not in self.df.columns:
            if 'timestamp' in self.df.columns:
                df = self.df.copy()
                df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            else:
                return {'error': 'No datetime or timestamp column found'}
        else:
            df = self.df
        
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        anomalies = df[df[anomaly_col] == 1]
        
        results = {
            'total_anomalies': len(anomalies),
            'total_records': len(df),
            'anomaly_rate': 100 * len(anomalies) / len(df),
            'dates_with_anomalies': int(anomalies['date'].nunique()),
            'hours_with_anomalies': sorted(anomalies['hour'].unique().tolist()),
            'most_common_hour': int(anomalies['hour'].mode()[0]) if len(anomalies) > 0 else None,
        }
        
        # Anomalies by date
        daily_anomalies = anomalies.groupby('date').size()
        results['daily_anomaly_stats'] = {
            'min': int(daily_anomalies.min()),
            'max': int(daily_anomalies.max()),
            'mean': float(daily_anomalies.mean()),
            'std': float(daily_anomalies.std()),
        }
        
        return results
    
    def outlier_feature_analysis(self, anomaly_col: str = 'anomaly', 
                                 z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Identify features that are extreme (>z_threshold std) in anomalies.
        
        Args:
            anomaly_col: Column with binary anomaly labels
            z_threshold: Z-score threshold for "extreme"
            
        Returns:
            DataFrame with feature extremeness analysis
        """
        normal_mask = self.df[anomaly_col] == 0
        anomaly_mask = self.df[anomaly_col] == 1
        
        results = []
        
        for feat in self.feature_cols:
            normal_vals = pd.to_numeric(self.df.loc[normal_mask, feat], errors='coerce')
            anomaly_vals = pd.to_numeric(self.df.loc[anomaly_mask, feat], errors='coerce')
            
            normal_mean = normal_vals.mean()
            normal_std = normal_vals.std()
            
            if normal_std > 0:
                # Count anomalies with extreme values
                z_scores = np.abs((anomaly_vals - normal_mean) / normal_std)
                extreme_count = (z_scores > z_threshold).sum()
                extreme_pct = 100 * extreme_count / len(anomaly_vals) if len(anomaly_vals) > 0 else 0
                
                results.append({
                    'feature': feat,
                    'extreme_anomalies': int(extreme_count),
                    'pct_extreme_in_anomalies': extreme_pct,
                    'max_z_score': float(z_scores.max()),
                })
        
        extreme_df = pd.DataFrame(results)
        extreme_df = extreme_df.sort_values('pct_extreme_in_anomalies', ascending=False)
        
        return extreme_df
    
    def anomaly_summary_report(self, z_score_pred: np.ndarray, pca_pred: np.ndarray,
                              tranad_pred: np.ndarray, ensemble_pred: np.ndarray,
                              output_path: Path = None) -> str:
        """
        Generate comprehensive anomaly diagnostic report.
        
        Args:
            z_score_pred: Z-Score binary predictions
            pca_pred: PCA binary predictions
            tranad_pred: TranAD binary predictions
            ensemble_pred: Final ensemble predictions
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ANOMALY DETECTION DIAGNOSTIC REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall counts
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 40)
        n_anomalies = ensemble_pred.sum()
        n_total = len(ensemble_pred)
        report_lines.append(f"Total Records: {n_total}")
        report_lines.append(f"Detected Anomalies: {int(n_anomalies)}")
        report_lines.append(f"Anomaly Rate: {100 * n_anomalies / n_total:.2f}%")
        report_lines.append("")
        
        # Detector agreement
        report_lines.append("DETECTOR AGREEMENT")
        report_lines.append("-" * 40)
        agreement = self.detector_agreement(z_score_pred, pca_pred, tranad_pred)
        for key, val in agreement.items():
            if isinstance(val, float):
                report_lines.append(f"{key}: {val:.2f}")
            else:
                report_lines.append(f"{key}: {val}")
        report_lines.append("")
        
        # Voting breakdown
        report_lines.append("VOTING BREAKDOWN")
        report_lines.append("-" * 40)
        voting = self.voting_breakdown(z_score_pred, pca_pred, tranad_pred, ensemble_pred)
        for key, val in voting.items():
            if isinstance(val, float):
                report_lines.append(f"{key}: {val:.2f}")
            else:
                report_lines.append(f"{key}: {val}")
        report_lines.append("")
        
        # Feature statistics (top 5)
        report_lines.append("TOP 5 FEATURES BY DIFFERENCE (Normal vs Anomaly)")
        report_lines.append("-" * 40)
        feat_stats = self.feature_statistics()
        for idx, row in feat_stats.head(5).iterrows():
            report_lines.append(f"{row['feature']:20s} | Diff: {row['mean_diff']:10.4f} ({row['mean_diff_pct']:6.2f}%)")
        report_lines.append("")
        
        # Extreme features (top 5)
        report_lines.append("TOP 5 FEATURES BY EXTREMENESS IN ANOMALIES")
        report_lines.append("-" * 40)
        extreme_feats = self.outlier_feature_analysis()
        for idx, row in extreme_feats.head(5).iterrows():
            report_lines.append(f"{row['feature']:20s} | Extreme: {int(row['extreme_anomalies']):4d} ({row['pct_extreme_in_anomalies']:5.1f}%) | Max Z: {row['max_z_score']:6.2f}")
        report_lines.append("")
        
        # Temporal patterns
        report_lines.append("TEMPORAL PATTERNS")
        report_lines.append("-" * 40)
        temporal = self.temporal_patterns()
        for key, val in temporal.items():
            if key == 'daily_anomaly_stats':
                report_lines.append(f"Daily Anomaly Statistics:")
                for k, v in val.items():
                    report_lines.append(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
            elif key != 'error':
                if isinstance(val, float):
                    report_lines.append(f"{key}: {val:.2f}")
                elif isinstance(val, list):
                    report_lines.append(f"{key}: {val}")
                else:
                    report_lines.append(f"{key}: {val}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report_text)
            print(f"✓ Diagnostic report saved to {output_path}")
        
        return report_text
    
    def get_anomaly_records(self, ensemble_pred: np.ndarray, confidence: str = 'all') -> pd.DataFrame:
        """
        Get anomalous records with optional confidence filtering.
        
        Args:
            ensemble_pred: Binary predictions from ensemble
            confidence: 'all' | 'high' (3/3 votes) | 'medium' (2/3 votes)
            
        Returns:
            DataFrame of anomalous records
        """
        anomaly_mask = ensemble_pred == 1
        anomaly_records = self.df[anomaly_mask].copy()
        
        return anomaly_records
    
    def get_top_features(self, n: int = 5, anomaly_col: str = 'anomaly') -> List[str]:
        """
        Get top N features by mean difference between normal and anomalies.
        
        Args:
            n: Number of top features to return
            anomaly_col: Column with binary anomaly labels
            
        Returns:
            List of top N feature names
        """
        stats = self.feature_statistics(anomaly_col)
        return stats.head(n)['feature'].tolist()
    
    def export_diagnostics(self, z_score_pred: np.ndarray, pca_pred: np.ndarray,
                          tranad_pred: np.ndarray, ensemble_pred: np.ndarray,
                          output_dir: Path) -> None:
        """
        Export all diagnostic analysis to files.
        
        Args:
            z_score_pred: Z-Score predictions
            pca_pred: PCA predictions
            tranad_pred: TranAD predictions
            ensemble_pred: Ensemble predictions
            output_dir: Directory to save all diagnostics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Feature statistics
            print("  Computing feature statistics...")
            feat_stats = self.feature_statistics()
            feat_stats.to_csv(output_dir / 'feature_statistics.csv', index=False)
            print(f"  ✓ Saved: feature_statistics.csv")
            
            # Extreme features
            print("  Computing extreme features...")
            extreme_feats = self.outlier_feature_analysis()
            extreme_feats.to_csv(output_dir / 'extreme_features.csv', index=False)
            print(f"  ✓ Saved: extreme_features.csv")
            
            # Anomaly summary report
            print("  Generating diagnostic report...")
            report = self.anomaly_summary_report(
                z_score_pred, pca_pred, tranad_pred, ensemble_pred,
                output_path=output_dir / 'diagnostic_report.txt'
            )
            
            # Print report to console
            print("\n" + report)
            
        except Exception as e:
            print(f"  ✗ Error during diagnostics: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
