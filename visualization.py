"""
Visualization Module for DQM Anomaly Detection

Generates comprehensive plots for anomaly detection results:
- Time series analysis
- Distribution analysis
- Anomaly score comparisons
- Confusion matrices
- Feature correlations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path


class DQMVisualizer:
    """Visualization suite for DQM anomaly detection results."""
    
    def __init__(self, output_dir="./plots", dpi=100, figsize=(14, 8)):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved plots
            figsize: Default figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
    def plot_timeseries_overview(self, df, feature_cols, anomaly_col='anomaly'):
        """
        Create 4-panel time series overview (matches Genesis notebook).
        
        Args:
            df: DataFrame with datetime column and features
            feature_cols: List of feature columns to plot
            anomaly_col: Column name for anomaly labels
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle("Time-Series Analysis Overview", fontsize=14, fontweight='bold')
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        selected_features = feature_cols[:4] if len(feature_cols) >= 4 else feature_cols
        
        # Plot 1: First feature over time
        ax = axes[0, 0]
        if len(selected_features) > 0:
            feat = selected_features[0]
            ax.plot(df['datetime'], pd.to_numeric(df[feat], errors='coerce'),
                   marker='o', markersize=4, linestyle='-', alpha=0.7, label=feat)
            ax.set_ylabel('Value')
            ax.set_title(f'{feat} Over Time')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Second feature over time
        ax = axes[0, 1]
        if len(selected_features) > 1:
            feat = selected_features[1]
            ax.plot(df['datetime'], pd.to_numeric(df[feat], errors='coerce'),
                   marker='s', markersize=4, linestyle='-', alpha=0.7, color='orange', label=feat)
            ax.set_ylabel('Value')
            ax.set_title(f'{feat} Over Time')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Distribution histogram
        ax = axes[1, 0]
        if len(selected_features) > 0:
            feat = selected_features[0]
            vals = pd.to_numeric(df[feat], errors='coerce').dropna()
            ax.hist(vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(feat)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feat}')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Data density over time
        ax = axes[1, 1]
        df_copy = df.copy()
        df_copy['date_only'] = df_copy['datetime'].dt.date
        records_per_date = df_copy.groupby('date_only').size()
        dates_str = [d.strftime('%Y-%m-%d') for d in records_per_date.index]
        ax.bar(dates_str, records_per_date.values, width=0.6, color='teal', alpha=0.75)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Subruns')
        ax.set_title('Data Density Over Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'timeseries_overview.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: timeseries_overview.png")
        plt.close()
    
    def plot_anomaly_scores(self, df, anomaly_scores, detector_name='Ensemble'):
        """
        Plot anomaly scores over time (basic version).
        
        Args:
            df: DataFrame with datetime column
            anomaly_scores: Array of anomaly scores
            detector_name: Name of detector for title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        ax.plot(df['datetime'], anomaly_scores, marker='o', markersize=4, 
               linestyle='-', alpha=0.7, color='darkred', label='Anomaly Score')
        
        # Add threshold line
        threshold = np.percentile(anomaly_scores, 95)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'95th Percentile ({threshold:.3f})')
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f'{detector_name} - Anomaly Scores Over Time')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_scores.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: anomaly_scores.png")
        plt.close()
    
    def plot_tranad_training(self, training_losses, detector_name='TranAD', batch_losses=None):
        """
        Plot TranAD training curve (MSE vs epoch) - raw, unsmoothed.
        
        Args:
            training_losses: List of mean reconstruction errors per epoch
            detector_name: Name of detector for title
            batch_losses: Optional list of batch-level losses for detailed view
        """
        if not training_losses or len(training_losses) == 0:
            print("⚠  No training losses available for TranAD")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = np.arange(1, len(training_losses) + 1)
        
        # Plot raw epoch-level losses (no smoothing)
        ax.plot(epochs, training_losses, marker='o', markersize=5, 
               linestyle='-', linewidth=1.5, alpha=0.8, color='purple', label='Training MSE')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.set_title(f'{detector_name} - Training Loss vs Epoch')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tranad_training_curve.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: tranad_training_curve.png")
        plt.close()
    
    def plot_anomaly_scores_highlighted(self, df, anomaly_scores, anomaly_predictions, 
                                       detector_name='Ensemble', threshold=None):
        """
        Plot anomaly scores over time with identified anomalies highlighted.
        
        Args:
            df: DataFrame with datetime column
            anomaly_scores: Array of anomaly scores
            anomaly_predictions: Binary array of detected anomalies
            detector_name: Name of detector for title
            threshold: Optional threshold line to display
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        datetime_axis = df['datetime']
        
        # Plot normal records
        normal_mask = anomaly_predictions == 0
        anomaly_mask = anomaly_predictions == 1
        
        ax.scatter(datetime_axis[normal_mask], anomaly_scores[normal_mask],
                  marker='o', s=30, alpha=0.5, color='blue', label='Normal')
        
        # Plot anomalies with different marker
        if anomaly_mask.sum() > 0:
            ax.scatter(datetime_axis[anomaly_mask], anomaly_scores[anomaly_mask],
                      marker='X', s=150, alpha=0.9, color='red', label='Anomaly', zorder=5)
        
        # Add threshold line if provided
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold ({threshold:.3f})', alpha=0.7)
        else:
            # Use 95th percentile as reference
            threshold = np.percentile(anomaly_scores, 95)
            ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1.5, 
                      label=f'95th percentile ({threshold:.3f})', alpha=0.5)
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Anomaly Score')
        ax.set_title(f'{detector_name} - Anomaly Scores with Detected Anomalies', 
                    fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{detector_name.lower()}_scores_highlighted.png', 
                   dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: {detector_name.lower()}_scores_highlighted.png")
        plt.close()
    
    def plot_detector_comparison(self, df, z_score_scores, pca_scores, tranad_scores):
        """
        Compare anomaly scores from all three detectors.
        
        Args:
            df: DataFrame with datetime column
            z_score_scores: Z-Score anomaly scores
            pca_scores: PCA reconstruction errors
            tranad_scores: TranAD reconstruction errors
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('Anomaly Detector Comparison', fontsize=14, fontweight='bold')
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        datetime_axis = df['datetime']
        
        # Z-Score
        axes[0].plot(datetime_axis, z_score_scores, marker='o', markersize=3, 
                    linestyle='-', alpha=0.7, color='blue', label='Z-Score')
        axes[0].set_ylabel('Max Z-Score')
        axes[0].set_title('Z-Score Detector')
        axes[0].axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # PCA
        axes[1].plot(datetime_axis, pca_scores, marker='s', markersize=3,
                    linestyle='-', alpha=0.7, color='green', label='PCA')
        axes[1].set_ylabel('Reconstruction Error')
        axes[1].set_title('PCA Detector')
        pca_threshold = np.percentile(pca_scores, 95)
        axes[1].axhline(y=pca_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # TranAD
        axes[2].plot(datetime_axis, tranad_scores, marker='^', markersize=3,
                    linestyle='-', alpha=0.7, color='purple', label='TranAD')
        axes[2].set_ylabel('Reconstruction Error')
        axes[2].set_xlabel('Timestamp')
        axes[2].set_title('TranAD Detector')
        tranad_threshold = np.percentile(tranad_scores, 95)
        axes[2].axhline(y=tranad_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detector_comparison.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: detector_comparison.png")
        plt.close()
    
    def plot_anomaly_distribution(self, df, anomaly_col='anomaly'):
        """
        Plot distribution of anomalies vs normal records.
        
        Args:
            df: DataFrame with anomaly labels
            anomaly_col: Column name for anomaly labels
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Anomaly Distribution', fontsize=14, fontweight='bold')
        
        # Pie chart
        counts = df[anomaly_col].value_counts()
        labels = ['Normal' if x == 0 else 'Anomaly' for x in counts.index]
        colors = ['#2ecc71', '#e74c3c']
        axes[0].pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors,
                   startangle=90, textprops={'fontsize': 11})
        axes[0].set_title('Record Composition')
        
        # Bar chart
        axes[1].bar(labels, counts.values, color=colors, alpha=0.75, edgecolor='black')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Anomaly Count')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, v in enumerate(counts.values):
            axes[1].text(i, v + max(counts.values) * 0.02, str(v), 
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_distribution.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: anomaly_distribution.png")
        plt.close()
    
    def plot_feature_distributions(self, df, feature_cols, n_features=6):
        """
        Plot distributions for top N features.
        
        Args:
            df: DataFrame
            feature_cols: List of feature columns
            n_features: Number of features to plot
        """
        n_feat = min(n_features, len(feature_cols))
        n_cols = 3
        n_rows = (n_feat + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        
        for idx, feat in enumerate(feature_cols[:n_feat]):
            vals = pd.to_numeric(df[feat], errors='coerce').dropna()
            axes[idx].hist(vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].set_xlabel(feat)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution: {feat}')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_feat, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: feature_distributions.png")
        plt.close()
    
    def plot_correlation_heatmap(self, df, feature_cols, anomaly_col='anomaly'):
        """
        Plot correlation heatmap for features and anomalies.
        
        Args:
            df: DataFrame
            feature_cols: List of feature columns
            anomaly_col: Column name for anomaly labels
        """
        # Select numeric features
        numeric_cols = []
        for col in feature_cols[:10]:  # Limit to top 10 features
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    numeric_cols.append(col)
            except:
                pass
        
        if len(numeric_cols) == 0:
            print("⚠  No numeric features found for correlation heatmap")
            return
        
        # Create correlation matrix
        corr_data = df[numeric_cols + [anomaly_col]].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Heatmap (including Anomalies)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: correlation_heatmap.png")
        plt.close()
    
    def plot_anomaly_timeline(self, df, anomaly_col='anomaly'):
        """
        Highlight anomalies on timeline.
        
        Args:
            df: DataFrame with datetime and anomaly columns
            anomaly_col: Column name for anomaly labels
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        # Plot all records as scatter
        normal_mask = df[anomaly_col] == 0
        anomaly_mask = df[anomaly_col] == 1
        
        ax.scatter(df.loc[normal_mask, 'datetime'], 
                  np.ones(normal_mask.sum()) * 0,
                  c='green', marker='o', s=30, alpha=0.6, label='Normal')
        
        if anomaly_mask.sum() > 0:
            ax.scatter(df.loc[anomaly_mask, 'datetime'],
                      np.ones(anomaly_mask.sum()) * 1,
                      c='red', marker='X', s=100, alpha=0.8, label='Anomaly')
        
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        ax.set_xlabel('Timestamp')
        ax.set_title('Anomaly Timeline', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_timeline.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: anomaly_timeline.png")
        plt.close()
    
    def plot_top_features_timeseries(self, df, top_features, anomaly_predictions=None, 
                                    anomaly_col='anomaly', title_suffix=''):
        """
        Plot time series for top differencing features with anomalies highlighted.
        
        Args:
            df: DataFrame with datetime and feature columns
            top_features: List of feature names to plot (e.g., top 5)
            anomaly_predictions: Optional binary predictions array for highlighting
            anomaly_col: Column name for anomaly labels (fallback if no predictions)
            title_suffix: Optional suffix for plot title
        """
        if len(top_features) == 0:
            print("⚠  No features to plot")
            return
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        else:
            df = df.copy()
        
        n_features = len(top_features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        fig.suptitle(f'Time Series: Top {n_features} Differencing Features{title_suffix}', 
                    fontsize=14, fontweight='bold')
        
        # Use predictions if provided, else fall back to anomaly_col
        if anomaly_predictions is not None:
            anomaly_mask = anomaly_predictions == 1
        else:
            anomaly_mask = df[anomaly_col] == 1
        
        normal_mask = ~anomaly_mask
        
        for idx, feat in enumerate(top_features):
            ax = axes[idx]
            
            # Skip if feature doesn't exist
            if feat not in df.columns:
                ax.text(0.5, 0.5, f'Feature not found:\n{feat}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feat, fontweight='bold')
                continue
            
            # Convert to numeric
            feat_vals = pd.to_numeric(df[feat], errors='coerce')
            
            # Plot normal records
            ax.scatter(df.loc[normal_mask, 'datetime'], feat_vals[normal_mask],
                      marker='o', s=20, alpha=0.5, color='blue', label='Normal')
            
            # Plot anomalies
            if anomaly_mask.sum() > 0:
                ax.scatter(df.loc[anomaly_mask, 'datetime'], feat_vals[anomaly_mask],
                          marker='X', s=100, alpha=0.9, color='red', label='Anomaly', zorder=5)
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Value')
            ax.set_title(feat, fontweight='bold', fontsize=11)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = 'top_features_timeseries.png'
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def plot_all_features_timeseries(self, df, feature_cols, anomaly_predictions=None, 
                                    anomaly_col='anomaly'):
        """
        Plot time series for all features with anomalies highlighted.
        Creates a timeseries subdirectory with organized plots.
        
        Args:
            df: DataFrame with datetime and feature columns
            feature_cols: List of all feature column names
            anomaly_predictions: Optional binary predictions array for highlighting
            anomaly_col: Column name for anomaly labels (fallback if no predictions)
        """
        if len(feature_cols) == 0:
            print("⚠  No features to plot")
            return
        
        # Create timeseries subdirectory
        ts_dir = self.output_dir / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        else:
            df = df.copy()
        
        # Use predictions if provided, else fall back to anomaly_col
        if anomaly_predictions is not None:
            anomaly_mask = anomaly_predictions == 1
        else:
            anomaly_mask = df[anomaly_col] == 1
        
        normal_mask = ~anomaly_mask
        
        print(f"\n  Plotting {len(feature_cols)} features to {ts_dir}/")
        
        for feat_idx, feat in enumerate(feature_cols):
            # Skip if feature doesn't exist
            if feat not in df.columns:
                continue
            
            # Convert to numeric
            feat_vals = pd.to_numeric(df[feat], errors='coerce')
            
            # Create individual plot
            fig, ax = plt.subplots(figsize=(14, 4))
            
            # Plot normal records
            ax.scatter(df.loc[normal_mask, 'datetime'], feat_vals[normal_mask],
                      marker='o', s=15, alpha=0.4, color='blue', label='Normal')
            
            # Plot anomalies
            if anomaly_mask.sum() > 0:
                ax.scatter(df.loc[anomaly_mask, 'datetime'], feat_vals[anomaly_mask],
                          marker='X', s=80, alpha=0.9, color='red', label='Anomaly', zorder=5)
            
            ax.set_xlabel('Timestamp', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'Time Series: {feat}', fontweight='bold', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save with sanitized filename
            safe_name = feat.replace('/', '_').replace(' ', '_').lower()
            filename = f"{feat_idx+1:03d}_{safe_name}.png"
            plt.savefig(ts_dir / filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            if (feat_idx + 1) % 10 == 0:
                print(f"    ... saved {feat_idx + 1}/{len(feature_cols)} features")
        
        print(f"  ✓ All {len(feature_cols)} feature time series saved to: {ts_dir}")
    
    def generate_all_plots(self, df, feature_cols, z_score_scores, pca_scores, 
                          tranad_scores, anomaly_col='anomaly', 
                          tranad_losses=None, anomaly_predictions=None):
        """
        Generate all visualization plots (except training curve, which is handled separately).
        
        Args:
            df: DataFrame with all data
            feature_cols: List of feature columns
            z_score_scores: Z-Score anomaly scores
            pca_scores: PCA reconstruction errors
            tranad_scores: TranAD reconstruction errors
            anomaly_col: Column name for anomaly labels
            tranad_losses: Ignored (training curve plotted separately in run_pipeline)
            anomaly_predictions: Optional array of binary predictions for highlighting
        """
        print("\n[VISUALIZATION] Generating Additional Plots")
        print("-" * 70)
        
        # Time-series overview
        self.plot_timeseries_overview(df, feature_cols, anomaly_col)
        
        # Anomaly score plots with highlighting
        if anomaly_predictions is not None:
            # Highlighted anomaly score plots for each detector
            z_threshold = np.percentile(z_score_scores, 95)
            self.plot_anomaly_scores_highlighted(df, z_score_scores, anomaly_predictions, 
                                                detector_name='Z-Score', threshold=z_threshold)
            
            pca_threshold = np.percentile(pca_scores, 95)
            self.plot_anomaly_scores_highlighted(df, pca_scores, anomaly_predictions,
                                                detector_name='PCA', threshold=pca_threshold)
            
            tranad_threshold = np.percentile(tranad_scores, 95)
            self.plot_anomaly_scores_highlighted(df, tranad_scores, anomaly_predictions,
                                                detector_name='TranAD', threshold=tranad_threshold)
            
            # Ensemble with voting
            ensemble_predictions = df[anomaly_col].values if anomaly_col in df.columns else anomaly_predictions
            self.plot_anomaly_scores_highlighted(df, tranad_scores, ensemble_predictions,
                                                detector_name='Ensemble', threshold=tranad_threshold)
        else:
            # Fallback to non-highlighted plots
            self.plot_anomaly_scores(df, tranad_scores, detector_name='TranAD')
        
        # Detector comparison
        self.plot_detector_comparison(df, z_score_scores, pca_scores, tranad_scores)
        
        # Distribution and correlations
        self.plot_anomaly_distribution(df, anomaly_col)
        self.plot_feature_distributions(df, feature_cols, n_features=6)
        self.plot_correlation_heatmap(df, feature_cols, anomaly_col)
        self.plot_anomaly_timeline(df, anomaly_col)
        
        # Plot all features time series
        self.plot_all_features_timeseries(df, feature_cols, anomaly_predictions, anomaly_col)
        
        print(f"\n✓ All plots saved to: {self.output_dir}")
