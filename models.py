"""
Phase 3: Anomaly Detection Models

Implementations of three complementary approaches:
  1. Z-Score    - Statistical thresholding (simple, interpretable)
  2. PCA        - Principal Component Analysis (multivariate outliers)
  3. TranAD     - Transformer-based Anomaly Detection (deep learning, for sequences)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from .tranad_model import TranADDetector, TRANAD_ANL_AVAILABLE
except ImportError:
    TRANAD_ANL_AVAILABLE = False
    TranADDetector = None
    print("⚠️  PyTorch/ANL TranAD not available.")


class ZScoreDetector:
    """
    Statistical anomaly detection using Z-scores.
    
    Detects records where any feature deviates more than threshold standard deviations
    from the mean.
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Args:
            threshold: Number of standard deviations to consider anomalous (default: 3σ)
        """
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'ZScoreDetector':
        """Fit scaler on training data."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies.
        
        Returns:
            Binary array: 1 = anomaly, 0 = normal
        """
        if self.mean_ is None:
            raise ValueError("Must call fit() first")
        
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-8))
        anomalies = (z_scores > self.threshold).any(axis=1).astype(int)
        return anomalies
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return max Z-score for each record."""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-8))
        return z_scores.max(axis=1)


class PCADetector:
    """
    PCA-based anomaly detection.
    
    Detects multivariate outliers by computing reconstruction error from PCA components.
    """
    
    def __init__(self, n_components: int = None, variance_threshold: float = 0.95):
        """
        Args:
            n_components: Number of PCA components (None = auto)
            variance_threshold: Variance to retain if n_components=None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca_ = None
        self.reconstruction_error_threshold_ = None
    
    def fit(self, X: np.ndarray, contamination: float = 0.05) -> 'PCADetector':
        """
        Fit PCA and determine reconstruction error threshold.
        
        Args:
            X: Training data
            contamination: Expected fraction of outliers
        """
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X)
        
        # Compute reconstruction errors on training data
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Set threshold as upper percentile
        self.reconstruction_error_threshold_ = np.percentile(errors, (1 - contamination) * 100)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies based on reconstruction error.
        
        Returns:
            Binary array: 1 = anomaly, 0 = normal
        """
        if self.pca_ is None:
            raise ValueError("Must call fit() first")
        
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        anomalies = (errors > self.reconstruction_error_threshold_).astype(int)
        return anomalies
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error for each record."""
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        return errors


class TranADDetectorWrapper:
    """Removed - using TranADDetector directly from tranad_model.py"""
    pass


def run_ensemble_detection(X: np.ndarray, 
                          df: pd.DataFrame,
                          active_features: List[str],
                          voting_threshold: float = 2.0,
                          include_tranad: bool = True,
                          threshold_percentile: float = 95) -> Tuple[np.ndarray, Dict]:
    """
    Run ensemble of Z-Score, PCA, and TranAD detectors.
    
    Args:
        X: Scaled feature matrix (n_samples, n_features)
        df: DataFrame for context/timestamps
        active_features: List of feature names
        voting_threshold: Detectors >= this value → anomaly
        include_tranad: Whether to include TranAD in ensemble
        threshold_percentile: TranAD threshold percentile (default: 95)
        
    Returns:
        Tuple of (ensemble_predictions, results_dict)
    """
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION - ENSEMBLE")
    print("=" * 70)
    
    # Z-Score detector
    z_detector = ZScoreDetector(threshold=3.0)
    z_detector.fit(X)
    z_pred = z_detector.predict(X)
    z_scores = z_detector.anomaly_scores(X)
    
    # PCA detector
    pca_detector = PCADetector()
    pca_detector.fit(X, contamination=0.05)
    pca_pred = pca_detector.predict(X)
    pca_scores = pca_detector.anomaly_scores(X)
    
    # TranAD detector (using official ANL implementation)
    tranad_pred = np.zeros_like(z_pred)
    tranad_scores = np.zeros_like(z_scores)
    tranad_losses = []
    tranad_batch_losses = []
    tranad_threshold = None
    
    if include_tranad and TRANAD_ANL_AVAILABLE:
        try:
            # Use official ANL TranAD with Genesis notebook parameters
            tranad_detector = TranADDetector(
                n_window=4,
                d_model=32,
                nhead=4,
                d_ff=64,
                num_layers=2,
                learning_rate=0.001,
                epochs=50,
                contamination=0.05,
                threshold_percentile=threshold_percentile,
                train_ratio=0.3
            )
            tranad_detector.fit(X)
            tranad_pred = tranad_detector.predict(X)
            tranad_scores = tranad_detector.anomaly_scores(X)
            tranad_losses = tranad_detector.training_losses
            tranad_threshold = tranad_detector.threshold  # Capture actual threshold
            print(f"  TranAD threshold captured: {tranad_threshold:.8f}")
        except Exception as e:
            print(f"⚠️  TranAD training failed: {e}")
            tranad_pred = np.zeros_like(z_pred)
            tranad_scores = np.zeros_like(z_scores)
            tranad_losses = []
            tranad_batch_losses = []
    
    # Voting ensemble (Z-Score + PCA + TranAD)
    votes = z_pred + pca_pred + tranad_pred
    ensemble_pred = (votes >= voting_threshold).astype(int)
    
    n_anomalies = ensemble_pred.sum()
    anomaly_rate = n_anomalies / len(ensemble_pred) * 100
    
    print(f"\nZ-Score Detector:")
    print(f"  Anomalies found:     {z_pred.sum()} ({z_pred.sum()/len(z_pred)*100:.2f}%)")
    print(f"  Max Z-score:         {z_scores.max():.2f}")
    
    print(f"\nPCA Detector:")
    print(f"  Anomalies found:     {pca_pred.sum()} ({pca_pred.sum()/len(pca_pred)*100:.2f}%)")
    print(f"  Max reconstruction error: {pca_scores.max():.2f}")
    
    if include_tranad and TRANAD_ANL_AVAILABLE:
        print(f"\nTranAD Detector:")
        print(f"  Anomalies found:     {tranad_pred.sum()} ({tranad_pred.sum()/len(tranad_pred)*100:.2f}%)")
        print(f"  Max reconstruction error: {tranad_scores.max():.2f}")
    
    print(f"\nEnsemble (voting threshold={voting_threshold}):")
    print(f"  Anomalies found:     {n_anomalies} ({anomaly_rate:.2f}%)")
    
    results = {
        'ensemble_predictions': ensemble_pred,
        'z_score_predictions': z_pred,
        'z_score_scores': z_scores,
        'pca_predictions': pca_pred,
        'pca_scores': pca_scores,
        'tranad_predictions': tranad_pred,
        'tranad_scores': tranad_scores,
        'tranad_losses': tranad_losses,
        'tranad_threshold': tranad_threshold,  # Add actual threshold
        'voting_scores': votes,
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate,
    }
    
    if tranad_threshold is not None:
        print(f"\nThreshold stored in results: {tranad_threshold:.8f}")
    
    return ensemble_pred, results
