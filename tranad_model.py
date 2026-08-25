"""
TranAD: Official ANL Implementation Wrapper

Uses the official ANL TranAD from anldq.deep_learning.models
Follows the same training approach as GenesisExample_v0.ipynb
"""

import sys
from pathlib import Path
import numpy as np

# Check for torch
TORCH_AVAILABLE = False
torch = None
optim = None
try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  PyTorch not available: {e}")

# Check for ANL modules
ANLDQ_AVAILABLE = False
TranAD = None
TranADConfig = None
try:
    src_path = str(Path(__file__).parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from anldq.deep_learning.models import TranAD
    from anldq.configs.model_config import TranADConfig
    ANLDQ_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ANL package not available: {e}")

# TranAD is only available if both torch and anldq are present
TRANAD_ANL_AVAILABLE = TORCH_AVAILABLE and ANLDQ_AVAILABLE


class TranADDetector:
    """
    Wrapper for official ANL TranAD model.
    
    Follows the exact training approach from GenesisExample_v0.ipynb:
    - 30% train / 70% test split
    - Sliding window reconstruction
    - Two-phase model (x1, x2 outputs)
    - Training losses tracked per epoch
    """
    
    def __init__(self, n_window=4, d_model=32, nhead=4, d_ff=64, num_layers=2,
                 learning_rate=0.001, epochs=50, contamination=0.05,
                 threshold_percentile=95,
                 train_ratio=0.3, device=None):
        """
        Args:
            n_window: Sliding window size
            d_model: Model dimension
            nhead: Number of attention heads
            d_ff: Feedforward dimension
            num_layers: Number of transformer layers
            learning_rate: Adam learning rate
            epochs: Training epochs
            contamination: Expected anomaly ratio (kept for compatibility)
            threshold_percentile: Percentile for threshold calculation (default: 95)
            train_ratio: Fraction of data for training baseline
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        if not TRANAD_ANL_AVAILABLE:
            raise ImportError("ANL TranAD not available. Install anldq package.")
        
        self.n_window = n_window
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile
        self.train_ratio = train_ratio
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.optimizer = None
        self.training_losses = []
        self.threshold = None
        self.reconstruction_errors = None
    
    def fit(self, X: np.ndarray) -> 'TranADDetector':
        """
        Train TranAD model using 30% of data for baseline.
        
        Args:
            X: (n_samples, n_features) array
        """
        print("Training ANL TranAD model...")
        
        # Split data: 30% train (baseline), 70% test
        n_train = int(len(X) * self.train_ratio)
        X_train = X[:n_train]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_full_tensor = torch.FloatTensor(X).to(self.device)
        
        # Configure and create model
        config = TranADConfig(
            n_window=self.n_window,
            d_model=self.d_model,
            nhead=self.nhead,
            d_ff=self.d_ff,
            num_layers=self.num_layers
        )
        
        self.model = TranAD(input_dims=X.shape[1], arch_config=config)
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"✓ Model created ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
        print(f"  Data split: {len(X_train)} train, {len(X) - len(X_train)} test")
        print(f"  Window size: {self.n_window}, Model dim: {self.d_model}")
        print(f"  Device: {self.device}")
        
        # Training loop (exactly as in notebook)
        self.training_losses = []
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            count = 0
            
            for i in range(self.n_window, len(X_train)):
                # Get sliding window and target
                x_window = X_train_tensor[i - self.n_window:i].unsqueeze(0)  # (1, T, F)
                x_target = X_train_tensor[i].unsqueeze(0).unsqueeze(1)  # (1, 1, F)
                
                # Forward pass (two outputs: x1, x2)
                x1, x2 = self.model(x_window, x_target)
                
                # Reconstruction loss on x2
                loss = torch.mean((x2.squeeze() - X_train_tensor[i]) ** 2)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
            
            avg_loss = epoch_loss / max(count, 1)
            self.training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:2d}/{self.epochs}: loss = {avg_loss:.8f}")
        
        print(f"✓ Training complete")
        
        # Compute reconstruction errors on full dataset
        print(f"  Computing reconstruction errors...")
        
        self.model.eval()
        reconstruction_errors = np.zeros(len(X) - self.n_window)
        
        with torch.no_grad():
            for i in range(self.n_window, len(X)):
                x_window = X_full_tensor[i - self.n_window:i].unsqueeze(0)
                x_target = X_full_tensor[i].unsqueeze(0).unsqueeze(1)
                x1, x2 = self.model(x_window, x_target)
                mse = torch.mean((x2.squeeze() - X_full_tensor[i]) ** 2).item()
                reconstruction_errors[i - self.n_window] = mse
        
        self.reconstruction_errors = reconstruction_errors
        
        # Set threshold from training period (percentile-based approach)
        train_errors = reconstruction_errors[:n_train - self.n_window]
        self.threshold = np.percentile(train_errors, self.threshold_percentile)
        
        print(f"✓ Threshold ({self.threshold_percentile}th percentile): {self.threshold:.8f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: (n_samples, n_features) array
            
        Returns:
            Binary predictions (1 = anomaly)
        """
        if self.reconstruction_errors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Reconstruction errors are computed for indices [n_window, len(X))
        # Prepend zeros for first n_window samples
        predictions = np.zeros(len(X), dtype=int)
        predictions[self.n_window:] = (self.reconstruction_errors > self.threshold).astype(int)
        
        return predictions
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction error scores.
        
        Args:
            X: (n_samples, n_features) array
            
        Returns:
            Reconstruction error scores
        """
        if self.reconstruction_errors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Pad first n_window samples with zeros
        scores = np.zeros(len(X))
        scores[self.n_window:] = self.reconstruction_errors
        
        return scores
