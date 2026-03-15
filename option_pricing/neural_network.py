"""
Neural Network Options Pricing Model.

Provides deep learning models (PyTorch) for options pricing with capabilities
for both call and put options, including data preprocessing and model evaluation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class OptionPricingNN(nn.Module):
    """
    Neural Network for options pricing.
    
    Architecture: Input -> Hidden Layer 1 (64 neurons) -> Hidden Layer 2 (64 neurons) 
                  -> Output (1 neuron with Softplus activation)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (e.g., moneyness, strike, IV, time_to_exp)
        hidden_dim : int
            Number of hidden units
        dropout_rate : float
            Dropout rate for regularization
        """
        super(OptionPricingNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive output (option prices are always positive)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        return self.network(x)


class NeuralNetworkPricer:
    """
    Neural network options pricing model with training and inference.
    """
    
    def __init__(self, input_features: list = None):
        """
        Initialize pricer.
        
        Parameters:
        -----------
        input_features : list
            Names of input features for documentation
        """
        self.input_features = input_features or ['moneyness', 'strike', 'iv', 'time_to_exp']
        self.model = None
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        self.best_loss = float('inf')
        
        logger.info(f"Initialized NeuralNetworkPricer with features: {self.input_features}")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare and split data for training.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        test_size : float
            Fraction of data for testing
        random_state : int
            Random seed
            
        Returns:
        --------
        Tuple
            (X_train, X_test, y_train, y_test) - scaled tensors on device
        """
        try:
            logger.info(f"Preparing data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Validate data
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                raise ValueError("Data contains NaN values")
            
            if np.any(np.isinf(X)) or np.any(np.isinf(y)):
                raise ValueError("Data contains infinite values")
            
            # Detect and handle outliers using IQR method
            X, y = self._remove_outliers(X, y)
            
            # Train-test split
            n_train = int(len(X) * (1 - test_size))
            idx = np.random.RandomState(random_state).permutation(len(X))
            
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Scale targets (optional, but helps training)
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)
            y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)
            
            logger.info(f"Data prepared: {len(train_idx)} train, {len(test_idx)} test")
            return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    @staticmethod
    def _remove_outliers(X: np.ndarray, y: np.ndarray, iqr_multiplier: float = 1.5) -> Tuple:
        """
        Remove outliers using interquartile range method.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Targets
        iqr_multiplier : float
            IQR multiplier for outlier threshold
            
        Returns:
        --------
        Tuple
            (X_clean, y_clean) - arrays without outliers
        """
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        
        n_removed = np.sum(~mask)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} outliers")
        
        return X[mask], y[mask]
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
             X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
             epochs: int = 200, batch_size: int = 32, lr: float = 0.001,
             early_stopping_patience: int = 20) -> Dict:
        """
        Train the neural network.
        
        Parameters:
        -----------
        X_train : torch.Tensor
            Training features
        y_train : torch.Tensor
            Training targets
        X_val : torch.Tensor, optional
            Validation features
        y_val : torch.Tensor, optional
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        early_stopping_patience : int
            Epochs without improvement before stopping
            
        Returns:
        --------
        Dict
            Training history with losses
        """
        try:
            logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
            
            # Initialize model
            input_dim = X_train.shape[1]
            self.model = OptionPricingNN(input_dim).to(device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=10, verbose=False)
            
            # Data loader
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            history = {'train_loss': [], 'val_loss': []}
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(dataloader)
                history['train_loss'].append(train_loss)
                
                # Validation
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_predictions = self.model(X_val)
                        val_loss = criterion(val_predictions, y_val).item()
                        history['val_loss'].append(val_loss)
                    
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
                    
                    if (epoch + 1) % 20 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    if (epoch + 1) % 20 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
            
            self.history = history
            logger.info(f"Training completed. Best validation loss: {self.best_loss:.6f}")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Predicted option prices
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            self.model.eval()
            
            # Scale input
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            
            # Predict
            with torch.no_grad():
                y_pred_scaled = self.model(X_tensor).cpu().numpy()
            
            # Inverse scale
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            
            return y_pred.ravel()
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True targets
            
        Returns:
        --------
        Dict
            Metrics (MSE, MAE, R2, RMSE)
        """
        try:
            y_pred = self.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            logger.info(f"Model Evaluation:\n  MSE: {mse:.6f}\n  MAE: {mae:.6f}\n  RMSE: {rmse:.6f}\n  R²: {r2:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'scaler_y': self.scaler_y,
                'history': self.history
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str, input_dim: int):
        """Load trained model from disk."""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            self.model = OptionPricingNN(input_dim).to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.scaler_y = checkpoint['scaler_y']
            self.history = checkpoint.get('history')
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def create_synthetic_training_data(n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic options data for training/testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (X, y) - features and targets
    """
    np.random.seed(seed)
    
    # Generate features
    moneyness = np.random.uniform(-0.5, 0.5, n_samples)  # log(S/K)
    strikes = np.random.uniform(80, 120, n_samples)
    iv = np.random.uniform(0.1, 0.5, n_samples)
    time_to_exp = np.random.uniform(0.01, 1.0, n_samples)
    
    X = np.column_stack([moneyness, strikes, iv, time_to_exp])
    
    # Generate synthetic targets (approximate Black-Scholes)
    # This is simplified - real implementation would use actual BS formula
    S = strikes * np.exp(moneyness)
    K = strikes
    
    # Approximate call price using BS formula
    from scipy.stats import norm
    d1 = (np.log(S / K) + 0.05 * time_to_exp) / (iv * np.sqrt(time_to_exp) + 1e-6)
    d2 = d1 - iv * np.sqrt(time_to_exp)
    
    call_prices = S * norm.cdf(d1) - K * np.exp(-0.05 * time_to_exp) * norm.cdf(d2)
    y = np.maximum(call_prices, 0.01)  # Ensure positive prices
    
    # Add small noise
    y += np.random.normal(0, 0.1, n_samples)
    y = np.maximum(y, 0.01)
    
    logger.info(f"Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


if __name__ == "__main__":
    print("=" * 70)
    print("Neural Network Options Pricing - Examples")
    print("=" * 70)
    
    # Example: Train and evaluate model
    print("\n[Example 1] Training Neural Network Model")
    print("-" * 70)
    
    try:
        # Generate synthetic data
        X, y = create_synthetic_training_data(n_samples=5000)
        
        # Create pricer
        pricer = NeuralNetworkPricer()
        
        # Prepare data
        X_train, X_test, y_train, y_test = pricer.prepare_data(X, y, test_size=0.2)
        
        # Train model
        history = pricer.train(X_train, y_train, X_test, y_test, 
                             epochs=100, batch_size=32, lr=0.001)
        
        # Evaluate on test set
        X_test_np = X[int(len(X)*0.8):]
        y_test_np = y[int(len(y)*0.8):]
        
        metrics = pricer.evaluate(X_test_np, y_test_np)
        
        print(f"\n✓ Model Training Complete!")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        if history['val_loss']:
            print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
        print(f"\n  Test Set Metrics:")
        for metric, value in metrics.items():
            print(f"    {metric.upper()}: {value:.6f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
