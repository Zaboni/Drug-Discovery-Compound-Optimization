"""
Training Module for Drug Discovery Compound Optimization

This module contains training loops, utilities, and experiment management
for machine learning models in drug discovery.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Training functionality will be limited.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Experiment tracking will be limited.")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some metrics will not be available.")

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and hasattr(model, 'state_dict'):
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Track and compute various metrics during training."""
    
    def __init__(self, task_type: str = "regression"):
        """
        Initialize metrics tracker.
        
        Args:
            task_type: 'regression' or 'classification'
        """
        self.task_type = task_type
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, predictions: np.ndarray, targets: np.ndarray, loss: float = None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss: Batch loss (optional)
        """
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        if loss is not None:
            self.losses.append(loss)
            
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.predictions or not self.targets:
            return {}
            
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
            
        if self.task_type == "regression":
            if SKLEARN_AVAILABLE:
                metrics['mse'] = mean_squared_error(targets, predictions)
                metrics['mae'] = mean_absolute_error(targets, predictions)
                metrics['r2'] = r2_score(targets, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
        elif self.task_type == "classification":
            if SKLEARN_AVAILABLE:
                # Convert to binary predictions if needed
                if len(np.unique(predictions)) > 2:
                    pred_binary = (predictions > 0.5).astype(int)
                else:
                    pred_binary = predictions
                    
                metrics['accuracy'] = accuracy_score(targets, pred_binary)
                metrics['precision'] = precision_score(targets, pred_binary, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(targets, pred_binary, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(targets, pred_binary, average='weighted', zero_division=0)
                
        return metrics


class Trainer:
    """
    Main trainer class for drug discovery models.
    """
    
    def __init__(self, model, config_path: Optional[str] = None, use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config_path: Path to configuration file
            use_wandb: Whether to use Weights & Biases for tracking
        """
        self.model = model
        self.config = self._load_config(config_path)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Training configuration
        self.device = self._get_device()
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        self.batch_size = self.config.get('training', {}).get('batch_size', 32)
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 0.001)
        self.early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 10)
        
        # Initialize components
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb()
            
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'training': {
                    'num_epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'early_stopping_patience': 10
                }
            }
    
    def _get_device(self) -> str:
        """Get the appropriate device for training."""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        device_config = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            return device_config
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases not available")
            return
            
        wandb_config = self.config.get('wandb', {})
        
        wandb.init(
            project=wandb_config.get('project', 'drug-discovery'),
            entity=wandb_config.get('entity'),
            config=self.config
        )
        
        # Watch model if it's a PyTorch model
        if TORCH_AVAILABLE and hasattr(self.model, 'parameters'):
            wandb.watch(self.model)
    
    def prepare_data(self, X_train, y_train, X_val=None, y_val=None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders for training.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training and validation data loaders
        """
        if not TORCH_AVAILABLE:
            return X_train, X_val
            
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.config.get('hardware', {}).get('num_workers', 0)
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.config.get('hardware', {}).get('num_workers', 0)
            )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, criterion) -> Dict[str, float]:
        """Train for one epoch."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
            
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Update metrics
            self.train_metrics.update(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                loss.item()
            )
        
        return self.train_metrics.compute()
    
    def validate_epoch(self, val_loader, criterion) -> Dict[str, float]:
        """Validate for one epoch."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for validation")
            
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                # Update metrics
                self.val_metrics.update(
                    output.cpu().numpy(),
                    target.cpu().numpy(),
                    loss.item()
                )
        
        return self.val_metrics.compute()
    
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        # Handle non-PyTorch models
        if not TORCH_AVAILABLE or not hasattr(self.model, 'parameters'):
            logger.info("Training non-PyTorch model...")
            self.model.train(X_train, y_train, X_val, y_val)
            return {"message": "Non-PyTorch model trained successfully"}
        
        # Move model to device
        self.model.to(self.device)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(X_train, y_train, X_val, y_val)
        
        # Setup optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()  # Default to MSE, can be configured
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_metrics.get('loss', 0))
            history['train_metrics'].append(train_metrics)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader, criterion)
                history['val_loss'].append(val_metrics.get('loss', 0))
                history['val_metrics'].append(val_metrics)
                
                # Early stopping
                if self.early_stopping(val_metrics.get('loss', float('inf')), self.model):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}/{self.num_epochs}"
                log_msg += f" - Train Loss: {train_metrics.get('loss', 0):.4f}"
                if val_metrics:
                    log_msg += f" - Val Loss: {val_metrics.get('loss', 0):.4f}"
                logger.info(log_msg)
            
            # Wandb logging
            if self.use_wandb:
                log_dict = {f"train_{k}": v for k, v in train_metrics.items()}
                log_dict.update({f"val_{k}": v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                wandb.log(log_dict)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if not TORCH_AVAILABLE or not hasattr(self.model, 'parameters'):
            # Handle non-PyTorch models
            predictions = self.model.predict(X_test)
            metrics_tracker = MetricsTracker()
            metrics_tracker.update(predictions, y_test)
            return metrics_tracker.compute()
        
        # PyTorch model evaluation
        self.model.eval()
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_metrics = MetricsTracker()
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                test_metrics.update(
                    output.cpu().numpy(),
                    target.cpu().numpy(),
                    loss.item()
                )
        
        results = test_metrics.compute()
        logger.info(f"Test Results: {results}")
        
        if self.use_wandb:
            wandb.log({f"test_{k}": v for k, v in results.items()})
        
        return results
    
    def save_model(self, path: str):
        """Save trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if TORCH_AVAILABLE and hasattr(self.model, 'state_dict'):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_class': self.model.__class__.__name__
            }, path)
        else:
            # Handle non-PyTorch models (placeholder)
            logger.warning("Model saving not implemented for this model type")
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if TORCH_AVAILABLE and hasattr(self.model, 'load_state_dict'):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning("Model loading not implemented for this model type")


def create_data_splits(X, y, test_size: float = 0.2, val_size: float = 0.1, 
                      random_state: int = 42) -> Tuple:
    """
    Create train/validation/test splits.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("Scikit-learn is required for data splitting")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Example usage of the training module."""
    # Generate dummy data
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 100)
    y_dummy = np.random.randn(1000, 1)
    
    # Create data splits
    if SKLEARN_AVAILABLE:
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
            X_dummy, y_dummy
        )
        print(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    else:
        print("Scikit-learn not available - skipping data splitting example")
    
    # Example with metrics tracker
    metrics = MetricsTracker(task_type="regression")
    predictions = np.random.randn(100)
    targets = np.random.randn(100)
    metrics.update(predictions, targets, loss=0.5)
    results = metrics.compute()
    print(f"Example metrics: {results}")
    
    print("Training module example completed!")


if __name__ == "__main__":
    main()