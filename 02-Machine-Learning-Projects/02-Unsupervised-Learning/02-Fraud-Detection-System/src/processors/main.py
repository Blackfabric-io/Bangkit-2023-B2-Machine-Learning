"""
Main processing logic for fraud detection system.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import numpy.typing as npt
from src.core.base import IsolationForest
from src.utils.helpers import (
    load_data,
    preprocess_data,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_anomaly_scores,
    plot_feature_importance,
    compute_metrics,
    save_results
)

ArrayType = npt.NDArray[np.float64]

class FraudDetector:
    """Fraud detection system using Isolation Forest."""
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.1, random_state: Optional[int] = None):
        """Initialize the fraud detector.
        
        Args:
            n_estimators: Number of isolation trees.
            max_samples: Number of samples to draw for each tree.
            contamination: Expected proportion of fraudulent transactions.
            random_state: Random state for reproducibility.
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )
        self.feature_cols: Optional[List[str]] = None
        self.scaler = None
        
    def analyze_data(self, data_path: str, feature_cols: List[str]) -> None:
        """Analyze transaction data.
        
        Args:
            data_path: Path to the data file.
            feature_cols: List of feature column names.
        """
        # Store feature columns
        self.feature_cols = feature_cols
        
        # Load and analyze data
        df = load_data(data_path)
        
        print("Feature Distributions:")
        plot_feature_distributions(df, feature_cols)
        
        print("\nFeature Correlations:")
        plot_correlation_matrix(df, feature_cols)
    
    def train(self, data_path: str, feature_cols: List[str],
              test_size: float = 0.2, random_state: int = 42
              ) -> Tuple[ArrayType, ArrayType]:
        """Train the fraud detection model.
        
        Args:
            data_path: Path to the data file.
            feature_cols: List of feature column names.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.
            
        Returns:
            Tuple containing training and test data.
            
        Raises:
            ValueError: If feature_cols is not set.
        """
        # Store feature columns
        self.feature_cols = feature_cols
        
        # Load and preprocess data
        df = load_data(data_path)
        X_train, X_test, self.scaler = preprocess_data(
            df, feature_cols, test_size, random_state
        )
        
        # Train model
        print("Training Isolation Forest model...")
        self.model.fit(X_train)
        
        # Plot anomaly scores
        scores = self.model.score_samples(X_train)
        plot_anomaly_scores(scores, self.model.threshold_,
                          "Training Data Anomaly Scores")
        
        return X_train, X_test
    
    def evaluate(self, X: ArrayType, y_true: Optional[ArrayType] = None
                ) -> Dict[str, Any]:
        """Evaluate model on test data.
        
        Args:
            X: Test data.
            y_true: True labels if available.
            
        Returns:
            Dictionary containing evaluation results.
            
        Raises:
            ValueError: If model is not trained.
        """
        if not hasattr(self.model, 'threshold_'):
            raise ValueError("Model must be trained before evaluation")
            
        # Get predictions and scores
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # Compute metrics if true labels are available
        metrics = None
        if y_true is not None:
            metrics = compute_metrics(y_true, predictions)
            print("\nModel Performance:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.3f}")
        
        # Plot anomaly scores
        plot_anomaly_scores(scores, self.model.threshold_,
                          "Test Data Anomaly Scores")
        
        return {
            'predictions': predictions,
            'scores': scores,
            'metrics': metrics,
            'threshold': self.model.threshold_
        }
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Predict if transactions are fraudulent.
        
        Args:
            X: Transaction features.
            
        Returns:
            Predictions (-1 for fraud, 1 for normal).
            
        Raises:
            ValueError: If model is not trained.
        """
        if not hasattr(self.model, 'threshold_'):
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict(X)
    
    def analyze_feature_importance(self, data_path: str) -> None:
        """Analyze feature importance based on model predictions.
        
        Args:
            data_path: Path to the data file.
            
        Raises:
            ValueError: If model is not trained or feature_cols not set.
        """
        if not hasattr(self.model, 'threshold_'):
            raise ValueError("Model must be trained before analysis")
        if self.feature_cols is None:
            raise ValueError("feature_cols must be set during training")
            
        # Load data and get predictions
        df = load_data(data_path)
        X = self.scaler.transform(df[self.feature_cols].values)
        predictions = self.predict(X)
        
        # Plot feature importance
        plot_feature_importance(df, self.feature_cols, predictions)
        
        # Print summary statistics
        print("\nFeature Statistics by Class:")
        for feature in self.feature_cols:
            print(f"\n{feature}:")
            normal = df[predictions == 1][feature]
            fraud = df[predictions == -1][feature]
            print(f"Normal - Mean: {normal.mean():.2f}, Std: {normal.std():.2f}")
            print(f"Fraud  - Mean: {fraud.mean():.2f}, Std: {fraud.std():.2f}")
    
    def save_model(self, model_dir: str) -> None:
        """Save trained model and scaler.
        
        Args:
            model_dir: Directory to save model files.
            
        Raises:
            ValueError: If model is not trained.
        """
        if not hasattr(self.model, 'threshold_'):
            raise ValueError("Model must be trained before saving")
            
        import joblib
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_dir / 'isolation_forest.joblib')
        
        # Save scaler if available
        if self.scaler is not None:
            joblib.dump(self.scaler, model_dir / 'scaler.joblib')
        
        # Save feature columns
        if self.feature_cols is not None:
            with open(model_dir / 'feature_cols.txt', 'w') as f:
                f.write('\n'.join(self.feature_cols))
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'FraudDetector':
        """Load trained model and scaler.
        
        Args:
            model_dir: Directory containing model files.
            
        Returns:
            Loaded FraudDetector instance.
            
        Raises:
            FileNotFoundError: If model files don't exist.
        """
        import joblib
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / 'isolation_forest.joblib'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Create instance and load model
        instance = cls()
        instance.model = joblib.load(model_path)
        
        # Load scaler if available
        scaler_path = model_dir / 'scaler.joblib'
        if scaler_path.exists():
            instance.scaler = joblib.load(scaler_path)
        
        # Load feature columns if available
        feature_cols_path = model_dir / 'feature_cols.txt'
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                instance.feature_cols = f.read().splitlines()
        
        return instance 