"""
Main processing logic for university admission classification.
"""

from typing import Dict, Any, List, Optional
import numpy.typing as npt
import numpy as np
from src.core.base import LogisticRegression
from src.utils.helpers import (
    load_data,
    preprocess_data,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_decision_boundary,
    plot_cost_history,
    evaluate_model
)

ArrayType = npt.NDArray[np.float64]

class AdmissionClassifier:
    """University admission classifier using logistic regression."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """Initialize the classifier.
        
        Args:
            learning_rate: Learning rate for gradient descent.
            n_iterations: Number of training iterations.
        """
        self.model = LogisticRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        self.feature_cols: Optional[List[str]] = None
        self.target_col: Optional[str] = None
        
    def train(self, data_path: str, feature_cols: List[str], target_col: str,
              test_size: float = 0.2, random_state: int = 42,
              plot_training: bool = True) -> Dict[str, Any]:
        """Train the admission classifier.
        
        Args:
            data_path: Path to the training data.
            feature_cols: List of feature column names.
            target_col: Name of target column.
            test_size: Proportion of dataset to include in test split.
            random_state: Random state for reproducibility.
            plot_training: Whether to plot training visualizations.
            
        Returns:
            Dictionary containing training results and evaluation metrics.
        """
        # Store column names
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Load and preprocess data
        df = load_data(data_path)
        X_train, X_test, y_train, y_test = preprocess_data(
            df, feature_cols, target_col, test_size, random_state
        )
        
        # Plot data distributions if requested
        if plot_training:
            plot_feature_distributions(df, feature_cols, target_col)
            plot_correlation_matrix(df, feature_cols)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Plot training progress if requested
        if plot_training:
            plot_cost_history(self.model.cost_history)
            plot_decision_boundary(X_train, y_train, self.model)
        
        # Evaluate model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self._compute_feature_importance()
        }
    
    def predict(self, X: ArrayType, return_proba: bool = False) -> ArrayType:
        """Make predictions on new data.
        
        Args:
            X: Input features.
            return_proba: Whether to return probabilities instead of class labels.
            
        Returns:
            Predicted class labels or probabilities.
            
        Raises:
            ValueError: If model is not trained.
        """
        if self.model.weights is None:
            raise ValueError("Model must be trained before making predictions")
            
        if return_proba:
            return self.model.predict_proba(X)
        return self.model.predict(X)
    
    def _compute_feature_importance(self) -> Dict[str, float]:
        """Compute feature importance based on model weights.
        
        Returns:
            Dictionary mapping feature names to their importance scores.
            
        Raises:
            ValueError: If model is not trained or feature names are not set.
        """
        if self.model.weights is None:
            raise ValueError("Model must be trained before computing feature importance")
        if self.feature_cols is None:
            raise ValueError("Feature columns must be set during training")
            
        # Use absolute weights as importance scores
        importance_scores = np.abs(self.model.weights)
        
        # Normalize scores
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return dict(zip(self.feature_cols, importance_scores)) 