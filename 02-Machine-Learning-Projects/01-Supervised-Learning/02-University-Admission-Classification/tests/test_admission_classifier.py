"""
Unit tests for university admission classifier.
"""

import numpy as np
import pytest
from src.core.base import LogisticRegression
from src.processors.main import AdmissionClassifier

def test_logistic_regression_initialization():
    """Test logistic regression model initialization."""
    model = LogisticRegression()
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.weights is None
    assert model.bias is None
    assert model.cost_history == []
    
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    assert model.learning_rate == 0.1
    assert model.n_iterations == 500

def test_sigmoid():
    """Test sigmoid function computation."""
    model = LogisticRegression()
    
    # Test on scalar
    assert model.sigmoid(0) == 0.5
    
    # Test on array
    x = np.array([-1, 0, 1])
    result = model.sigmoid(x)
    expected = np.array([0.26894142, 0.5, 0.73105858])
    np.testing.assert_array_almost_equal(result, expected)

def test_logistic_regression_training():
    """Test logistic regression model training."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    w_true = np.array([1.5, -0.8])
    b_true = 0.5
    
    # Generate labels using true parameters
    z = np.dot(X, w_true) + b_true
    y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    # Check if parameters are learned
    assert model.weights is not None
    assert model.bias is not None
    assert len(model.cost_history) > 0
    
    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # Check accuracy
    accuracy = np.mean(y_pred == y)
    assert accuracy > 0.8

def test_admission_classifier_initialization():
    """Test admission classifier initialization."""
    classifier = AdmissionClassifier()
    assert classifier.model is not None
    assert classifier.feature_cols is None
    assert classifier.target_col is None
    
    classifier = AdmissionClassifier(learning_rate=0.1, n_iterations=500)
    assert classifier.model.learning_rate == 0.1
    assert classifier.model.n_iterations == 500

def test_admission_classifier_feature_importance():
    """Test feature importance computation."""
    classifier = AdmissionClassifier()
    
    # Set dummy weights
    classifier.model.weights = np.array([1.0, 2.0, 3.0])
    classifier.feature_cols = ['f1', 'f2', 'f3']
    
    importance = classifier._compute_feature_importance()
    
    assert isinstance(importance, dict)
    assert len(importance) == 3
    assert list(importance.keys()) == ['f1', 'f2', 'f3']
    assert abs(sum(importance.values()) - 1.0) < 1e-6

def test_input_validation():
    """Test input validation and error handling."""
    classifier = AdmissionClassifier()
    
    # Test prediction without training
    X = np.random.randn(10, 5)
    with pytest.raises(ValueError):
        classifier.predict(X)
    
    # Test feature importance without training
    with pytest.raises(ValueError):
        classifier._compute_feature_importance()
    
    # Test feature importance without feature names
    classifier.model.weights = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        classifier._compute_feature_importance() 