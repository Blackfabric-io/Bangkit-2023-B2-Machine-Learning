"""
Unit tests for fraud detection system.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import pandas as pd
from src.core.base import IsolationForest, IsolationTree, IsolationTreeNode
from src.processors.main import FraudDetector

def create_synthetic_data(n_samples: int = 1000, n_features: int = 10,
                        contamination: float = 0.1, random_state: int = 42
                        ) -> tuple:
    """Create synthetic data for testing.
    
    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        contamination: Proportion of anomalies.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple containing:
            X: Feature array.
            y: Labels (-1 for anomalies, 1 for normal).
    """
    rng = np.random.RandomState(random_state)
    
    # Generate normal samples
    n_normal = int(n_samples * (1 - contamination))
    X_normal = rng.randn(n_normal, n_features)
    
    # Generate anomalous samples
    n_anomalies = n_samples - n_normal
    X_anomalies = rng.randn(n_anomalies, n_features) * 2 + 3
    
    # Combine samples and create labels
    X = np.vstack([X_normal, X_anomalies])
    y = np.ones(n_samples)
    y[n_normal:] = -1
    
    # Shuffle data
    indices = rng.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

def test_isolation_tree_node():
    """Test IsolationTreeNode initialization."""
    node = IsolationTreeNode()
    assert node.split_feature is None
    assert node.split_value is None
    assert node.left is None
    assert node.right is None
    assert node.size == 0
    assert node.is_external is True
    assert node.depth == 0
    
    node = IsolationTreeNode(split_feature=1, split_value=0.5,
                           size=10, depth=2, is_external=False)
    assert node.split_feature == 1
    assert node.split_value == 0.5
    assert node.size == 10
    assert node.is_external is False
    assert node.depth == 2

def test_isolation_tree():
    """Test IsolationTree functionality."""
    # Create synthetic data
    X, _ = create_synthetic_data(n_samples=100, n_features=5)
    
    # Test initialization
    tree = IsolationTree(height_limit=10)
    assert tree.height_limit == 10
    assert tree.root is None
    
    # Test fitting
    tree.fit(X, random_state=42)
    assert tree.root is not None
    assert isinstance(tree.root, IsolationTreeNode)
    
    # Test path length computation
    path_length = tree._path_length(X[0], tree.root)
    assert isinstance(path_length, (int, float))
    assert path_length >= 0

def test_isolation_forest():
    """Test IsolationForest functionality."""
    # Create synthetic data
    X, y = create_synthetic_data()
    
    # Test initialization
    forest = IsolationForest(n_estimators=10, max_samples=100,
                           contamination=0.1, random_state=42)
    assert forest.n_estimators == 10
    assert forest.max_samples == 100
    assert forest.contamination == 0.1
    assert forest.random_state == 42
    assert len(forest.trees) == 0
    assert forest.threshold_ is None
    
    # Test parameter validation
    with pytest.raises(ValueError):
        IsolationForest(n_estimators=0)
    with pytest.raises(ValueError):
        IsolationForest(max_samples=0)
    with pytest.raises(ValueError):
        IsolationForest(contamination=0)
    
    # Test fitting
    forest.fit(X)
    assert len(forest.trees) == 10
    assert forest.threshold_ is not None
    
    # Test prediction
    y_pred = forest.predict(X)
    assert y_pred.shape == (len(X),)
    assert set(y_pred) == {-1, 1}
    
    # Test scoring
    scores = forest.score_samples(X)
    assert scores.shape == (len(X),)
    assert np.all(scores >= 0) and np.all(scores <= 1)

def test_fraud_detector():
    """Test FraudDetector functionality."""
    # Create synthetic data
    X, y = create_synthetic_data()
    feature_cols = [f'v{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    
    # Save data to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / 'data.csv'
        df.to_csv(data_path, index=False)
        
        # Test initialization
        detector = FraudDetector(n_estimators=10, max_samples=100,
                               contamination=0.1, random_state=42)
        assert detector.feature_cols is None
        assert detector.scaler is None
        
        # Test training
        X_train, X_test = detector.train(str(data_path), feature_cols)
        assert detector.feature_cols == feature_cols
        assert detector.scaler is not None
        assert X_train.shape[1] == X_test.shape[1] == len(feature_cols)
        
        # Test evaluation
        results = detector.evaluate(X_test)
        assert 'predictions' in results
        assert 'scores' in results
        assert 'threshold' in results
        
        # Test prediction
        predictions = detector.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert set(predictions) == {-1, 1}
        
        # Test model saving and loading
        model_dir = Path(temp_dir) / 'model'
        detector.save_model(str(model_dir))
        
        loaded_detector = FraudDetector.load_model(str(model_dir))
        assert loaded_detector.feature_cols == detector.feature_cols
        
        # Test predictions are consistent
        loaded_predictions = loaded_detector.predict(X_test)
        assert np.array_equal(loaded_predictions, predictions)

def test_fraud_detector_errors():
    """Test FraudDetector error handling."""
    detector = FraudDetector()
    
    # Test prediction without training
    with pytest.raises(ValueError):
        detector.predict(np.random.randn(10, 5))
    
    # Test evaluation without training
    with pytest.raises(ValueError):
        detector.evaluate(np.random.randn(10, 5))
    
    # Test feature importance without training
    with pytest.raises(ValueError):
        detector.analyze_feature_importance('dummy_path')
    
    # Test model saving without training
    with pytest.raises(ValueError):
        detector.save_model('dummy_path')
    
    # Test model loading with invalid path
    with pytest.raises(FileNotFoundError):
        FraudDetector.load_model('invalid_path') 