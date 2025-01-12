"""
Unit tests for K-means clustering and customer segmentation.
"""

import numpy as np
import pytest
from src.core.base import KMeans, InitMethod
from src.processors.main import CustomerSegmentation

def test_kmeans_initialization():
    """Test KMeans model initialization."""
    # Test default parameters
    model = KMeans()
    assert model.n_clusters == 3
    assert model.max_iters == 300
    assert model.init_method == InitMethod.KMEANS_PLUS_PLUS
    assert model.random_state is None
    assert model.tol == 1e-4
    assert model.centroids is None
    assert model.labels_ is None
    assert model.inertia_ is None
    assert model.n_iters_ is None
    
    # Test custom parameters
    model = KMeans(n_clusters=5, max_iters=100, init_method=InitMethod.RANDOM,
                  random_state=42, tol=1e-5)
    assert model.n_clusters == 5
    assert model.max_iters == 100
    assert model.init_method == InitMethod.RANDOM
    assert model.random_state == 42
    assert model.tol == 1e-5

def test_kmeans_parameter_validation():
    """Test KMeans parameter validation."""
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    
    with pytest.raises(ValueError):
        KMeans(max_iters=0)
    
    with pytest.raises(ValueError):
        KMeans(tol=-1)

def test_kmeans_random_init():
    """Test random centroid initialization."""
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [9, 8]])
    model = KMeans(n_clusters=2, init_method=InitMethod.RANDOM, random_state=42)
    
    centroids = model._init_random(X)
    assert centroids.shape == (2, 2)
    assert np.all(np.isin(centroids, X))

def test_kmeans_plus_plus_init():
    """Test k-means++ centroid initialization."""
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [9, 8]])
    model = KMeans(n_clusters=2, init_method=InitMethod.KMEANS_PLUS_PLUS,
                  random_state=42)
    
    centroids = model._init_kmeans_plus_plus(X)
    assert centroids.shape == (2, 2)
    assert np.all(np.isin(centroids, X))

def test_kmeans_fit_predict():
    """Test KMeans fitting and prediction."""
    # Generate synthetic data with clear clusters
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 2),
        np.random.randn(50, 2) + [4, 4]
    ])
    
    # Fit model
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    
    # Check results
    assert model.centroids is not None
    assert model.labels_ is not None
    assert model.inertia_ is not None
    assert model.n_iters_ is not None
    assert model.centroids.shape == (2, 2)
    assert len(model.labels_) == len(X)
    
    # Check predictions
    y_pred = model.predict(X)
    assert np.array_equal(y_pred, model.labels_)
    
    # Check cluster assignment makes sense
    assert len(np.unique(y_pred)) == 2
    assert np.abs(np.mean(y_pred[:50]) - np.mean(y_pred[50:])) > 0.8

def test_kmeans_transform():
    """Test KMeans transform method."""
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [9, 8]])
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    
    distances = model.transform(X)
    assert distances.shape == (len(X), 2)
    assert np.all(distances >= 0)

def test_customer_segmentation():
    """Test CustomerSegmentation class."""
    # Initialize segmentation
    segmentation = CustomerSegmentation(n_clusters=3, random_state=42)
    assert segmentation.model is not None
    assert segmentation.feature_cols is None
    assert segmentation.scaler is None
    
    # Test prediction without fitting
    X = np.random.randn(10, 5)
    with pytest.raises(ValueError):
        segmentation.predict(X)
    
    # Test segment profile without fitting
    with pytest.raises(ValueError):
        segmentation.get_segment_profile(0)
    
    # Test invalid segment ID
    segmentation.model.centroids = np.random.randn(3, 5)
    with pytest.raises(ValueError):
        segmentation.get_segment_profile(-1)
    with pytest.raises(ValueError):
        segmentation.get_segment_profile(3) 