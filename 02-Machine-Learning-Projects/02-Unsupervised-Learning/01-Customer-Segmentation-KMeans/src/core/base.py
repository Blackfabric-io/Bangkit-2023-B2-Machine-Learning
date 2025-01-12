"""
Core functionality for K-means clustering algorithm.
"""

import numpy as np
from typing import Tuple, Optional, List
import numpy.typing as npt
from enum import Enum

ArrayType = npt.NDArray[np.float64]

class InitMethod(Enum):
    """Initialization methods for K-means clustering."""
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "kmeans++"

class KMeans:
    """K-means clustering implementation.
    
    This class implements the K-means clustering algorithm with multiple
    initialization strategies and early stopping based on convergence.
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 300,
                 init_method: InitMethod = InitMethod.KMEANS_PLUS_PLUS,
                 random_state: Optional[int] = None,
                 tol: float = 1e-4):
        """Initialize the model.
        
        Args:
            n_clusters: Number of clusters.
            max_iters: Maximum number of iterations.
            init_method: Method for initializing centroids.
            random_state: Random state for reproducibility.
            tol: Tolerance for declaring convergence.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if max_iters < 1:
            raise ValueError("max_iters must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
            
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_method = init_method
        self.random_state = random_state
        self.tol = tol
        
        self.centroids: Optional[ArrayType] = None
        self.labels_: Optional[ArrayType] = None
        self.inertia_: Optional[float] = None
        self.n_iters_: Optional[int] = None
        
    def _init_random(self, X: ArrayType) -> ArrayType:
        """Initialize centroids randomly.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            
        Returns:
            Initial centroids.
        """
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n_samples)[:self.n_clusters]
        return X[indices].copy()
    
    def _init_kmeans_plus_plus(self, X: ArrayType) -> ArrayType:
        """Initialize centroids using k-means++ method.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            
        Returns:
            Initial centroids.
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # Choose first centroid randomly
        centroids = [X[rng.randint(n_samples)]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances to closest centroid
            distances = np.min([
                np.sum((X - centroid) ** 2, axis=1)
                for centroid in centroids
            ], axis=0)
            
            # Choose next centroid with probability proportional to distance squared
            probs = distances / distances.sum()
            next_centroid = X[rng.choice(n_samples, p=probs)]
            centroids.append(next_centroid)
        
        return np.array(centroids)
    
    def _compute_distances(self, X: ArrayType, centroids: ArrayType) -> ArrayType:
        """Compute distances between samples and centroids.
        
        Args:
            X: Data points of shape (n_samples, n_features).
            centroids: Centroids of shape (n_clusters, n_features).
            
        Returns:
            Distances of shape (n_samples, n_clusters).
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for i in range(n_clusters):
            distances[:, i] = np.sum((X - centroids[i]) ** 2, axis=1)
            
        return distances
    
    def fit(self, X: ArrayType) -> 'KMeans':
        """Fit K-means clustering model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            
        Returns:
            self: Fitted model.
            
        Raises:
            ValueError: If input is invalid.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be greater than n_clusters")
            
        # Initialize centroids
        if self.init_method == InitMethod.RANDOM:
            self.centroids = self._init_random(X)
        else:
            self.centroids = self._init_kmeans_plus_plus(X)
        
        # Initialize variables
        prev_centroids = None
        self.n_iters_ = 0
        
        # Main loop
        for iteration in range(self.max_iters):
            # Assign points to clusters
            distances = self._compute_distances(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            prev_centroids = self.centroids.copy()
            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # Check convergence
            if prev_centroids is not None:
                diff = np.max(np.abs(self.centroids - prev_centroids))
                if diff < self.tol:
                    break
            
            self.n_iters_ = iteration + 1
        
        # Compute inertia (sum of squared distances to closest centroid)
        self.inertia_ = np.sum(np.min(self._compute_distances(X, self.centroids), axis=1))
        
        return self
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Predict cluster labels for new data.
        
        Args:
            X: New data of shape (n_samples, n_features).
            
        Returns:
            Predicted cluster labels.
            
        Raises:
            ValueError: If model is not fitted or input is invalid.
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError("X must have same number of features as training data")
            
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def transform(self, X: ArrayType) -> ArrayType:
        """Transform X to a cluster-distance space.
        
        Args:
            X: New data of shape (n_samples, n_features).
            
        Returns:
            Distances to all clusters of shape (n_samples, n_clusters).
            
        Raises:
            ValueError: If model is not fitted or input is invalid.
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before transforming data")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError("X must have same number of features as training data")
            
        return self._compute_distances(X, self.centroids) 