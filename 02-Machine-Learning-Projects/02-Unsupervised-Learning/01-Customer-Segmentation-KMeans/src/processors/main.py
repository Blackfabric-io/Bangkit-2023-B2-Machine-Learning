"""
Main processing logic for customer segmentation.
"""

from typing import Dict, Any, List, Optional
import numpy.typing as npt
import numpy as np
import pandas as pd
from src.core.base import KMeans, InitMethod
from src.utils.helpers import (
    load_data,
    preprocess_data,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_elbow_curve,
    plot_clusters_2d,
    analyze_clusters,
    plot_cluster_profiles
)

ArrayType = npt.NDArray[np.float64]

class CustomerSegmentation:
    """Customer segmentation using K-means clustering."""
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 300,
                 init_method: InitMethod = InitMethod.KMEANS_PLUS_PLUS,
                 random_state: Optional[int] = None):
        """Initialize the segmentation model.
        
        Args:
            n_clusters: Number of customer segments.
            max_iters: Maximum number of iterations.
            init_method: Method for initializing centroids.
            random_state: Random state for reproducibility.
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iters=max_iters,
            init_method=init_method,
            random_state=random_state
        )
        self.feature_cols: Optional[List[str]] = None
        self.scaler: Optional[Any] = None
        
    def analyze_data(self, data_path: str, feature_cols: List[str]) -> None:
        """Analyze customer data.
        
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
        
        # Preprocess data
        X, self.scaler = preprocess_data(df, feature_cols)
        
        print("\nElbow Curve for Optimal k:")
        plot_elbow_curve(X, max_clusters=10, random_state=self.model.random_state)
    
    def fit(self, data_path: str, feature_cols: List[str]) -> Dict[str, Any]:
        """Fit segmentation model.
        
        Args:
            data_path: Path to the data file.
            feature_cols: List of feature column names.
            
        Returns:
            Dictionary containing segmentation results.
            
        Raises:
            ValueError: If feature_cols is not set.
        """
        # Store feature columns
        self.feature_cols = feature_cols
        
        # Load and preprocess data
        df = load_data(data_path)
        X, self.scaler = preprocess_data(df, feature_cols)
        
        # Fit model
        print("Fitting K-means model...")
        self.model.fit(X)
        
        # Get cluster assignments
        labels = self.model.labels_
        
        # Visualize clusters
        print("\nCluster Visualization:")
        plot_clusters_2d(X, labels, self.model.centroids, feature_cols)
        
        # Analyze clusters
        print("\nCluster Analysis:")
        cluster_stats = analyze_clusters(df, labels, feature_cols)
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        print("\nCluster Profiles:")
        plot_cluster_profiles(cluster_stats, feature_cols)
        
        return {
            'labels': labels,
            'centroids': self.model.centroids,
            'inertia': self.model.inertia_,
            'n_iterations': self.model.n_iters_,
            'cluster_stats': cluster_stats
        }
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Predict segments for new customers.
        
        Args:
            X: Customer features.
            
        Returns:
            Predicted segment labels.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if self.model.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.model.predict(X)
    
    def get_segment_profile(self, segment_id: int) -> pd.Series:
        """Get profile for a specific customer segment.
        
        Args:
            segment_id: ID of the segment to profile.
            
        Returns:
            Series containing segment characteristics.
            
        Raises:
            ValueError: If model is not fitted or segment_id is invalid.
        """
        if self.model.centroids is None:
            raise ValueError("Model must be fitted before getting segment profiles")
        if segment_id < 0 or segment_id >= self.model.n_clusters:
            raise ValueError(f"segment_id must be between 0 and {self.model.n_clusters - 1}")
        if self.feature_cols is None:
            raise ValueError("feature_cols must be set during fitting")
            
        # Get centroid for the segment
        centroid = self.model.centroids[segment_id]
        
        # If scaler is available, inverse transform the centroid
        if self.scaler is not None:
            centroid = self.scaler.inverse_transform(centroid.reshape(1, -1)).ravel()
        
        return pd.Series(centroid, index=self.feature_cols) 