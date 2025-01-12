"""
Utility functions for data preprocessing, visualization, and cluster analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ArrayType = npt.NDArray[np.float64]

def load_data(filepath: str) -> pd.DataFrame:
    """Load customer data from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame containing customer data.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")

def preprocess_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ArrayType, StandardScaler]:
    """Preprocess data for clustering.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        
    Returns:
        Tuple containing:
            X: Scaled feature array.
            scaler: Fitted StandardScaler instance.
    """
    # Extract features
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def plot_feature_distributions(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Plot feature distributions.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    """
    n_features = len(feature_cols)
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(15, 5 * n_rows))
    for idx, feature in enumerate(feature_cols, 1):
        plt.subplot(n_rows, 2, idx)
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Plot correlation matrix of features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    """
    corr = df[feature_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()

def plot_elbow_curve(X: ArrayType, max_clusters: int = 10,
                    random_state: int = 42) -> None:
    """Plot elbow curve for determining optimal number of clusters.
    
    Args:
        X: Feature array of shape (n_samples, n_features).
        max_clusters: Maximum number of clusters to try.
        random_state: Random state for reproducibility.
    """
    from src.core.base import KMeans
    inertias = []
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_clusters_2d(X: ArrayType, labels: ArrayType, centroids: ArrayType,
                    feature_names: List[str], pca: bool = True) -> None:
    """Plot clusters in 2D space.
    
    Args:
        X: Feature array of shape (n_samples, n_features).
        labels: Cluster labels of shape (n_samples,).
        centroids: Cluster centroids of shape (n_clusters, n_features).
        feature_names: List of feature names.
        pca: Whether to use PCA for dimensionality reduction.
    """
    if X.shape[1] > 2 and pca:
        # Apply PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        centroids_2d = pca.transform(centroids)
        
        # Get explained variance
        explained_var = pca.explained_variance_ratio_
        xlabel = f'First Principal Component ({explained_var[0]:.1%} variance)'
        ylabel = f'Second Principal Component ({explained_var[1]:.1%} variance)'
    else:
        # Use first two features
        X_2d = X[:, :2]
        centroids_2d = centroids[:, :2]
        xlabel = feature_names[0]
        ylabel = feature_names[1]
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Customer Segments')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def analyze_clusters(df: pd.DataFrame, labels: ArrayType,
                    feature_cols: List[str]) -> pd.DataFrame:
    """Analyze cluster characteristics.
    
    Args:
        df: Input DataFrame.
        labels: Cluster labels.
        feature_cols: List of feature column names.
        
    Returns:
        DataFrame containing cluster statistics.
    """
    # Add cluster labels to DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Compute cluster statistics
    cluster_stats = []
    for cluster in range(labels.max() + 1):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        stats = {
            'Cluster Size': len(cluster_data),
            'Cluster Percentage': len(cluster_data) / len(df) * 100
        }
        
        # Add mean and std for each feature
        for feature in feature_cols:
            stats[f'{feature} (Mean)'] = cluster_data[feature].mean()
            stats[f'{feature} (Std)'] = cluster_data[feature].std()
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)

def plot_cluster_profiles(cluster_stats: pd.DataFrame,
                         feature_cols: List[str]) -> None:
    """Plot cluster profiles using radar charts.
    
    Args:
        cluster_stats: DataFrame containing cluster statistics.
        feature_cols: List of feature column names.
    """
    # Prepare data for radar chart
    mean_cols = [f'{col} (Mean)' for col in feature_cols]
    values = cluster_stats[mean_cols].values
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False)
    values = np.concatenate((values, values[:, [0]]), axis=1)
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for i in range(len(cluster_stats)):
        ax.plot(angles, values[i], 'o-', linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values[i], alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols)
    ax.set_title('Cluster Profiles')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show() 