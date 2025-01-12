"""
Core functionality for isolation forest anomaly detection.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import numpy.typing as npt
from dataclasses import dataclass

ArrayType = npt.NDArray[np.float64]

@dataclass
class IsolationTreeNode:
    """Node in an isolation tree."""
    split_feature: Optional[int] = None
    split_value: Optional[float] = None
    left: Optional['IsolationTreeNode'] = None
    right: Optional['IsolationTreeNode'] = None
    size: int = 0
    is_external: bool = True
    depth: int = 0

class IsolationTree:
    """Single isolation tree implementation."""
    
    def __init__(self, height_limit: int):
        """Initialize the isolation tree.
        
        Args:
            height_limit: Maximum height of the tree.
        """
        self.height_limit = height_limit
        self.root: Optional[IsolationTreeNode] = None
        
    def _select_split_feature(self, X: ArrayType, rng: np.random.RandomState) -> int:
        """Randomly select a feature to split on.
        
        Args:
            X: Data of shape (n_samples, n_features).
            rng: Random number generator.
            
        Returns:
            Selected feature index.
        """
        n_features = X.shape[1]
        return rng.randint(n_features)
    
    def _select_split_value(self, X: ArrayType, feature: int,
                          rng: np.random.RandomState) -> float:
        """Select a random split value between min and max of the feature.
        
        Args:
            X: Data of shape (n_samples, n_features).
            feature: Feature index to split on.
            rng: Random number generator.
            
        Returns:
            Split value.
        """
        min_val = X[:, feature].min()
        max_val = X[:, feature].max()
        return rng.uniform(min_val, max_val)
    
    def _build_tree(self, X: ArrayType, current_height: int,
                   rng: np.random.RandomState) -> IsolationTreeNode:
        """Recursively build the isolation tree.
        
        Args:
            X: Data of shape (n_samples, n_features).
            current_height: Current height in the tree.
            rng: Random number generator.
            
        Returns:
            Root node of the tree.
        """
        n_samples = X.shape[0]
        node = IsolationTreeNode(size=n_samples, depth=current_height)
        
        # Check termination conditions
        if current_height >= self.height_limit or n_samples <= 1:
            node.is_external = True
            return node
        
        # Select split feature and value
        node.split_feature = self._select_split_feature(X, rng)
        node.split_value = self._select_split_value(X, node.split_feature, rng)
        
        # Split data
        left_mask = X[:, node.split_feature] < node.split_value
        X_left = X[left_mask]
        X_right = X[~left_mask]
        
        # Check if split was successful
        if len(X_left) == 0 or len(X_right) == 0:
            node.is_external = True
            return node
        
        # Build subtrees
        node.is_external = False
        node.left = self._build_tree(X_left, current_height + 1, rng)
        node.right = self._build_tree(X_right, current_height + 1, rng)
        
        return node
    
    def fit(self, X: ArrayType, random_state: Optional[int] = None) -> None:
        """Fit the isolation tree.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            random_state: Random state for reproducibility.
        """
        rng = np.random.RandomState(random_state)
        self.root = self._build_tree(X, current_height=0, rng=rng)
    
    def _path_length(self, x: ArrayType, node: IsolationTreeNode) -> float:
        """Compute path length for a single sample.
        
        Args:
            x: Single sample of shape (n_features,).
            node: Current tree node.
            
        Returns:
            Path length.
        """
        if node.is_external:
            # Adjust for unsuccessful termination
            if node.size <= 1:
                return node.depth
            else:
                return node.depth + self._c(node.size)
        
        # Traverse tree
        if x[node.split_feature] < node.split_value:
            return self._path_length(x, node.left)
        else:
            return self._path_length(x, node.right)
    
    @staticmethod
    def _c(n: int) -> float:
        """Compute average path length of unsuccessful search in BST.
        
        Args:
            n: Number of nodes.
            
        Returns:
            Average path length.
        """
        if n <= 1:
            return 0
        
        h = np.log(n - 1) + 0.5772156649  # Euler's constant
        return 2 * h - (2 * (n - 1) / n)

class IsolationForest:
    """Isolation forest for anomaly detection.
    
    This class implements the isolation forest algorithm for detecting anomalies
    in data. The algorithm isolates anomalies by recursively partitioning the
    data space and computing the average path length to each point.
    """
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.1, random_state: Optional[int] = None):
        """Initialize the isolation forest.
        
        Args:
            n_estimators: Number of isolation trees.
            max_samples: Number of samples to draw for each tree.
            contamination: Expected proportion of anomalies.
            random_state: Random state for reproducibility.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
            
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        self.trees: List[IsolationTree] = []
        self.threshold_: Optional[float] = None
    
    def fit(self, X: ArrayType) -> 'IsolationForest':
        """Fit the isolation forest.
        
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
            
        n_samples = X.shape[0]
        
        # Determine subsample size
        if self.max_samples > n_samples:
            self.max_samples = n_samples
        
        # Compute height limit
        height_limit = int(np.ceil(np.log2(self.max_samples)))
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Build trees
        self.trees = []
        for i in range(self.n_estimators):
            # Subsample data
            indices = rng.choice(n_samples, self.max_samples, replace=False)
            X_sub = X[indices]
            
            # Build tree
            tree = IsolationTree(height_limit=height_limit)
            tree.fit(X_sub, random_state=rng.randint(np.iinfo(np.int32).max))
            self.trees.append(tree)
        
        # Compute threshold
        scores = self.score_samples(X)
        self.threshold_ = np.percentile(scores, 100 * self.contamination)
        
        return self
    
    def score_samples(self, X: ArrayType) -> ArrayType:
        """Compute anomaly scores for samples.
        
        Args:
            X: Data of shape (n_samples, n_features).
            
        Returns:
            Anomaly scores of shape (n_samples,).
            
        Raises:
            ValueError: If model is not fitted or input is invalid.
        """
        if len(self.trees) == 0:
            raise ValueError("Model must be fitted before scoring samples")
            
        # Compute average path length for each sample
        scores = np.zeros(X.shape[0])
        for tree in self.trees:
            for i, x in enumerate(X):
                path_length = tree._path_length(x, tree.root)
                scores[i] += path_length
        
        scores /= len(self.trees)
        
        # Normalize scores
        n_samples = X.shape[0]
        scores = 2 ** (-scores / IsolationTree._c(n_samples))
        
        return scores
    
    def predict(self, X: ArrayType) -> ArrayType:
        """Predict if samples are anomalies.
        
        Args:
            X: Data of shape (n_samples, n_features).
            
        Returns:
            Predictions of shape (n_samples,). -1 for anomalies, 1 for normal.
            
        Raises:
            ValueError: If model is not fitted or input is invalid.
        """
        if self.threshold_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, 1, -1) 