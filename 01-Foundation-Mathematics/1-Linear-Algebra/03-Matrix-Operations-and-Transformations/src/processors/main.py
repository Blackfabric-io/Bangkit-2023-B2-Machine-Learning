"""
Main processing module for matrix transformations and analysis.
x
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import numpy.typing as npt
from ..core.base import (
    find_eigenvalues_eigenvectors,
    reflection_matrix,
    shear_matrix,
    rotation_matrix,
    scaling_matrix,
    projection_matrix,
    markov_matrix
)
from ..utils.helpers import (
    plot_transformation,
    plot_eigenvectors,
    simulate_webpage_navigation
)

MatrixType = npt.NDArray[np.float64]

def analyze_transformation(T: MatrixType) -> Dict[str, MatrixType]:
    """Analyze a transformation matrix.
    
    This function performs comprehensive analysis of a transformation matrix,
    including finding eigenvalues, eigenvectors, and determinant.
    
    Args:
        T: Input transformation matrix.
        
    Returns:
        Dictionary containing:
            eigenvalues: Array of eigenvalues.
            eigenvectors: Matrix of eigenvectors.
            determinant: Determinant of the matrix.
            
    Raises:
        ValueError: If T is not square.
    """
    if T.shape[0] != T.shape[1]:
        raise ValueError("Matrix must be square")
        
    eigenvalues, eigenvectors = find_eigenvalues_eigenvectors(T)
    determinant = np.linalg.det(T)
    
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "determinant": determinant
    }

def compose_transformations(transformations: List[MatrixType]) -> MatrixType:
    """Compose multiple transformations.
    
    Args:
        transformations: List of transformation matrices to compose.
        
    Returns:
        Composed transformation matrix.
        
    Raises:
        ValueError: If matrices have incompatible dimensions.
    """
    if not transformations:
        raise ValueError("List of transformations cannot be empty")
        
    result = transformations[0]
    for T in transformations[1:]:
        if result.shape[1] != T.shape[0]:
            raise ValueError("Incompatible matrix dimensions")
        result = T @ result
        
    return result

def analyze_webpage_ranking(P: MatrixType, X0: MatrixType, 
                          steps: int = 20) -> Dict[str, MatrixType]:
    """Analyze webpage ranking using Markov chain.
    
    Args:
        P: Transition probability matrix (Markov matrix).
        X0: Initial state vector.
        steps: Number of steps to simulate.
        
    Returns:
        Dictionary containing:
            final_state: Final probability distribution.
            steady_state: Steady state distribution.
            eigenvalues: Eigenvalues of transition matrix.
            
    Raises:
        ValueError: If P is not a valid Markov matrix.
    """
    # Verify P is a valid Markov matrix
    if not np.allclose(P.sum(axis=0), np.ones(P.shape[1])):
        raise ValueError("P must be a Markov matrix (columns sum to 1)")
    if np.any(P < 0):
        raise ValueError("P must contain only non-negative entries")
        
    # Simulate navigation
    final_state, eigenvalues = simulate_webpage_navigation(P, X0, steps)
    
    # Find steady state (eigenvector corresponding to eigenvalue 1)
    _, eigenvectors = np.linalg.eig(P)
    steady_state = eigenvectors[:, 0]
    steady_state = steady_state / steady_state.sum()  # Normalize
    
    return {
        "final_state": final_state,
        "steady_state": steady_state.reshape(-1, 1),
        "eigenvalues": eigenvalues
    }

def create_transformation(transform_type: str, **kwargs) -> MatrixType:
    """Create a transformation matrix of specified type.
    
    Args:
        transform_type: Type of transformation ('reflection', 'shear', 'rotation', 
                       'scaling', 'projection').
        **kwargs: Arguments specific to the transformation type.
        
    Returns:
        Transformation matrix.
        
    Raises:
        ValueError: If transform_type is not recognized.
    """
    transform_map = {
        'reflection': reflection_matrix,
        'shear': shear_matrix,
        'rotation': rotation_matrix,
        'scaling': scaling_matrix,
        'projection': projection_matrix
    }
    
    if transform_type not in transform_map:
        raise ValueError(f"Unknown transformation type: {transform_type}")
        
    return transform_map[transform_type](**kwargs) 