"""
Core functionality for matrix operations and transformations.
x
"""

import numpy as np
from typing import Dict, Tuple, Optional
import numpy.typing as npt

MatrixType = npt.NDArray[np.float64]

def find_eigenvalues_eigenvectors(A: MatrixType) -> Tuple[MatrixType, MatrixType]:
    """Find eigenvalues and eigenvectors of a matrix.
    
    This function calculates the eigenvalues and eigenvectors of a given matrix using
    NumPy's linear algebra package.
    
    Args:
        A: Input matrix of shape (n, n) for which to find eigenvalues and eigenvectors.
        
    Returns:
        A tuple containing:
            eigenvalues: Array of eigenvalues.
            eigenvectors: Matrix where each column is an eigenvector.
            
    Raises:
        ValueError: If matrix A is not square.
        TypeError: If input is not a numpy array.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
        
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

def reflection_matrix(axis: str = 'y') -> MatrixType:
    """Create a reflection matrix about specified axis.
    
    Args:
        axis: Axis about which to reflect ('x' or 'y').
        
    Returns:
        2x2 reflection matrix.
        
    Raises:
        ValueError: If axis is not 'x' or 'y'.
    """
    if axis not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
        
    if axis == 'y':
        return np.array([[-1, 0], [0, 1]])
    else:
        return np.array([[1, 0], [0, -1]])

def shear_matrix(k: float, direction: str = 'x') -> MatrixType:
    """Create a shear matrix.
    
    Args:
        k: Shear factor.
        direction: Direction of shear ('x' or 'y').
        
    Returns:
        2x2 shear matrix.
        
    Raises:
        ValueError: If direction is not 'x' or 'y'.
    """
    if direction not in ['x', 'y']:
        raise ValueError("Direction must be 'x' or 'y'")
        
    if direction == 'x':
        return np.array([[1, k], [0, 1]])
    else:
        return np.array([[1, 0], [k, 1]])

def rotation_matrix(theta: float) -> MatrixType:
    """Create a rotation matrix for given angle.
    
    Args:
        theta: Rotation angle in radians.
        
    Returns:
        2x2 rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def scaling_matrix(sx: float, sy: Optional[float] = None) -> MatrixType:
    """Create a scaling matrix.
    
    Args:
        sx: Scale factor in x direction.
        sy: Scale factor in y direction. If None, uses sx.
        
    Returns:
        2x2 scaling matrix.
    """
    sy = sy if sy is not None else sx
    return np.array([[sx, 0], [0, sy]])

def projection_matrix(axis: str = 'x') -> MatrixType:
    """Create a projection matrix onto specified axis.
    
    Args:
        axis: Axis onto which to project ('x' or 'y').
        
    Returns:
        2x2 projection matrix.
        
    Raises:
        ValueError: If axis is not 'x' or 'y'.
    """
    if axis not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
        
    if axis == 'x':
        return np.array([[1, 0], [0, 0]])
    else:
        return np.array([[0, 0], [0, 1]])

def markov_matrix(n: int) -> MatrixType:
    """Create a random Markov matrix.
    
    Args:
        n: Size of the matrix.
        
    Returns:
        nxn Markov matrix where each column sums to 1.
        
    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError("Matrix size must be positive")
        
    # Generate random positive numbers
    M = np.random.rand(n, n)
    # Set diagonal to 0
    np.fill_diagonal(M, 0)
    # Normalize columns to sum to 1
    M = M / M.sum(axis=0)
    
    return M 