"""
Helper functions for visualization and data handling.
x
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import numpy.typing as npt

MatrixType = npt.NDArray[np.float64]

def plot_transformation(T: MatrixType, v1: Optional[MatrixType] = None, 
                       v2: Optional[MatrixType] = None, title: str = "") -> None:
    """Plot the effect of a transformation matrix on vectors.
    
    Args:
        T: 2x2 transformation matrix.
        v1: First vector to transform (2x1). If None, uses [1, 0].
        v2: Second vector to transform (2x1). If None, uses [0, 1].
        title: Title for the plot.
        
    Raises:
        ValueError: If T is not a 2x2 matrix.
    """
    if T.shape != (2, 2):
        raise ValueError("Transformation matrix must be 2x2")
        
    # Default to standard basis vectors if none provided
    if v1 is None:
        v1 = np.array([[1], [0]])
    if v2 is None:
        v2 = np.array([[0], [1]])
        
    # Colors for original and transformed vectors
    color_original = "#129cab"
    color_transformed = "#cc8933"
    
    # Create plot
    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-6, 6))
    ax.set_yticks(np.arange(-6, 6))
    plt.axis([-6, 6, -6, 6])
    
    # Plot original vectors
    plt.quiver([0, 0], [0, 0], [v1[0], v2[0]], [v1[1], v2[1]], 
               color=color_original, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, v2[0], v1[0]+v2[0], v1[0]], [0, v2[1], v1[1]+v2[1], v1[1]], 
             color=color_original)
    
    # Add labels for original vectors
    v1_sgn = 0.4 * np.sign(v1)
    v2_sgn = 0.4 * np.sign(v2)
    ax.text(v1[0]-0.2+v1_sgn[0], v1[1]-0.2+v1_sgn[1], f'$v_1$', 
            fontsize=14, color=color_original)
    ax.text(v2[0]-0.2+v2_sgn[0], v2[1]-0.2+v2_sgn[1], f'$v_2$', 
            fontsize=14, color=color_original)
    
    # Apply transformation
    v1_transformed = T @ v1
    v2_transformed = T @ v2
    
    # Plot transformed vectors
    plt.quiver([0, 0], [0, 0], [v1_transformed[0], v2_transformed[0]], 
               [v1_transformed[1], v2_transformed[1]], 
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, v2_transformed[0], v1_transformed[0]+v2_transformed[0], v1_transformed[0]],
             [0, v2_transformed[1], v1_transformed[1]+v2_transformed[1], v1_transformed[1]],
             color=color_transformed)
    
    # Add labels for transformed vectors
    v1_t_sgn = 0.4 * np.sign(v1_transformed)
    v2_t_sgn = 0.4 * np.sign(v2_transformed)
    ax.text(v1_transformed[0]-0.2+v1_t_sgn[0], v1_transformed[1]-0.2+v1_t_sgn[1], 
            f'$T(v_1)$', fontsize=14, color=color_transformed)
    ax.text(v2_transformed[0]-0.2+v2_t_sgn[0], v2_transformed[1]-0.2+v2_t_sgn[1], 
            f'$T(v_2)$', fontsize=14, color=color_transformed)
    
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

def plot_eigenvectors(A: MatrixType, eigenvalues: MatrixType, 
                     eigenvectors: MatrixType) -> None:
    """Plot matrix transformation with eigenvectors.
    
    Args:
        A: 2x2 transformation matrix.
        eigenvalues: Array of eigenvalues.
        eigenvectors: Matrix where each column is an eigenvector.
        
    Raises:
        ValueError: If A is not a 2x2 matrix.
    """
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")
        
    # Plot the transformation
    plot_transformation(A, eigenvectors[:, 0:1], eigenvectors[:, 1:2], 
                       title="Transformation with Eigenvectors")
    
    # Print eigenvalue information
    print("\nEigenvalues and their effects:")
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"λ_{i+1} = {eigenval:.2f}")
        print(f"v_{i+1} = [{eigenvec[0]:.2f}, {eigenvec[1]:.2f}]")
        if abs(eigenval - 1) < 1e-10:
            print("This eigenvector is unchanged by the transformation")
        elif eigenval > 0:
            print(f"This eigenvector is stretched by a factor of {eigenval:.2f}")
        elif eigenval < 0:
            print(f"This eigenvector is stretched by a factor of {abs(eigenval):.2f} and reversed")
        print()

def simulate_webpage_navigation(P: MatrixType, X0: MatrixType, 
                              steps: int = 20) -> Tuple[MatrixType, MatrixType]:
    """Simulate webpage navigation using Markov chain.
    
    Args:
        P: nxn Markov matrix where each column sums to 1.
        X0: Initial state vector (nx1).
        steps: Number of steps to simulate.
        
    Returns:
        Tuple containing:
            X_final: Final state vector.
            eigenvalues: Eigenvalues of P.
            
    Raises:
        ValueError: If P is not square or X0 shape doesn't match P.
    """
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if P.shape[0] != X0.shape[0]:
        raise ValueError("X0 must have same number of rows as P")
        
    # Simulate steps
    X = X0.copy()
    for _ in range(steps):
        X = P @ X
        
    # Find steady state using eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(P)
    
    return X, eigenvalues 