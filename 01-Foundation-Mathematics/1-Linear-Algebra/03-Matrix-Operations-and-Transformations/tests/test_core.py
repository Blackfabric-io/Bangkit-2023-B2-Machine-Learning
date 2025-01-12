"""
Tests for core matrix transformation functionality.
x
"""

import numpy as np
import pytest
from src.core.base import (
    find_eigenvalues_eigenvectors,
    reflection_matrix,
    shear_matrix,
    rotation_matrix,
    scaling_matrix,
    projection_matrix,
    markov_matrix
)

def test_find_eigenvalues_eigenvectors():
    # Test identity matrix
    A = np.eye(2)
    eigenvalues, eigenvectors = find_eigenvalues_eigenvectors(A)
    assert np.allclose(eigenvalues, np.ones(2))
    assert np.allclose(np.abs(eigenvectors), np.eye(2))
    
    # Test invalid input
    with pytest.raises(TypeError):
        find_eigenvalues_eigenvectors([[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        find_eigenvalues_eigenvectors(np.array([[1, 0]]))

def test_reflection_matrix():
    # Test y-axis reflection
    R_y = reflection_matrix('y')
    assert np.allclose(R_y, np.array([[-1, 0], [0, 1]]))
    
    # Test x-axis reflection
    R_x = reflection_matrix('x')
    assert np.allclose(R_x, np.array([[1, 0], [0, -1]]))
    
    # Test invalid axis
    with pytest.raises(ValueError):
        reflection_matrix('z')

def test_shear_matrix():
    # Test x-direction shear
    k = 2.0
    S_x = shear_matrix(k, 'x')
    assert np.allclose(S_x, np.array([[1, k], [0, 1]]))
    
    # Test y-direction shear
    S_y = shear_matrix(k, 'y')
    assert np.allclose(S_y, np.array([[1, 0], [k, 1]]))
    
    # Test invalid direction
    with pytest.raises(ValueError):
        shear_matrix(k, 'z')

def test_rotation_matrix():
    # Test 90 degree rotation
    theta = np.pi/2
    R = rotation_matrix(theta)
    assert np.allclose(R, np.array([[0, -1], [1, 0]]), atol=1e-10)
    
    # Test 180 degree rotation
    theta = np.pi
    R = rotation_matrix(theta)
    assert np.allclose(R, np.array([[-1, 0], [0, -1]]), atol=1e-10)

def test_scaling_matrix():
    # Test uniform scaling
    s = 2.0
    S = scaling_matrix(s)
    assert np.allclose(S, np.array([[s, 0], [0, s]]))
    
    # Test non-uniform scaling
    sx, sy = 2.0, 3.0
    S = scaling_matrix(sx, sy)
    assert np.allclose(S, np.array([[sx, 0], [0, sy]]))

def test_projection_matrix():
    # Test x-axis projection
    P_x = projection_matrix('x')
    assert np.allclose(P_x, np.array([[1, 0], [0, 0]]))
    
    # Test y-axis projection
    P_y = projection_matrix('y')
    assert np.allclose(P_y, np.array([[0, 0], [0, 1]]))
    
    # Test invalid axis
    with pytest.raises(ValueError):
        projection_matrix('z')

def test_markov_matrix():
    # Test matrix properties
    n = 5
    P = markov_matrix(n)
    
    # Test shape
    assert P.shape == (n, n)
    
    # Test non-negative entries
    assert np.all(P >= 0)
    
    # Test column sums
    assert np.allclose(P.sum(axis=0), np.ones(n))
    
    # Test diagonal is zero
    assert np.allclose(np.diag(P), np.zeros(n))
    
    # Test invalid size
    with pytest.raises(ValueError):
        markov_matrix(0) 