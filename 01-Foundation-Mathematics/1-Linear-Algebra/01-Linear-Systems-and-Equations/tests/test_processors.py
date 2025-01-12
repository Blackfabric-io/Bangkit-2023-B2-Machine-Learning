import numpy as np
import pytest
from src.processors.main import augmented_to_ref, ref_to_diagonal, solve_linear_system

def test_augmented_to_ref():
    A = np.array([
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]
    ], dtype=float)
    
    b = np.array([6, 3, 14, 8], dtype=float)
    
    result = augmented_to_ref(A, b)
    
    # Check dimensions
    assert result.shape == (4, 5)
    
    # Check row echelon form
    for i in range(1, 4):
        assert np.all(result[i, :i] == 0)

def test_ref_to_diagonal():
    A_ref = np.array([
        [1, 2, -1, -1, 3],
        [0, 1, 4, 3, 22],
        [0, 0, 1, 3, 7],
        [0, 0, 0, 1, 1]
    ], dtype=float)
    
    result = ref_to_diagonal(A_ref)
    
    # Check diagonal
    for i in range(4):
        for j in range(4):
            if i != j:
                assert abs(result[i, j]) < 1e-10

def test_solve_linear_system():
    A = np.array([
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]
    ], dtype=float)
    
    b = np.array([6, 3, 14, 8], dtype=float)
    
    x1, x2, x3, x4 = solve_linear_system(A, b)
    
    # Check solution matches
    assert abs(x1 - 2) < 1e-10
    assert abs(x2 - 3) < 1e-10
    assert abs(x3 - 4) < 1e-10
    assert abs(x4 - 1) < 1e-10
    
    result = A @ np.array([x1, x2, x3, x4])
    assert np.allclose(result, b) 