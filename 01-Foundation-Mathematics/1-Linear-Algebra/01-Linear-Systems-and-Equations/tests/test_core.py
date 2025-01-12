import numpy as np
import pytest
from src.core.base import multiply_row, add_rows, swap_rows

def test_multiply_row():
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    result = multiply_row(matrix, 0, 2)
    expected = np.array([[2, 4], [3, 4]], dtype=float)
    assert np.array_equal(result, expected)
    
    with pytest.raises(ValueError):
        multiply_row(matrix, -1, 2)
    
    with pytest.raises(ValueError):
        multiply_row(matrix, 0, 0)

def test_add_rows():
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    result = add_rows(matrix, 0, 1, 2)
    expected = np.array([[1, 2], [5, 8]], dtype=float)
    assert np.array_equal(result, expected)
    
    with pytest.raises(ValueError):
        add_rows(matrix, -1, 1, 2)

def test_swap_rows():
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    result = swap_rows(matrix, 0, 1)
    expected = np.array([[3, 4], [1, 2]], dtype=float)
    assert np.array_equal(result, expected)
    
    with pytest.raises(ValueError):
        swap_rows(matrix, -1, 1) 