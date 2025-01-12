import numpy as np
from typing import Union, Tuple
import numpy.typing as npt

MatrixType = npt.NDArray[np.float64]

def multiply_row(matrix: MatrixType, row_num: int, row_num_multiple: float) -> MatrixType:
    """
    Multiply a row in a matrix by a number
    like giving it a boost 🚀

    This function takes a matrix, picks a specific row 
    and multiplies all the numbers in that row by a scalar. 
    The result is a new matrix with the updated row.

    Args:
        matrix: The input matrix (your starting team _of numbers_).
        row_num: The index of the row to multiply (0-based, so the first row is 0).
        row_num_multiple: The number to multiply the row by (the power-up).

    Returns:
        A new matrix with the row multiplied by the scalar. It's like your team 
        after the power-up — stronger and ready to go! 💪✨

    Tip: Make sure the row_num is within the matrix's size, or this function 
    will get confused
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix must be a numpy array")
    if row_num < 0 or row_num >= matrix.shape[0]:
        raise ValueError("Invalid row number")
    if row_num_multiple == 0:
        raise ValueError("Row multiplier cannot be zero")
        
    matrix_new = matrix.copy()
    matrix_new[row_num] = matrix_new[row_num] * row_num_multiple
    return matrix_new

def add_rows(matrix: MatrixType, row_num_1: int, row_num_2: int, row_num_1_multiple: float) -> MatrixType:
    """
    Add a boosted row to another row—like teaming up! 🤝💥

    This function takes a matrix, picks two rows (like two team members), boosts 
    the first row by multiplying it with a number (giving it extra power), and 
    then adds it to the second row. The result is a new matrix with the updated 
    row—like a combo move!

    Args:
        matrix: The input matrix (your team of numbers).
        row_num_1: The index of the row to boost and add (0-based, so the first row is 0).
        row_num_2: The index of the row to add the boosted row to (the teammate getting the combo).
        row_num_1_multiple: The number to multiply row_num_1 by before adding 
                            (the power boost!).

    Returns:
        A new matrix with the combo move applied. It's like your team after 
        pulling off an awesome combo — stronger and ready to win! 🏆

    Tip: Make sure row_num_1 and row_num_2 are within the matrix's size, or 
    this function will get stuck like a failed combo
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix must be a numpy array")
    if row_num_1 < 0 or row_num_1 >= matrix.shape[0] or row_num_2 < 0 or row_num_2 >= matrix.shape[0]:
        raise ValueError("Invalid row number")
        
    matrix_new = matrix.copy()
    matrix_new[row_num_2] += matrix_new[row_num_1] * row_num_1_multiple
    return matrix_new

def swap_rows(matrix: MatrixType, row_num_1: int, row_num_2: int) -> MatrixType:
    """
    Swap two rows in a matrix—like swapping players on a team! 🔄

    This function takes a matrix and swaps two rows (like swapping two players 
    on a team). The result is a new matrix where the rows have traded places
    just like a coach switching players to try a new strategy!

    Args:
        matrix: The input matrix (your team of numbers).
        row_num_1: The index of the first row to swap (0-based, so the first row is 0).
        row_num_2: The index of the second row to swap (the other player's position).

    Returns:
        A new matrix with the rows swapped. It's like your team after the 
        player swap—same team, new lineup! 🏀

    Tip: Make sure row_num_1 and row_num_2 are within the matrix's size, or 
    this function will get confused like a coach trying to swap players don't exist!
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix must be a numpy array")
    if row_num_1 < 0 or row_num_1 >= matrix.shape[0] or row_num_2 < 0 or row_num_2 >= matrix.shape[0]:
        raise ValueError("Invalid row number")
        
    matrix_new = matrix.copy()
    matrix_new[row_num_1] = matrix_new[row_num_1] + matrix_new[row_num_2]
    matrix_new[row_num_2] = matrix_new[row_num_1] - matrix_new[row_num_2]
    matrix_new[row_num_1] = matrix_new[row_num_1] - matrix_new[row_num_2]
    return matrix_new 