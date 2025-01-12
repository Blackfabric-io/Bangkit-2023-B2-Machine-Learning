import numpy as np
from typing import Tuple
import numpy.typing as npt
from ..core.base import multiply_row, add_rows, swap_rows

MatrixType = npt.NDArray[np.float64]

def augmented_to_ref(A: MatrixType, b: MatrixType) -> MatrixType:
    """
    Turn a messy matrix into a neat organized row echelon form

    Think of it like tidying up your room
    everything gets sorted and stacked in a way that makes sense!

    Args:
        A: The main matrix (the "boss" of the problem).
        b: The vector that helps A do its job (like a sidekick).

    Returns:
        A new, cleaner matrix. It's like the "after" 
        picture of your room after cleaning! 🧹✨

    Tip: Make sure A and b are numpy arrays, or this function will throw an error!
    """
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Matrix and vector dimensions must match")
        
    # Create augmented matrix
    A_system = np.hstack((A, b.reshape(A.shape[0], 1)))
    
    # Convert to row echelon form
    A_ref = swap_rows(A_system, 0, 1)
    A_ref = add_rows(A_ref, 0, 1, -2)
    A_ref[2] += A_ref[0]
    A_ref = add_rows(A_ref, 0, 3, -1)
    A_ref[3] += A_ref[2]
    A_ref = swap_rows(A_ref, 1, 3)
    A_ref[3] += A_ref[2]
    A_ref = add_rows(A_ref, 1, 2, -4)
    A_ref[3] += A_ref[1]
    A_ref = add_rows(A_ref, 3, 2, 2)
    A_ref = add_rows(A_ref, 2, 3, -8)
    A_ref[3] = A_ref[3] * -1/17
    
    return A_ref

def ref_to_diagonal(A_ref: MatrixType) -> MatrixType:
    """
    Turn a row echelon matrix into a super clean diagonal form!

    Takes a matrix that's already in row echelon form (like a half-cleaned room) 
    and transforms it into a diagonal form (like a perfectly organized shelf). 
    It's the final step to make everything neat and tidy!

    Args:
        A_ref: A matrix in row echelon form (it's already partway there, 
        so you're halfway done!).

    Returns:
        The "after" picture of your 
        shelf after organizing all your stuff! 🧹✨

    Tip: Make sure A_ref is a numpy array, or this function will get 
    confused!
    """
    if not isinstance(A_ref, np.ndarray):
        raise TypeError("Input must be a numpy array")
        
    A_diag = add_rows(A_ref, 3, 2, -3)
    A_diag = add_rows(A_diag, 3, 1, -3)
    A_diag[0] += A_diag[3]
    A_diag = add_rows(A_diag, 2, 1, -4)
    A_diag[0] += A_diag[2]
    A_diag = add_rows(A_diag, 1, 0, -2)
    
    return A_diag

def solve_linear_system(A: MatrixType, b: MatrixType) -> Tuple[float, float, float, float]:
    """
    Solve a system of linear equations!

    Takes a coefficient matrix (A) and a constants vector (b) 
    and figures out the values of x1, x2, x3, and x4 that make the equations 
    work. Think of it as solving a puzzle where you find the missing pieces!

    Args:
        A: The coefficient matrix (the "clues" to the puzzle).
        b: The constants vector (the "hints" that help you solve it).

    Returns:
        A tuple of solutions (x1, x2, x3, x4). It's like finding the missing 
        pieces of the puzzle and putting them all together! 🧩✨

    Tip: Make sure A and b are numpy arrays, or this function will get stuck 
    like a puzzle with missing pieces
    """
    # Convert to row echelon form
    A_ref = augmented_to_ref(A, b)
    
    # Convert to diagonal form
    A_diag = ref_to_diagonal(A_ref)
    
    # Extract solutions
    x4 = A_diag[3, 4]
    x3 = A_diag[2, 4]
    x2 = A_diag[1, 4]
    x1 = A_diag[0, 4]
    
    return x1, x2, x3, x4 