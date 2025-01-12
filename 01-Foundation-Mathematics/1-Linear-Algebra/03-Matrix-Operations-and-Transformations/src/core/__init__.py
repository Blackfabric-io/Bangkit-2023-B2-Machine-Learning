"""
🌟✨ Superpowers for Playing with Matrices! 🧮🎮

This module is like a magic wand for working with matrices. It helps you do cool stuff 
like spinning, stretching, flipping, and even predicting the future (well, kind of)! 

What's inside:
- find_eigenvalues_eigenvectors: Finds the "secret directions" and "scaling factors" in your matrix.
- reflection_matrix: Flips your matrix like a mirror! 🪞
- shear_matrix: Tilts your matrix sideways (like a funhouse mirror!).
- rotation_matrix: Spins your matrix around like a DJ turntable! 🎧
- scaling_matrix: Makes your matrix bigger or smaller (like zooming in or out!).
- projection_matrix: Squishes your matrix onto a flat surface (like a shadow!).
- markov_matrix: Predicts what happens next (like a fortune teller for data! 🔮).

Use these tools to make your matrix adventures more fun and exciting! 🚀🎲
"""

from .base import (
    find_eigenvalues_eigenvectors,
    reflection_matrix,
    shear_matrix,
    rotation_matrix,
    scaling_matrix,
    projection_matrix,
    markov_matrix
)

__all__ = [
    'find_eigenvalues_eigenvectors',
    'reflection_matrix',
    'shear_matrix',
    'rotation_matrix',
    'scaling_matrix',
    'projection_matrix',
    'markov_matrix'
]
