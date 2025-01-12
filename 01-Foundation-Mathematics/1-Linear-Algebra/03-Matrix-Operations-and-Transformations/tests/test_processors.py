"""
🧪🔍 Let's Test Those Matrix Magic Tricks! 🎩✨

This module is like a science lab for testing all the cool matrix transformations 
and webpage ranking stuff. It makes sure everything works perfectly, like checking 
if your magic wand is casting the right spells!

What's being tested:
- analyze_transformation: Checks if the matrix transformations are doing their job.
- compose_transformations: Makes sure combining transformations works like a charm.
- analyze_webpage_ranking: Tests if the webpage ranking predictions are on point.
- create_transformation: Verifies if creating new transformations is smooth and easy.

Let's make sure everything is working as expected! 🚀🧪
"""

import numpy as np
import pytest
from src.processors.main import (
    analyze_transformation,
    compose_transformations,
    analyze_webpage_ranking,
    create_transformation
)


def test_analyze_transformation():
    """
    🧪🔍 Let's Test the Matrix Analyzer! 🎯✨

    This test checks if the `analyze_transformation` function works like a pro. 
    It makes sure the function can handle a simple identity matrix (which is like 
    a "do nothing" matrix) and also catches errors when given bad input.

    What's being tested:
    - Does it find the right eigenvalues and eigenvectors? (The "secret directions" and "scaling factors"!)
    - Does it calculate the determinant correctly? (A number that tells you if the matrix is special or not!)
    - Does it raise an error when given bad input? (Like a matrix that's not square!)

    Let's make sure this function is as sharp as a ninja! 🥷🎲
    """
    # Test identity matrix (the "do nothing" matrix)
    T = np.eye(2)
    result = analyze_transformation(T)
    
    assert "eigenvalues" in result  # Check if it found the eigenvalues
    assert "eigenvectors" in result  # Check if it found the eigenvectors
    assert "determinant" in result  # Check if it calculated the determinant
    assert np.allclose(result["eigenvalues"], np.ones(2))  # Eigenvalues should be 1 for identity matrix
    assert np.isclose(result["determinant"], 1.0)  # Determinant should be 1 for identity matrix
    
    # Test invalid input (a matrix that's not square)
    with pytest.raises(ValueError):
        analyze_transformation(np.array([[1, 0]]))  
        # Should raise an error for bad input

def test_compose_transformations():
    """
    🧪🔍 Let's Test Combining Matrix Magic! 🎩✨

    This test checks if the `compose_transformations` function can combine 
    transformations like a pro. It makes sure that combining two rotations 
    works correctly and also catches errors when given bad input.

    What's being tested:
    - Does it combine two rotations correctly? (Like spinning twice and ending up in the right spot!)
    - Does it raise an error when given an empty list? (No transformations to combine? That's a no-no!)
    - Does it raise an error when given incompatible matrices? (Like trying to mix a 1x1 matrix with a 2x2 matrix!)

    Let's make sure this function is as smooth as a DJ mixing tracks! 🎧🎲
    """
    # Test composition of rotations (spinning twice!)
    R1 = create_transformation('rotation', theta=np.pi/4)  # Rotate by 45 degrees
    R2 = create_transformation('rotation', theta=np.pi/4)  # Rotate by another 45 degrees
    R_composed = compose_transformations([R1, R2])  # Combine the two rotations
    R_expected = create_transformation('rotation', theta=np.pi/2)  # Should be the same as rotating by 90 degrees
    
    assert np.allclose(R_composed, R_expected, atol=1e-10)  # Check if the result is correct
    
    # Test empty list (no transformations to combine!)
    with pytest.raises(ValueError):
        compose_transformations([])  # Should raise an error for an empty list
    
    # Test incompatible matrices (mixing different sizes!)
    with pytest.raises(ValueError):
        compose_transformations([np.array([[1]]), np.array([[1, 0], [0, 1]])])  # Should raise an error for incompatible matrices


def test_analyze_webpage_ranking():
    """
    🧪🔍 Let's Test the Webpage Ranking Analyzer! 🌐📊

    This test checks if the `analyze_webpage_ranking` function can predict 
    how people might move around a website. It makes sure the function works 
    with a valid Markov matrix (a special kind of matrix for predicting stuff) 
    and catches errors when given bad input.

    What's being tested:
    - Does it calculate the final state, steady state, and eigenvalues correctly? 
      (The "final state" is where people end up, the "steady state" is where they 
      stay forever, and "eigenvalues" are like the secret sauce of the matrix!)
    - Does it check the shape of the results? (Making sure everything fits together!)
    - Does it make sure the steady state adds up to 1 and has no negative values? 
      (Because probabilities can't be negative or add up to more than 100%!)
    - Does it raise an error when given an invalid Markov matrix? (Like a matrix 
      where the columns don't add up to 1!)

    Let's make sure this function is as reliable as your favorite search engine! 🔍🚀
    """
    # Create valid Markov matrix (a matrix for predicting webpage visits)
    n = 3
    P = np.array([
        [0, 0.5, 0.3],
        [0.6, 0, 0.3],
        [0.4, 0.5, 0.4]
    ])
    X0 = np.array([[1], [0], [0]])  # Starting state (everyone on the first page)
    
    result = analyze_webpage_ranking(P, X0)  # Analyze the webpage ranking
    
    assert "final_state" in result  # Check if it found the final state
    assert "steady_state" in result  # Check if it found the steady state
    assert "eigenvalues" in result  # Check if it found the eigenvalues
    
    # Test shape of results (making sure everything fits!)
    assert result["final_state"].shape == (n, 1)  # Final state should be 3x1
    assert result["steady_state"].shape == (n, 1)  # Steady state should be 3x1
    assert len(result["eigenvalues"]) == n  # Eigenvalues should be 3
    
    # Test properties of steady state (probabilities should make sense!)
    assert np.isclose(result["steady_state"].sum(), 1.0)  # Should add up to 1
    assert np.all(result["steady_state"] >= 0)  # No negative probabilities
    
    # Test invalid Markov matrix (columns don't add up to 1!)
    invalid_P = np.array([
        [0, 0.5, 0.3],
        [0.6, 0, 0.3],
        [0.4, 0.5, 0.5]  # Column sum > 1 (oops!)
    ])
    with pytest.raises(ValueError):
        analyze_webpage_ranking(invalid_P, X0)  # Should raise an error for invalid matrix

def test_create_transformation():
    """
    🧪🔍 Let's Test the Transformation Creator! 🎨✨

    This test checks if the `create_transformation` function can create 
    different types of transformations (like flipping, stretching, rotating, 
    and squishing) correctly. It also makes sure the function catches errors 
    when given bad input.

    What's being tested:
    - Does it create a reflection (flip) correctly? (Like flipping over the y-axis!)
    - Does it create a shear (stretch) correctly? (Like pushing one side of a square!)
    - Does it create a rotation (spin) correctly? (Like turning something 90 degrees!)
    - Does it create a scaling (resizing) correctly? (Like making something twice as big!)
    - Does it create a projection (squish) correctly? (Like flattening something onto an axis!)
    - Does it raise an error when given an invalid transformation type? (Like asking for a "magic" transformation!)

    Let's make sure this function is as creative as an artist! 🎨🎲
    """
    # Test reflection (flip over the y-axis)
    T = create_transformation('reflection', axis='y')
    assert np.allclose(T, np.array([[-1, 0], [0, 1]]))  # Should flip over the y-axis
    
    # Test shear (stretch along the x-axis)
    T = create_transformation('shear', k=2.0, direction='x')
    assert np.allclose(T, np.array([[1, 2], [0, 1]]))  # Should stretch along the x-axis
    
    # Test rotation (spin 90 degrees)
    T = create_transformation('rotation', theta=np.pi/2)
    assert np.allclose(T, np.array([[0, -1], [1, 0]]), atol=1e-10)  # Should rotate 90 degrees
    
    # Test scaling (make everything twice as big)
    T = create_transformation('scaling', sx=2.0)
    assert np.allclose(T, np.array([[2, 0], [0, 2]]))  # Should scale by 2
    
    # Test projection (squish onto the x-axis)
    T = create_transformation('projection', axis='x')
    assert np.allclose(T, np.array([[1, 0], [0, 0]]))  # Should project onto the x-axis
    
    # Test invalid transformation type (oops, no magic here!)
    with pytest.raises(ValueError):
        create_transformation('invalid')  # Should raise an error for invalid type
