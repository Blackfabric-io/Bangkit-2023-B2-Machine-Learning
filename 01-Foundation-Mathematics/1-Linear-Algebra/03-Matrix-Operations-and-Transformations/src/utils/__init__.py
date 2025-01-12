"""
Utility functions for visualization and data handling.

This module provides helper functions for visualizing transformations
and handling data structures.

What's inside:
- plot_transformation: Draws how data changes shape (like squishing or stretching!).
- plot_eigenvectors: Shows the "main directions" in your data (like finding the most important paths!).
- simulate_webpage_navigation: Pretends to click around a website to see how it works.

Use these tools to make your data adventures more exciting! 🚀📈
"""

from .helpers import (
    plot_transformation,
    plot_eigenvectors,
    simulate_webpage_navigation
)

__all__ = [
    'plot_transformation',
    'plot_eigenvectors',
    'simulate_webpage_navigation'
] 