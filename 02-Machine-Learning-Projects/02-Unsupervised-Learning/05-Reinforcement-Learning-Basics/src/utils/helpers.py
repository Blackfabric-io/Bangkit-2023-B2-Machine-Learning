"""
Utility functions for Q-learning implementation.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = npt.NDArray[np.float64]

class GridWorldEnv:
    """Simple grid world environment for Q-learning demonstration."""
    
    def __init__(self, size: int = 4, target_state: int = 15):
        """Initialize grid world environment.
        
        Args:
            size: Size of the grid (size x size).
            target_state: Goal state index.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if size <= 0:
            raise ValueError(f"Invalid grid size: {size}")
        if not 0 <= target_state < size * size:
            raise ValueError(f"Invalid target state: {target_state}")
        
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, right, down, left
        self.target_state = target_state
        
        # Define action effects
        self.action_effects = [
            -size,  # up
            1,      # right
            size,   # down
            -1      # left
        ]
        
        logger.info("Initialized grid world: size=%dx%d, target=%d",
                   size, size, target_state)
    
    def reset(self) -> int:
        """Reset environment to initial state.
        
        Returns:
            Initial state index.
        """
        # Start in random non-target state
        initial_states = [s for s in range(self.n_states)
                         if s != self.target_state]
        self.current_state = np.random.choice(initial_states)
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take a step in the environment.
        
        Args:
            action: Action to take (0-3).
            
        Returns:
            Tuple of (next_state, reward, done).
            
        Raises:
            ValueError: If action is invalid.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Get current position
        row = self.current_state // self.size
        col = self.current_state % self.size
        
        # Apply action
        next_state = self.current_state + self.action_effects[action]
        next_row = next_state // self.size
        next_col = next_state % self.size
        
        # Check if move is valid
        valid = (
            0 <= next_state < self.n_states and
            abs(next_row - row) <= 1 and
            abs(next_col - col) <= 1 and
            not (next_row != row and next_col != col)
        )
        
        if valid:
            self.current_state = next_state
        
        # Compute reward
        if self.current_state == self.target_state:
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False
        
        return self.current_state, reward, done

def plot_training_progress(episode_rewards: List[float],
                         window_size: int = 100,
                         title: str = "Training Progress") -> None:
    """Plot episode rewards during training.
    
    Args:
        episode_rewards: List of rewards per episode.
        window_size: Window size for moving average.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    
    # Plot moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        plt.plot(
            np.arange(window_size-1, len(episode_rewards)),
            moving_avg,
            label=f'Moving Average (n={window_size})'
        )
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_policy_grid(policy: ArrayType, size: int,
                    target_state: int) -> None:
    """Visualize policy on grid.
    
    Args:
        policy: Array of optimal actions for each state.
        size: Size of the grid.
        target_state: Goal state index.
        
    Raises:
        ValueError: If parameters are invalid.
    """
    if len(policy) != size * size:
        raise ValueError("Policy length does not match grid size")
    
    # Create grid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True)
    
    # Plot arrows for each state
    for state in range(size * size):
        if state == target_state:
            continue
            
        row = state // size
        col = state % size
        
        action = policy[state]
        if action == 0:    # up
            dx, dy = 0, 0.3
        elif action == 1:  # right
            dx, dy = 0.3, 0
        elif action == 2:  # down
            dx, dy = 0, -0.3
        else:             # left
            dx, dy = -0.3, 0
            
        ax.arrow(
            col + 0.5, row + 0.5,
            dx, dy,
            head_width=0.1,
            head_length=0.1,
            fc='blue',
            ec='blue'
        )
    
    # Mark target state
    target_row = target_state // size
    target_col = target_state % size
    ax.add_patch(plt.Circle(
        (target_col + 0.5, target_row + 0.5),
        0.3,
        color='green',
        alpha=0.3
    ))
    
    ax.set_xlim(-0.1, size + 0.1)
    ax.set_ylim(size + 0.1, -0.1)
    ax.set_xticks(np.arange(size + 1))
    ax.set_yticks(np.arange(size + 1))
    plt.title('Optimal Policy')
    plt.show()

def plot_value_grid(values: ArrayType, size: int,
                   target_state: int) -> None:
    """Visualize state values on grid.
    
    Args:
        values: Array of state values.
        size: Size of the grid.
        target_state: Goal state index.
        
    Raises:
        ValueError: If parameters are invalid.
    """
    if len(values) != size * size:
        raise ValueError("Values length does not match grid size")
    
    # Reshape values to grid
    value_grid = values.reshape((size, size))
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        value_grid,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'State Value'}
    )
    
    # Mark target state
    target_row = target_state // size
    target_col = target_state % size
    plt.plot(
        target_col + 0.5,
        target_row + 0.5,
        'go',
        markersize=15,
        alpha=0.3
    )
    
    plt.title('State Values')
    plt.show()

def save_training_results(results: Dict[str, Any],
                         save_dir: str) -> None:
    """Save training results to files.
    
    Args:
        results: Dictionary containing training results.
        save_dir: Directory to save results.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save episode rewards
    np.save(
        save_dir / 'episode_rewards.npy',
        np.array(results['episode_rewards'])
    )
    
    # Save training parameters
    with open(save_dir / 'parameters.txt', 'w') as f:
        for key, value in results['parameters'].items():
            f.write(f"{key}: {value}\n")
    
    logger.info("Saved training results to %s", save_dir) 