"""
Core implementation of Q-learning algorithm.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ArrayType = npt.NDArray[np.float64]
StateType = int
ActionType = int

@dataclass
class QLearningConfig:
    """Configuration for Q-learning agent."""
    n_states: int
    n_actions: int
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.995

class QLearningAgent:
    """Q-learning agent implementation."""
    
    def __init__(self, config: QLearningConfig):
        """Initialize Q-learning agent.
        
        Args:
            config: Configuration parameters.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        self._validate_config(config)
        self.config = config
        
        # Initialize Q-table
        self.q_table = np.zeros((config.n_states, config.n_actions))
        self.epsilon = config.epsilon
        
        logger.info("Initialized Q-learning agent with config: %s", config)
    
    def _validate_config(self, config: QLearningConfig) -> None:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if config.n_states <= 0:
            raise ValueError(f"Invalid number of states: {config.n_states}")
        if config.n_actions <= 0:
            raise ValueError(f"Invalid number of actions: {config.n_actions}")
        if not 0 <= config.learning_rate <= 1:
            raise ValueError(f"Invalid learning rate: {config.learning_rate}")
        if not 0 <= config.discount_factor <= 1:
            raise ValueError(f"Invalid discount factor: {config.discount_factor}")
        if not 0 <= config.epsilon <= 1:
            raise ValueError(f"Invalid epsilon: {config.epsilon}")
        if not 0 <= config.min_epsilon <= config.epsilon:
            raise ValueError(f"Invalid min_epsilon: {config.min_epsilon}")
        if not 0 <= config.epsilon_decay <= 1:
            raise ValueError(f"Invalid epsilon decay: {config.epsilon_decay}")
    
    def select_action(self, state: StateType) -> ActionType:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            
        Returns:
            Selected action.
            
        Raises:
            ValueError: If state is invalid.
        """
        if not 0 <= state < self.config.n_states:
            raise ValueError(f"Invalid state: {state}")
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select random action
            action = random.randint(0, self.config.n_actions - 1)
            logger.debug("Exploring: selected random action %d", action)
        else:
            # Exploit: select best action
            action = np.argmax(self.q_table[state])
            logger.debug("Exploiting: selected best action %d", action)
        
        return action
    
    def update(self, state: StateType, action: ActionType,
               reward: float, next_state: StateType) -> None:
        """Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state.
            action: Selected action.
            reward: Received reward.
            next_state: Next state.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0 <= state < self.config.n_states:
            raise ValueError(f"Invalid state: {state}")
        if not 0 <= action < self.config.n_actions:
            raise ValueError(f"Invalid action: {action}")
        if not 0 <= next_state < self.config.n_states:
            raise ValueError(f"Invalid next_state: {next_state}")
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.config.discount_factor * \
                   self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.config.learning_rate * td_error
        
        logger.debug("Updated Q-value: state=%d, action=%d, new_value=%.3f",
                    state, action, self.q_table[state, action])
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.min_epsilon,
            self.epsilon * self.config.epsilon_decay
        )
        logger.debug("Decayed epsilon to %.3f", self.epsilon)
    
    def get_optimal_policy(self) -> ArrayType:
        """Get optimal policy from current Q-table.
        
        Returns:
            Array of optimal actions for each state.
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_state_values(self) -> ArrayType:
        """Get state values from current Q-table.
        
        Returns:
            Array of maximum Q-values for each state.
        """
        return np.max(self.q_table, axis=1)
    
    def save(self, filepath: str) -> None:
        """Save Q-table to file.
        
        Args:
            filepath: Path to save file.
        """
        np.save(filepath, self.q_table)
        logger.info("Saved Q-table to %s", filepath)
    
    def load(self, filepath: str) -> None:
        """Load Q-table from file.
        
        Args:
            filepath: Path to load file.
            
        Raises:
            ValueError: If loaded Q-table has wrong shape.
        """
        q_table = np.load(filepath)
        if q_table.shape != (self.config.n_states, self.config.n_actions):
            raise ValueError(
                f"Loaded Q-table shape {q_table.shape} does not match "
                f"expected shape {(self.config.n_states, self.config.n_actions)}"
            )
        self.q_table = q_table
        logger.info("Loaded Q-table from %s", filepath) 