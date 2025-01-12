"""
Main processing logic for Q-learning implementation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import logging
from src.core.base import QLearningAgent, QLearningConfig
from src.utils.helpers import (
    GridWorldEnv,
    plot_training_progress,
    plot_policy_grid,
    plot_value_grid,
    save_training_results
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLearningTrainer:
    """Trainer for Q-learning agent."""
    
    def __init__(self, grid_size: int = 4,
                 target_state: int = 15,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 min_epsilon: float = 0.01,
                 epsilon_decay: float = 0.995):
        """Initialize trainer.
        
        Args:
            grid_size: Size of the grid world.
            target_state: Goal state index.
            learning_rate: Learning rate for Q-learning.
            discount_factor: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            min_epsilon: Minimum exploration rate.
            epsilon_decay: Decay rate for exploration.
        """
        # Initialize environment
        self.env = GridWorldEnv(grid_size, target_state)
        
        # Initialize agent
        config = QLearningConfig(
            n_states=self.env.n_states,
            n_actions=self.env.n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            min_epsilon=min_epsilon,
            epsilon_decay=epsilon_decay
        )
        self.agent = QLearningAgent(config)
        
        # Store parameters
        self.parameters = {
            'grid_size': grid_size,
            'target_state': target_state,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon,
            'min_epsilon': min_epsilon,
            'epsilon_decay': epsilon_decay
        }
        
        logger.info("Initialized Q-learning trainer with parameters: %s",
                   self.parameters)
    
    def train(self, n_episodes: int = 1000,
              max_steps: int = 100,
              eval_interval: int = 100) -> Dict[str, Any]:
        """Train the Q-learning agent.
        
        Args:
            n_episodes: Number of training episodes.
            max_steps: Maximum steps per episode.
            eval_interval: Interval for evaluation and plotting.
            
        Returns:
            Dictionary containing training results.
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Select and take action
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Update Q-values
                self.agent.update(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Decay exploration rate
            self.agent.decay_epsilon()
            
            # Store episode reward
            episode_rewards.append(total_reward)
            
            # Log progress
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(episode_rewards[-eval_interval:])
                logger.info(
                    "Episode %d/%d: avg_reward=%.3f, epsilon=%.3f",
                    episode + 1, n_episodes, avg_reward, self.agent.epsilon
                )
                
                # Plot current progress
                self.plot_current_status(episode_rewards)
        
        # Compile results
        results = {
            'episode_rewards': episode_rewards,
            'parameters': self.parameters,
            'final_policy': self.agent.get_optimal_policy(),
            'final_values': self.agent.get_state_values()
        }
        
        return results
    
    def plot_current_status(self, episode_rewards: List[float]) -> None:
        """Plot current training status.
        
        Args:
            episode_rewards: List of rewards per episode.
        """
        # Plot training progress
        plot_training_progress(episode_rewards)
        
        # Plot current policy
        policy = self.agent.get_optimal_policy()
        plot_policy_grid(
            policy,
            self.env.size,
            self.env.target_state
        )
        
        # Plot state values
        values = self.agent.get_state_values()
        plot_value_grid(
            values,
            self.env.size,
            self.env.target_state
        )
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """Evaluate trained agent.
        
        Args:
            n_episodes: Number of evaluation episodes.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        rewards = []
        steps = []
        success_rate = 0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            n_steps = 0
            
            while True:
                # Select best action (no exploration)
                action = np.argmax(self.agent.q_table[state])
                next_state, reward, done = self.env.step(action)
                
                total_reward += reward
                n_steps += 1
                state = next_state
                
                if done:
                    success_rate += 1
                    break
                    
                if n_steps >= 100:
                    break
            
            rewards.append(total_reward)
            steps.append(n_steps)
        
        metrics = {
            'mean_reward': float(np.mean(rewards)),
            'mean_steps': float(np.mean(steps)),
            'success_rate': float(success_rate / n_episodes)
        }
        
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
    
    def save(self, save_dir: str) -> None:
        """Save trained agent and results.
        
        Args:
            save_dir: Directory to save results.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Q-table
        self.agent.save(save_dir / 'q_table.npy')
        
        # Save parameters
        with open(save_dir / 'parameters.txt', 'w') as f:
            for key, value in self.parameters.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Saved agent to %s", save_dir)
    
    @classmethod
    def load(cls, load_dir: str) -> 'QLearningTrainer':
        """Load trained agent.
        
        Args:
            load_dir: Directory containing saved files.
            
        Returns:
            Loaded QLearningTrainer instance.
        """
        load_dir = Path(load_dir)
        
        # Load parameters
        parameters = {}
        with open(load_dir / 'parameters.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                parameters[key] = float(value) if '.' in value else int(value)
        
        # Create trainer with loaded parameters
        trainer = cls(
            grid_size=parameters['grid_size'],
            target_state=parameters['target_state'],
            learning_rate=parameters['learning_rate'],
            discount_factor=parameters['discount_factor'],
            epsilon=parameters['epsilon'],
            min_epsilon=parameters['min_epsilon'],
            epsilon_decay=parameters['epsilon_decay']
        )
        
        # Load Q-table
        trainer.agent.load(load_dir / 'q_table.npy')
        
        logger.info("Loaded agent from %s", load_dir)
        return trainer 