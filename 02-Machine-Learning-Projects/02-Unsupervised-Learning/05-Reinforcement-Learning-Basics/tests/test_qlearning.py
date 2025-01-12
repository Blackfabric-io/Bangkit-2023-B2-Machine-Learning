"""
Unit tests for Q-learning implementation.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.core.base import QLearningAgent, QLearningConfig
from src.utils.helpers import GridWorldEnv
from src.processors.main import QLearningTrainer

def test_qlearning_config():
    """Test Q-learning configuration."""
    # Test valid configuration
    config = QLearningConfig(n_states=16, n_actions=4)
    assert config.n_states == 16
    assert config.n_actions == 4
    assert 0 <= config.learning_rate <= 1
    assert 0 <= config.discount_factor <= 1
    assert 0 <= config.epsilon <= 1
    assert 0 <= config.min_epsilon <= config.epsilon
    assert 0 <= config.epsilon_decay <= 1

def test_qlearning_agent_initialization():
    """Test initialization of Q-learning agent."""
    config = QLearningConfig(n_states=16, n_actions=4)
    agent = QLearningAgent(config)
    
    assert agent.q_table.shape == (16, 4)
    assert np.all(agent.q_table == 0)
    assert agent.epsilon == config.epsilon

def test_qlearning_agent_invalid_config():
    """Test Q-learning agent with invalid configuration."""
    # Test invalid number of states
    with pytest.raises(ValueError):
        config = QLearningConfig(n_states=0, n_actions=4)
        QLearningAgent(config)
    
    # Test invalid number of actions
    with pytest.raises(ValueError):
        config = QLearningConfig(n_states=16, n_actions=-1)
        QLearningAgent(config)
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        config = QLearningConfig(n_states=16, n_actions=4, learning_rate=1.5)
        QLearningAgent(config)

def test_qlearning_agent_action_selection():
    """Test action selection in Q-learning agent."""
    config = QLearningConfig(n_states=16, n_actions=4)
    agent = QLearningAgent(config)
    
    # Test invalid state
    with pytest.raises(ValueError):
        agent.select_action(-1)
    with pytest.raises(ValueError):
        agent.select_action(16)
    
    # Test action selection
    state = 0
    action = agent.select_action(state)
    assert 0 <= action < 4

def test_qlearning_agent_update():
    """Test Q-value update in Q-learning agent."""
    config = QLearningConfig(n_states=16, n_actions=4)
    agent = QLearningAgent(config)
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        agent.update(-1, 0, 0.0, 0)
    with pytest.raises(ValueError):
        agent.update(0, -1, 0.0, 0)
    with pytest.raises(ValueError):
        agent.update(0, 0, 0.0, -1)
    
    # Test Q-value update
    old_value = agent.q_table[0, 0]
    agent.update(0, 0, 1.0, 1)
    assert agent.q_table[0, 0] != old_value

def test_gridworld_env():
    """Test grid world environment."""
    env = GridWorldEnv(size=4, target_state=15)
    
    assert env.n_states == 16
    assert env.n_actions == 4
    assert env.target_state == 15
    
    # Test reset
    state = env.reset()
    assert 0 <= state < 16
    assert state != 15
    
    # Test step
    next_state, reward, done = env.step(0)
    assert 0 <= next_state < 16
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    
    # Test invalid action
    with pytest.raises(ValueError):
        env.step(-1)

def test_qlearning_trainer():
    """Test Q-learning trainer."""
    trainer = QLearningTrainer(
        grid_size=3,
        target_state=8,
        n_episodes=10,
        max_steps=50
    )
    
    assert isinstance(trainer.env, GridWorldEnv)
    assert isinstance(trainer.agent, QLearningAgent)
    assert trainer.env.size == 3
    assert trainer.env.target_state == 8

def test_qlearning_training():
    """Test Q-learning training process."""
    trainer = QLearningTrainer(
        grid_size=3,
        target_state=8,
        n_episodes=10,
        max_steps=50
    )
    
    # Train agent
    results = trainer.train(
        n_episodes=10,
        max_steps=50,
        eval_interval=5
    )
    
    assert 'episode_rewards' in results
    assert len(results['episode_rewards']) == 10
    assert 'final_policy' in results
    assert 'final_values' in results

def test_qlearning_evaluation():
    """Test Q-learning evaluation."""
    trainer = QLearningTrainer(
        grid_size=3,
        target_state=8,
        n_episodes=10,
        max_steps=50
    )
    
    # Train and evaluate
    trainer.train(n_episodes=10, max_steps=50)
    metrics = trainer.evaluate(n_episodes=5)
    
    assert 'mean_reward' in metrics
    assert 'mean_steps' in metrics
    assert 'success_rate' in metrics
    assert 0 <= metrics['success_rate'] <= 1

def test_qlearning_save_load():
    """Test saving and loading Q-learning agent."""
    trainer = QLearningTrainer(
        grid_size=3,
        target_state=8,
        n_episodes=10,
        max_steps=50
    )
    
    # Train agent
    trainer.train(n_episodes=10, max_steps=50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save agent
        trainer.save(tmp_dir)
        
        # Load agent
        loaded_trainer = QLearningTrainer.load(tmp_dir)
        
        # Compare Q-tables
        np.testing.assert_array_equal(
            trainer.agent.q_table,
            loaded_trainer.agent.q_table
        )
        
        # Compare parameters
        assert trainer.parameters == loaded_trainer.parameters 