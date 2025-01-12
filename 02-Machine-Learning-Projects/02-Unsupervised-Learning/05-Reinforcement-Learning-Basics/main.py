"""
Main script for training and evaluating Q-learning agent.
"""

import argparse
from pathlib import Path
import logging
from src.processors.main import QLearningTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate Q-learning agent.'
    )
    
    # Environment parameters
    parser.add_argument('--grid_size', type=int, default=4,
                       help='Size of the grid world')
    parser.add_argument('--target_state', type=int, default=15,
                       help='Goal state index')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate for Q-learning')
    parser.add_argument('--discount_factor', type=float, default=0.95,
                       help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial exploration rate')
    parser.add_argument('--min_epsilon', type=float, default=0.01,
                       help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Decay rate for exploration')
    parser.add_argument('--n_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Interval for evaluation and plotting')
    
    # Mode selection
    parser.add_argument('--train', action='store_true',
                       help='Train a new agent')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained agent')
    parser.add_argument('--n_eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    # Model saving/loading
    parser.add_argument('--save_dir', type=str,
                       help='Directory to save trained agent')
    parser.add_argument('--load_dir', type=str,
                       help='Directory to load trained agent from')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize or load trainer
    if args.load_dir:
        logger.info("Loading agent from %s", args.load_dir)
        trainer = QLearningTrainer.load(args.load_dir)
    else:
        trainer = QLearningTrainer(
            grid_size=args.grid_size,
            target_state=args.target_state,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon=args.epsilon,
            min_epsilon=args.min_epsilon,
            epsilon_decay=args.epsilon_decay
        )
    
    # Train agent if requested
    if args.train:
        logger.info("Starting training for %d episodes", args.n_episodes)
        results = trainer.train(
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval
        )
        
        # Save agent if requested
        if args.save_dir:
            logger.info("Saving agent to %s", args.save_dir)
            trainer.save(args.save_dir)
    
    # Evaluate agent if requested
    if args.evaluate:
        logger.info("Evaluating agent for %d episodes", args.n_eval_episodes)
        metrics = trainer.evaluate(n_episodes=args.n_eval_episodes)
        
        print("\nEvaluation Results:")
        print(f"Mean Reward: {metrics['mean_reward']:.3f}")
        print(f"Mean Steps: {metrics['mean_steps']:.1f}")
        print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
        
        # Plot final policy and values
        trainer.plot_current_status([])

if __name__ == '__main__':
    main() 