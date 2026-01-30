# Day 23: Reinforcement Learning for Game Playing

## Overview
Train AI agents to play a GridWorld game using Q-Learning and Deep Q-Networks (DQN).

## Concepts Covered
- **Q-Learning**: Tabular value-based learning
- **Deep Q-Network**: Neural network function approximation
- **Epsilon-Greedy**: Exploration vs exploitation
- **Experience Replay**: Learning from past experiences

## Environment: GridWorld
- 5x5 grid with obstacles
- Agent starts at (0,0), goal at (4,4)
- Actions: Up, Down, Left, Right
- Rewards: +100 (goal), -1 (step), -5 (obstacle)

## Requirements
```bash
pip install numpy matplotlib
```

## Usage
```bash
python reinforcement_learning.py
```

## Output Files
- `rl_training_results.png` - Training curves
- `rl_results.json` - Performance metrics

## Key Parameters
- `learning_rate`: 0.1
- `discount_factor`: 0.95
- `epsilon_decay`: 0.995
- `episodes`: 500

## Extend to Other Games
```python
# Install gym for more environments
pip install gymnasium
import gymnasium as gym
env = gym.make('CartPole-v1')
```
