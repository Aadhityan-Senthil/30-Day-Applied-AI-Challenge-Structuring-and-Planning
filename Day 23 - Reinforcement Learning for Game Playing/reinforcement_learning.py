"""
Day 23: Reinforcement Learning for Game Playing
30-Day AI Challenge

Train an AI agent to play games using Q-Learning and Deep Q-Networks.
Demonstrates on a simple GridWorld environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import random

class GridWorld:
    """Simple GridWorld environment for RL."""
    
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
        # Define rewards
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
    def reset(self):
        """Reset environment to initial state."""
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action):
        """Take action and return new state, reward, done."""
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        new_pos = (
            max(0, min(self.size-1, self.agent_pos[0] + moves[action][0])),
            max(0, min(self.size-1, self.agent_pos[1] + moves[action][1]))
        )
        
        # Check obstacles
        if new_pos in self.obstacles:
            new_pos = self.agent_pos  # Can't move into obstacle
            reward = -5
        elif new_pos == self.goal:
            reward = 100
        else:
            reward = -1  # Small penalty for each step
        
        self.agent_pos = new_pos
        done = (new_pos == self.goal)
        
        return new_pos, reward, done
    
    def render(self):
        """Render the grid as ASCII."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        print('\n'.join([' '.join(row) for row in grid]))
        print()


class QLearningAgent:
    """Q-Learning agent for GridWorld."""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-Learning update rule."""
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-Learning update
        self.q_table[state][action] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DQNAgent:
    """Deep Q-Network agent (simplified numpy version)."""
    
    def __init__(self, state_size, action_size, hidden_size=24):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001
        
        # Simple neural network weights
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1
        self.b2 = np.zeros(action_size)
    
    def _forward(self, state):
        """Forward pass through network."""
        h = np.maximum(0, np.dot(state, self.W1) + self.b1)  # ReLU
        return np.dot(h, self.W2) + self.b2
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_vec = np.array([state[0], state[1]])
        q_values = self._forward(state_vec)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)
    
    def replay(self, batch_size=32):
        """Train on batch from memory."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_vec = np.array([state[0], state[1]])
            next_vec = np.array([next_state[0], next_state[1]])
            
            target = reward
            if not done:
                target += self.gamma * np.max(self._forward(next_vec))
            
            # Simple gradient update
            q_values = self._forward(state_vec)
            error = target - q_values[action]
            
            # Update weights (simplified backprop)
            h = np.maximum(0, np.dot(state_vec, self.W1) + self.b1)
            self.W2[:, action] += self.lr * error * h
            self.b2[action] += self.lr * error
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(env, agent, episodes=500):
    """Train the RL agent."""
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:  # Max steps per episode
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            if isinstance(agent, QLearningAgent):
                agent.learn(state, action, reward, next_state, done)
            else:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        if isinstance(agent, QLearningAgent):
            agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{episodes} - Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history


def evaluate_agent(env, agent, episodes=10):
    """Evaluate trained agent."""
    total_rewards = []
    total_steps = []
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 100:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'success_rate': sum(1 for r in total_rewards if r > 0) / episodes * 100
    }


def visualize_policy(env, agent):
    """Visualize the learned policy."""
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    print("\nLearned Policy:")
    for i in range(env.size):
        row = []
        for j in range(env.size):
            if (i, j) == env.goal:
                row.append('G')
            elif (i, j) in env.obstacles:
                row.append('X')
            else:
                action = agent.choose_action((i, j))
                row.append(arrows[action])
        print(' '.join(row))


def plot_training_results(rewards_history, steps_history):
    """Plot training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Smoothed rewards
    window = 50
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    
    axes[0].plot(rewards_history, alpha=0.3, color='blue')
    axes[0].plot(range(window-1, len(rewards_history)), smoothed, color='red', linewidth=2)
    axes[0].set_title('Training Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    smoothed_steps = np.convolve(steps_history, np.ones(window)/window, mode='valid')
    axes[1].plot(steps_history, alpha=0.3, color='green')
    axes[1].plot(range(window-1, len(steps_history)), smoothed_steps, color='red', linewidth=2)
    axes[1].set_title('Steps per Episode')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_training_results.png', dpi=150)
    plt.close()
    print("Training results saved to 'rl_training_results.png'")


def main():
    print("=" * 50)
    print("Day 23: Reinforcement Learning for Game Playing")
    print("=" * 50)
    
    # Create environment
    print("\n[1] Creating GridWorld environment...")
    env = GridWorld(size=5)
    
    print("\nInitial Grid:")
    env.render()
    print("A = Agent, G = Goal, X = Obstacle")
    
    # Train Q-Learning Agent
    print("\n[2] Training Q-Learning Agent...")
    q_agent = QLearningAgent(state_size=25, action_size=4)
    q_rewards, q_steps = train_agent(env, q_agent, episodes=500)
    
    # Evaluate Q-Learning
    print("\n[3] Evaluating Q-Learning Agent...")
    q_results = evaluate_agent(env, q_agent)
    print(f"  Average Reward: {q_results['avg_reward']:.2f}")
    print(f"  Average Steps: {q_results['avg_steps']:.2f}")
    print(f"  Success Rate: {q_results['success_rate']:.1f}%")
    
    # Visualize policy
    print("\n[4] Learned Policy Visualization:")
    agent_epsilon = q_agent.epsilon
    q_agent.epsilon = 0
    visualize_policy(env, q_agent)
    q_agent.epsilon = agent_epsilon
    
    # Train DQN Agent
    print("\n[5] Training DQN Agent...")
    dqn_agent = DQNAgent(state_size=2, action_size=4)
    dqn_rewards, dqn_steps = train_agent(env, dqn_agent, episodes=500)
    
    # Evaluate DQN
    print("\n[6] Evaluating DQN Agent...")
    dqn_results = evaluate_agent(env, dqn_agent)
    print(f"  Average Reward: {dqn_results['avg_reward']:.2f}")
    print(f"  Average Steps: {dqn_results['avg_steps']:.2f}")
    print(f"  Success Rate: {dqn_results['success_rate']:.1f}%")
    
    # Plot results
    print("\n[7] Generating visualizations...")
    plot_training_results(q_rewards, q_steps)
    
    # Save results
    results = {
        'q_learning': q_results,
        'dqn': dqn_results,
        'training_episodes': 500
    }
    
    with open('rl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to 'rl_results.json'")
    
    # Demo run
    print("\n[8] Demo: Watch trained agent play...")
    env.reset()
    q_agent.epsilon = 0
    
    print("\nAgent path to goal:")
    for step in range(20):
        print(f"Step {step + 1}:")
        env.render()
        
        action = q_agent.choose_action(env.agent_pos)
        _, _, done = env.step(action)
        
        if done:
            print("Goal reached! 🎉")
            env.render()
            break
    
    print("\n" + "=" * 50)
    print("Day 23 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
