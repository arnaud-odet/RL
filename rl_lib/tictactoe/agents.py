import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt 

from rl_lib.tictactoe.environment import TicTacToeEnv, play_demonstration_game

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 64)    # Input: flattened 3x3 board
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 9)    # Output: Q-value for each position
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class Agent:
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 min_epsilon=0.01, learning_rate=0.001):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network and move it to device
        self.q_network = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = []
        
    def select_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            # Convert state to tensor and move to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
            
            # Create a mask for valid moves
            valid_moves_mask = np.ones(9) * float('-inf')
            for move in valid_moves:
                valid_moves_mask[move[0] * 3 + move[1]] = q_values[move[0] * 3 + move[1]]
            
            # Select the best valid move
            action_idx = np.argmax(valid_moves_mask)
            return (action_idx // 3, action_idx % 3)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        # Calculate memory size and number of batches
        memory_size = len(self.memory)
        num_batches = memory_size // batch_size
        
        # Generate random permutation of indices
        indices = np.random.permutation(memory_size)
        
        # Train on all full batches
        for batch_idx in range(num_batches):
            # Get batch indices
            batch_start = batch_idx * batch_size
            batch_indices = indices[batch_start:batch_start + batch_size]
            
            # Get batch data
            batch = [self.memory[i] for i in batch_indices]
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors and move to device
            states = torch.FloatTensor(np.array(states)).to(self.device)          # Shape: (batch_size, 9)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # Shape: (batch_size, 9)
            rewards = torch.FloatTensor(rewards).to(self.device)                  # Shape: (batch_size,)
            dones = torch.FloatTensor(dones).to(self.device)                      # Shape: (batch_size,)
            
            # Current Q-values
            current_q_values = self.q_network(states)            # Shape: (batch_size, 9)
            # Next Q-values
            next_q_values = self.q_network(next_states)         # Shape: (batch_size, 9)
            
            # Calculate target Q-values
            max_next_q_values = torch.max(next_q_values, dim=1)[0]  # Shape: (batch_size,)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
            # Update Q-values for taken actions
            action_indices = [a[0] * 3 + a[1] for a in actions]  # Convert (row, col) to flat index
            q_values = current_q_values[range(batch_size), action_indices]
            
            # Calculate loss and update network
            loss = nn.MSELoss()(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


        
        
def train_agents(episodes=2000,
                 agent1_eps:float = 0.1,
                 agent2_eps:float = 0.1,
                 epsilon_decay:float=0.995,
                 min_epsilon:float = 0.01):
    env = TicTacToeEnv()
    agent1 = Agent(epsilon=agent1_eps)
    agent2 = Agent(epsilon=agent2_eps)
    
    player1_rewards = []
    player2_rewards = []
    moving_avg_window = 100
    
    for episode in range(episodes):
        done = False
        episode_reward_p1 = 0
        episode_reward_p2 = 0
        
        # Determine starting player based on episode number
        player1_starts = episode % 2 == 0
        current_player = 1 if player1_starts else 2
        state = env.reset(starting_player=current_player)
        
        while not done:
            valid_moves = env.get_valid_moves()
            
            if current_player == 1:
                # Player 1's turn
                action = agent1.select_action(state, valid_moves)
                next_state, reward, done = env.make_move(action)
                
                # Store transition for player 1
                agent1.store_transition(state, action, reward, next_state, done)
                episode_reward_p1 += reward
                
                # If game ends, store opposite reward for player 2
                if done:
                    agent2.store_transition(state, action, -reward, next_state, done)
                    episode_reward_p2 -= reward
                    
            else:
                # Player 2's turn
                action = agent2.select_action(state, valid_moves)
                next_state, reward, done = env.make_move(action)
                
                # Store transition for player 2
                agent2.store_transition(state, action, reward, next_state, done)
                episode_reward_p2 += reward
                
                # If game ends, store opposite reward for player 1
                if done:
                    agent1.store_transition(state, action, -reward, next_state, done)
                    episode_reward_p1 -= reward
            
            state = next_state
            current_player = 1 if current_player == 2 else -1
    
        
        # Train both agents using all memory
        agent1.train()
        agent2.train()
        
        # Store the episode rewards
        player1_rewards.append(episode_reward_p1)
        player2_rewards.append(episode_reward_p2)
        
        # Decay epsilon after each episode
        agent1.epsilon = max(min_epsilon, agent1.epsilon * epsilon_decay)
        agent2.epsilon = max(min_epsilon, agent2.epsilon * epsilon_decay)

        print(f"Training in progress : game {episode+1}/{episodes}, epsilon : agent1 = {agent1.epsilon:.4f}, agent2 = {agent2.epsilon:.4f}, {'P1' if player1_starts else 'P2'} started", end='\r')

        print(f"Training in progress : game {episode+1}/{episodes}, epsilon : agent1 = {agent1.epsilon:.4f}, agent2 = {agent2.epsilon:.4f}", end = '\r')
    
    # Plot the rewards for both players
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(player1_rewards, label='Episode Reward', alpha=0.3)
    moving_avg = [np.mean(player1_rewards[max(0, i-moving_avg_window):i+1]) 
                 for i in range(len(player1_rewards))]
    plt.plot(moving_avg, label=f'{moving_avg_window}-Episode Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Player 1 Training Progress')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(player2_rewards, label='Episode Reward', alpha=0.3)
    moving_avg = [np.mean(player2_rewards[max(0, i-moving_avg_window):i+1]) 
                 for i in range(len(player2_rewards))]
    plt.plot(moving_avg, label=f'{moving_avg_window}-Episode Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Player 2 Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Play demonstration game
    print("\nPlaying a demonstration game with trained agents...")
    play_demonstration_game(agent1, agent2, env)