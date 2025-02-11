import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import random
from rl_lib.swiss_round.environment import SwissRoundEnv

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 3 actions: win/draw/lose
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, 
                 env:SwissRoundEnv, 
                 hidden_size:int=128, 
                 buffer_size:int=10000, 
                 batch_size:int=64, 
                 gamma:float=1, # No need to discount future rewards as the ultimate goal is the qualification
                 lr:float=1e-3, 
                 epsilon_start:float=1.0,
                 epsilon_end:float=0.01, 
                 epsilon_decay:float=0.995):
        """
        Args:
            env: SwissRoundEnv instance
            hidden_size: size of hidden layers in Q-network
            buffer_size: size of replay buffer
            batch_size: size of batches for training
            gamma: discount factor
            lr: learning rate
            epsilon_start: starting value for exploration rate
            epsilon_end: minimum value for exploration rate
            epsilon_decay: decay rate for exploration
        """
        self.env = env
        # Verify that the environment has an agent
        if self.env.agent is None:
            raise ValueError("Environment must be initialized with an agent_id")
            
        # Get state size from environment
        # We can do this by getting a state and checking its size
        state = env.reset()
        self.state_size = len(state)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.qnetwork_local = QNetwork(self.state_size, hidden_size).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        # Initialize parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize episode history for analysis
        self.episode_rewards = []
        self.episode_actions = []
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_values = self.qnetwork_local(state)
                return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice([0, 1, 2])
            
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
    def learn(self, experiences):
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(self.device)
        
        # Shape
        # states: [batch_size, state_size]
        # actions: [batch_size]
        # rewards: [batch_size]
        # next_states: [batch_size, state_size]
        # dones: [batch_size]
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]  # shape: [batch_size]
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))  # shape: [batch_size]
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))  # shape: [batch_size, 1]
        
        # Make sure Q_targets has the same shape as Q_expected
        Q_targets = Q_targets.unsqueeze(1)  # shape: [batch_size, 1]     
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def soft_update(self, tau=1e-3):
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                           self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def train(self, n_episodes=1000):
        """
        Train the agent with error handling for failed tournament pairings
        
        Args:
            n_episodes (int): Number of episodes to train
        """
        successful_episodes = 0
        failed_episodes = 0
        
        while successful_episodes < n_episodes:
            try:
                # Initialize episode
                state = self.env.reset()
                episode_reward = 0
                episode_actions = []
                
                # Run episode
                for t in range(self.env.n_rounds):
                    action = self.select_action(state)
                    try:
                        next_state, reward, done = self.env.step(action)
                    except ValueError as e:
                        # If pairing fails, abandon this episode and try again
                        #print(f"Tournament failed: {str(e)}")
                        raise
                    
                    self.step(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                    episode_actions.append(action)
                    
                    if done:
                        break
                
                # Episode completed successfully
                self.episode_rewards.append(episode_reward)
                self.episode_actions.append(episode_actions)
                successful_episodes += 1
                
                # Print progress
                if successful_episodes % 100 == 0:
                    print(f'Episode {successful_episodes}/{n_episodes}, '
                        f'Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}, '
                        f'Epsilon: {self.epsilon:.3f}, '
                        f'(failed episodes: {failed_episodes})')
                
            except ValueError:
                failed_episodes += 1
                if failed_episodes % 100 == 0:
                    print(f"Failed episodes so far: {failed_episodes}")
                continue
            
            except Exception as e:
                print(f"Unexpected error occurred: {str(e)}")
                raise
