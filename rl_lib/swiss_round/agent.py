import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from collections import namedtuple
import random
from rl_lib.swiss_round.environment import SwissRoundEnv


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_dims=[256,128,64], dropout:float=0.1):
        super(QNetwork, self).__init__()
        
        layers = []
        
        for i, hidden_size in enumerate(hidden_dims):
            input_dim = state_size if i == 0 else hidden_size
            output_dim = 3 if i == len(hidden_dims) -1 else hidden_dims[i+1]
            layers.append(nn.Linear(input_dim, output_dim))
            if not i == len(hidden_dims)-1 :
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.Dropout(dropout))
                
        self.network = nn.Sequential(*layers)    
        
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
                 hidden_dims:list=[256,128,64], 
                 dropout:float=0.1,
                 buffer_size:int=10000, 
                 batch_size:int=64, 
                 gamma:float=1, # No need to discount future rewards as the ultimate goal is the qualification
                 lr:float=1e-3, 
                 epsilon_start:float=1.0,
                 epsilon_end:float=0.01, 
                 epsilon_decay:float=0.995,
                 log_dir:str = "/home/admin/code/arnaud-odet/2_projets/reinforcement_learning/logs",
                 history_prefix:str = "loc"):
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
        self.qnetwork_local = QNetwork(self.state_size, hidden_dims, dropout).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, hidden_dims, dropout).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        # Recording network parameters
        self.lr = lr
        self.n_layers = len(hidden_dims)
        self.layers = ('_').join([str(i) for i in hidden_dims])
        self.dropout = dropout
        self.batch_size=batch_size
        self.buffer_size = buffer_size
        
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
        self.gambits_count = []
        
        self.log_folder= log_dir
        self.id = history_prefix + '_' + str(self.find_id())
        
    def find_id(self):
        filepath = os.path.join(self.log_folder, 'exp_logs.csv')
        if os.path.exists(filepath):
            ldf = pd.read_csv(filepath, index_col=0)
            pass_exp_ids = [int(s.split('_')[-1]) for s in ldf['exp_id']]
            exp_id = max(pass_exp_ids)
            return exp_id +1
        else :
            return 1
        
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
            
    def log_hyperparameters(self, avg_test_reward, std_rewards, avg_test_gambits_count, std_gambits):
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        
        logs =  pd.DataFrame(
            [{
                'exp_id':self.id,
                'env_name':self.env.name,
                'n_teams':self.env.n_teams,
                'n_rounds':self.env.n_rounds,
                'thresholds':('_').join([str(i) for i in self.env.threshold_ranks]),
                'bonuses':('_').join([str(i) for i in self.env.bonus_points]),
                'agent_id' : self.env.agent.id,
                'strengths':('_').join([str(t.strength) for t in self.env.teams]),                
                'n_episodes' : self.n_train_episodes,
                'n_test_episodes' : self.n_test_episodes,
                'lr' : self.lr,
                'n_layers' : self.n_layers,
                'layers' : self.layers, 
                'dropout' : self.dropout,
                'batch_size' : self.batch_size,
                'buffer_size' : self.buffer_size,
                'batch_size' : self.batch_size,
                'gamma' : self.gamma,
                'epsilon' : self.epsilon,
                'epsilon_end' : self.epsilon_end,
                'epsilon_decay' : self.epsilon_decay, 
                'avg_test_rewards' : avg_test_reward,
                'std_test_rewards' : std_rewards,
                'avg_test_gambits' : avg_test_gambits_count,
                'std_test_gambits' : std_gambits,
            }]
        )

        # Check if the file exists, if not, create it as a .csv with pandas
        filepath = os.path.join(self.log_folder, 'exp_logs.csv')
        if os.path.exists(filepath):
            log_df = pd.read_csv(filepath, index_col=0)
            logs = pd.concat([log_df,logs], ignore_index=True)
        logs.to_csv(filepath)
        
        print(f"Hyperparameters logs saved in file {filepath}")
    
    def log_history(self):
    
        history_folder = os.path.join(self.log_folder,'histories')
        index = 1
        # Check if the directory exists, if not, create it
        if not os.path.exists(history_folder):
            os.makedirs(history_folder)
        
        filename = f'training_history_{self.id}'

        filepath = os.path.join(history_folder, filename)
        data = np.array([self.episode_rewards]+[self.gambits_count]).T
        np.save(filepath, data)
        
        print(f"History logs saved in file {filepath}")
            
    def train(self, n_episodes=1000):
        """
        Train the agent with error handling for failed tournament pairings
        
        Args:
            n_episodes (int): Number of episodes to train
        """
        successful_episodes = 0
        failed_episodes = 0
        self.n_train_episodes = n_episodes
        verbose_step = 100
        print('--- Training in progress ---')        
        while successful_episodes < n_episodes:
            if self.epsilon - self.epsilon_end < 0.001 :
                verbose_step = n_episodes // 10
            try:
                # Initialize episode
                state = self.env.reset()
                episode_reward = 0
                episode_actions = []
                gambit_count = 0
                
                # Run episode
                for t in range(self.env.n_rounds):
                    action = self.select_action(state)
                    gambit_count += int(action != 0)
                    try:
                        next_state, reward, done = self.env.step(action)
                    except ValueError as e:
                        # If pairing fails, abandon this episode and try again
                        #print(f"Tournament failed: {str(e)}")
                        failed_episodes+=1
                        raise
                    
                    self.step(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                    episode_actions.append(action)
                    
                    if done:
                        break
                
                # Episode completed successfully
                #print(f"Episode reward : {episode_reward}")
                self.episode_rewards.append(episode_reward)
                self.episode_actions.append(episode_actions)
                self.gambits_count.append(gambit_count)
                successful_episodes += 1
                
                # Print progress
                if successful_episodes % verbose_step == 0 or successful_episodes == n_episodes:
                    print(f'Episode {successful_episodes}/{n_episodes} | '
                        f'Avg Reward: {np.mean(self.episode_rewards[-verbose_step:]):.2f} ± {np.std(self.episode_rewards[-verbose_step:]):.2f} |  '
                        f'Avg nb gambits played {np.mean(self.gambits_count[-verbose_step:]):.2f} ± {np.std(self.gambits_count[-verbose_step:]):.2f} | '
                        f'Epsilon: {self.epsilon:.3f} | '
                        f'Failed episodes: {failed_episodes}')
                
            except ValueError:
                continue
            
            except Exception as e:
                print(f"Unexpected error occurred: {str(e)}")
                raise

    def evaluate(self, n_episodes=1000):
        """
        Evaluate the trained agent without exploration
        
        Args:
            n_episodes (int): Number of test episodes to run
            
        """
        # Store the original epsilon
        original_epsilon = self.epsilon
        
        # Set epsilon to 0 for pure exploitation
        self.epsilon = 0
        
        successful_episodes = 0
        failed_episodes = 0
        self.n_test_episodes = n_episodes
        test_rewards = []
        test_actions = []
        test_gambits_count = []
        verbose_step = n_episodes // 5
        print('--- Evaluation in progress ---')
        while successful_episodes < n_episodes:
            try:
                # Initialize episode
                state = self.env.reset()
                episode_reward = 0
                episode_actions = []
                gambit_count = 0
                
                # Run episode
                for t in range(self.env.n_rounds):
                    action = self.select_action(state)
                    gambit_count += int(action != 0)
                    try:
                        next_state, reward, done = self.env.step(action)
                    except ValueError as e:
                        failed_episodes+=1
                        raise
                    
                    
                    state = next_state
                    episode_reward += reward
                    episode_actions.append(action)
                    
                    if done:
                        break
                
                # Episode completed successfully
                #print(f"Episode reward : {episode_reward}")
                test_rewards.append(episode_reward)
                test_actions.append(episode_actions)
                test_gambits_count.append(gambit_count)
                successful_episodes += 1
                
                # Print progress
                if successful_episodes % verbose_step == 0 or successful_episodes == n_episodes:
                    print(f'Episode {successful_episodes}/{n_episodes} | '
                        f'Avg Reward: {np.mean(test_rewards[-verbose_step:]):.2f} ± {np.std(test_rewards[-verbose_step:]):.2f} | '
                        f'Avg nb gambits played {np.mean(test_gambits_count[-verbose_step:]):.2f} ± {np.std(test_gambits_count[-verbose_step:]):.2f} | '
                        f'Failed episodes: {failed_episodes}')
                
            except ValueError:
                continue
            
            except Exception as e:
                print(f"Unexpected error occurred: {str(e)}")
                raise
        
        # Restore the original epsilon
        self.epsilon = original_epsilon
        
        # Calculate statistics
        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        mean_gambits = np.mean(test_gambits_count)
        std_gambits = np.std(test_gambits_count)
        
        print("\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean Gambits count: {mean_gambits:.2f} ± {std_gambits:.2f}")        
        print(f"Best Episode Reward: {max(test_rewards):.2f}")
        print(f"Worst Episode Reward: {min(test_rewards):.2f}")
  
        self.log_hyperparameters(avg_test_reward=mean_reward, 
                                 std_rewards= std_reward,
                                 avg_test_gambits_count=mean_gambits,
                                 std_gambits=std_gambits)
        self.log_history()
