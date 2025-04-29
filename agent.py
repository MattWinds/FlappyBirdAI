#Matthew Windsor and Jay Parmar
#CSCI 6660
#agent.py: This code holds a DQN class (deep neural network that predicts q-values) and a DQN agent that uses the DQN to learn 
#           what to do in the flappy bird environment. The agent learns to choose the best actions by using experience replay,
#           epsilon-greedy exploration, a target network, and a double DQN.

#Libraries
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

#DQN Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        #Define a simple feedforward network
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  #First hidden layer (input_dim -> 256)
            nn.ReLU(),                  #Activation
            nn.Linear(256, 256),         #Second hidden layer (256 -> 256)
            nn.ReLU(),                  #Activation
            nn.Linear(256, output_dim)   #Output layer (256 -> number of actions)
        )

    def forward(self, x):
        return self.fc(x)

#--------------------------------------------------------------------------------------------------------------------------

#DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.00035, batch_size=64, memory_size=100000, epsilon_start=0.9999,
                 epsilon_end=0.01, epsilon_decay=0.9998, target_update_freq=1000):
        
        #Envrionmental properties
        self.state_size = state_size
        self.action_size = action_size
        #Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
    
        #Replay memory
        self.memory = deque(maxlen=memory_size)

        #Epsilon-greedy exploration settings
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        #Choose GPU as device if available, if not use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Main and target model
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  #Initialize target to match model
        self.target_model.eval()    #Target mdel is not trained directly

        #Optimizer and loss functions
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        #Counter to track when to update target network
        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

#--------------------------------------------------------------------------------------------------------------------------

    #Function to save a transition to replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   #Store a transition in the experience replay buffer

#--------------------------------------------------------------------------------------------------------------------------

    #Choose action, explore randomly or pick best
    def act(self, state):

        #Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size) #Random action (exploration)
        
        #Otherwise, use the model to predict best action (exploitation)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()    #Return the action with highest q-value

#--------------------------------------------------------------------------------------------------------------------------

    #Learn from a minibatch sampled from memory
    def replay(self):

        #If not enough samples in memory, don't learn yet
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size) #Sample a random minibatch

        states, actions, rewards, next_states, dones = zip(*minibatch)  #Unpack minibatch into separate batches

        #Convert batches to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        #Current Q-values for taken actions
        q_values = self.model(states).gather(1, actions).squeeze()

        #Double DQN 
        next_actions = self.model(next_states).argmax(1)  #Pick action with main model
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()  #Evaluate that action with target model
        expected_q_values = rewards + self.gamma * next_q_values * (~dones)

        #Loss between current and expected q-values
        loss = self.loss_fn(q_values, expected_q_values)

        #Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Epsilon decay, reduce exploration by epsilon * decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        #Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
