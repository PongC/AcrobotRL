import ipympl
import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import os
import double_dqn
env = gym.make("Acrobot-v1")

class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.state_space = env.observation_space.shape[0] #size of the observation space (state)
        self.action_space = env.action_space.n #number of possible actions
        self.hidden = 100
        #we will only use 2 linear layers here since our environment is quite simple
        self.lin1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.lin2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x):    
        model = torch.nn.Sequential(
            self.lin1,
            self.lin2,
        )
        return model(x)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)
        
#return height of the given state
def get_height(state):
    cos_a, sin_a, cos_b, sin_b , _ , _ = state
    return -cos_a - (cos_a*cos_b - sin_a*sin_b)

def train(env,model, epsilon = 0.3 ,gamma = 0.99, steps =300,episodes=500,learning_rate=0.001 ,successful_episodes=0 ,visual = False):
    #set this variable to True to render the training session
    visualization = visual

    # Parameters
    state = env.reset()
    loss_history = []
    reward_history = []

    #keep tracking best height of each episode and total rewards of each episode
    best_episode_height = []
    rewards = []

    # Initialize DQN_model
    model1 = model
#     model1.apply(weights_init)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model1.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    #close previous environment (in case, we haven't close any environment)
    env.close()

    #the training starts here
    for episode in trange(episodes):
        running_reward = 0
        episode_loss = 0
        episode_reward = 0
        max_height = -2
        state = env.reset()

        for s in range(steps):
            # render the environment every 100 episodes when visualization variable is True
            if episode % 100 == 0 and episode > 0 and visualization==True:
               env.render()

            # get Q-values of the input state from the model
            Q = model1(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

            # Choose epsilon-greedy action
            action = np.random.choice( np.arange(len(Q)), p = double_dqn.eps_greedy_policy(Q.detach().numpy(), epsilon))
            

            # get the next state and reward
            state_1, reward, terminal, _ = env.step(action)
            running_reward += reward

            # Find max Q of the next state from predicted Q-values of our model
            Q1 = model1(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
            maxQ1, _ = torch.max(Q1, -1)

            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target)
            Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)

            # Calculate loss
            loss = loss_fn(Q, Q_target)

            # Update policy
            model1.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            episode_reward += reward

            #compute height of the current step 
            current_height = get_height(state)

            #update max_height
            if current_height > max_height:
                max_height = current_height

            #reach terminal
            if terminal:
                #height of next state
                next_height = get_height(state_1)

                if next_height > max_height:
                    max_height = next_height

                #if terminal is goal state
                if next_height > 1:
                    #success steps count
                    successful_episodes += 1

                    #reduce epsilon everytime we exceed goal height
                    epsilon *= .995

                    # Adjust learning rate
                    scheduler.step()
                break
            else:
                state = state_1

        #record best height of this episode and its reward
        best_episode_height.append(max_height)
        reward_history.append(running_reward)

    print('successful episodes: {:d} - {:.4f}%'.format(successful_episodes, successful_episodes/episodes*100))
    return model, best_episode_height, reward_history

def train_mod_r(env,model, epsilon = 0.3 ,gamma = 0.99, steps =100,
                  episodes=500,learning_rate=0.001 ,successful_episodes=0 ,visual = False):
    #set this variable to True to render the training session
    visualization = visual

    # Parameters
    state = env.reset()
    loss_history = []
    reward_history = []

    #keep tracking best height of each episode and total rewards of each episode
    best_episode_height = []
    rewards = []

    # Initialize DQN_model
    model1 = model
#     model1.apply(weights_init)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model1.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    #close previous environment (in case, we haven't close any environment)
    env.close()

    #the training starts here
    for episode in trange(episodes):
        running_reward = 0
        episode_loss = 0
        episode_reward = 0
        max_height = -2
        state = env.reset()

        for s in range(steps):
            # render the environment every 100 episodes when visualization variable is True
            if episode % 100 == 0 and episode > 0 and visualization==True:
               env.render()

            # get Q-values of the input state from the model
            Q = model1(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

            # Choose epsilon-greedy action
            action = np.random.choice( np.arange(len(Q)), p = double_dqn.eps_greedy_policy(Q.detach().numpy(), epsilon))
            

            # get the next state and reward
            state_1, reward, terminal, _ = env.step(action)
            
            # modified reward so that it takes normalized angular velocity into account -----------(*)
            reward += (2/steps)*(np.absolute(state_1[4])/(4*np.pi))
            
            running_reward += reward

            # Find max Q of the next state from predicted Q-values of our model
            Q1 = model1(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
            maxQ1, _ = torch.max(Q1, -1)

            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target)
            Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)

            # Calculate loss
            loss = loss_fn(Q, Q_target)

            # Update policy
            model1.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            episode_reward += reward

            #compute height of the current step 
            current_height = get_height(state)

            #update max_height
            if current_height > max_height:
                max_height = current_height

            #reach terminal
            if terminal:
                #compute height of next state
                next_height = get_height(state_1)

                if next_height > max_height:
                    max_height = next_height

                #if terminal is goal state
                if next_height > 1:
                    #success steps count
                    successful_episodes += 1

                    #reduce epsilon everytime we exceed goal height
                    epsilon *= .995

                    # Adjust learning rate
                    scheduler.step()
                break
            else:
                state = state_1

        #record best height of this episode and its reward
        best_episode_height.append(max_height)
        reward_history.append(running_reward)

    print('successful episodes: {:d} - {:.4f}%'.format(successful_episodes, successful_episodes/episodes*100))
    return model, best_episode_height, reward_history