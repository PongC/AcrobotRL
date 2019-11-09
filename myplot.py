#this python file is to keep plot function here so that the main file is clean

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
import time
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import my_qlearning

def plot_best_height(best_episode_height):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(best_episode_height)
    ax.set_title("best height of each episode")
    return 0

def plot_total_reward(reward_history):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(reward_history)
    ax.set_title("total reward of each episode")
    return 0

def test_run(env, model, n_episodes, n_steps, fps=20, visualization=False):
    #setupvariables
    N=n_episodes
    success_growths = []
    first_successful_steps = []
    successful_stepss = []

    for n in range(N):
        state = env.reset()
        steps = n_steps
        successful_steps = 0
        first_successful_step = steps
        success_growth = []
        for i in range(steps):
            if visualization==True:
                time.sleep(1/fps)
                env.render()
            Q = model(Variable(torch.from_numpy(state).type(torch.FloatTensor))).detach().numpy()
            action = np.argmax(Q)
            state, reward, done, _ = env.step(action)
#             running_reward += reward
            current_height = my_qlearning.get_height(state)
            #success steps count
            if current_height>1:
                successful_steps += 1
                if first_successful_step > i:
                    first_successful_step=i
            success_growth.append(successful_steps)
        success_growths.append(success_growth)
        first_successful_steps.append(first_successful_step)
        successful_stepss.append(successful_steps)

    print('avg successful steps: {} - {:.4f}%'.format(np.mean(successful_stepss), np.mean(successful_stepss)/steps*100))
    print('avg first successful step: {} '.format(np.mean(first_successful_steps)))
    env.close()

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    for n in range(N):
        ax.plot(success_growths[n],'--')
    ax.plot(np.mean(success_growths, axis=0),'.', label="avg_success")
    ax.legend()
    ax.set_title("avg_success from {} sample runs".format(N))
    return np.mean(success_growths, axis=0)
    
def prepare_plot_vars(model, n_samples=2000):
    #number of samples we will plot
    num = n_samples

    #random state variables
    t1 = np.random.uniform(0, 2*np.pi, num)
    t2 = np.random.uniform(0, 2*np.pi, num)
    Y1 = np.cos(t1)
    X1 = np.sin(t1)
    Y2 = np.cos(t2)
    X2 = np.sin(t2)

    #random angular velocity of each state
    S1 = np.random.uniform(-4*np.pi, 4*np.pi, num)
    S2 = np.random.uniform(-9*np.pi, 9*np.pi, num)

    #Z is our action variables predicted by the model from all random variables above
    Z = []
    for i in range(len(t1)):
        _, action = torch.max(model(Variable(torch.from_numpy(np.array([
            Y1[i],X1[i],Y2[i],X2[i],S1[i],S2[i]
            ]))).type(torch.FloatTensor)), dim = -1)
        Z.append(action.item())
    #convert to pandas' Series object
    Z = pd.Series(Z)

    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['-1','0','+1']
    return (Y1,X1,Y2,X2,S1,S2,colors,labels,Z)

def prepare_plot_vars_agent(agent, n_samples=2000):
    #number of samples we will plot
    num = n_samples

    #random state variables
    t1 = np.random.uniform(0, 2*np.pi, num)
    t2 = np.random.uniform(0, 2*np.pi, num)
    Y1 = np.cos(t1)
    X1 = np.sin(t1)
    Y2 = np.cos(t2)
    X2 = np.sin(t2)

    #random angular velocity of each state
    S1 = np.random.uniform(-4*np.pi, 4*np.pi, num)
    S2 = np.random.uniform(-9*np.pi, 9*np.pi, num)

    #Z is our action variables predicted by the model from all random variables above
    Z = []
    for i in range(len(t1)):
        state = np.array([Y1[i],X1[i],Y2[i],X2[i],S1[i],S2[i]])
        action = agent.choose_action(state)
        Z.append(action)
    #convert to pandas' Series object
    Z = pd.Series(Z)

    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['-1','0','+1']
    return (Y1,X1,Y2,X2,S1,S2,colors,labels,Z)

def plot_top_down(Y1,X1,Y2,X2,S1,S2,colors,labels,Z,title=""):
    fig1 = plt.figure(3, figsize=[7,7])
    ax = fig1.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X1,Y1, c=Z)
    ax.set_xlabel('X1')
    ax.set_ylabel('Y1')
    ax.set_title('Policy: '+title)
    recs = []
    for i in range(0,3):
         recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
    plt.show()
    
def plot_side(Y1,X1,Y2,X2,S1,S2,colors,labels,Z,title=""):
    fig2 = plt.figure(3, figsize=[7,7])
    ax = fig2.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X1,S1, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy '+title)
    recs = []
    for i in range(0,3):
         recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
    plt.show()

def plot_3D(Y1,X1,Y2,X2,S1,S2,colors,labels,Z,title=""):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, Y1, S1, c=Z, marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('Y1')
    ax.set_zlabel('Velocity1')
    ax.set_title('Policy '+title)
    recs = []
    for i in range(0,3):
         recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
    plt.show()
