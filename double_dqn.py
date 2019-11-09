import torch
from torch import nn
import numpy as np
from collections import deque
from collections import namedtuple
import my_qlearning
import myplot
import matplotlib.pyplot as plt
import time


class ExperienceReplay:
    def __init__(self, device, num_states, buffer_size=1e+6):
        self._device = device
        self.__buffer = deque(maxlen=int(buffer_size))
        self._num_states = num_states

    @property
    def buffer_length(self):
        return len(self.__buffer)

    def add(self, transition):
        '''
        Adds a transition <s, a, r, s', t > to the replay buffer
        :param transition:
        :return:
        '''
        self.__buffer.append(transition)

    def sample_minibatch(self, batch_size=128):
        '''
        :param batch_size:
        :return:
        '''
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = np.zeros([batch_size, self._num_states],
                               dtype=np.float32)
        action_batch = np.zeros([
            batch_size,
        ], dtype=np.int64)
        reward_batch = np.zeros([
            batch_size,
        ], dtype=np.float32)
        nonterminal_batch = np.zeros([
            batch_size,
        ], dtype=np.bool)
        next_state_batch = np.zeros([batch_size, self._num_states],
                                    dtype=np.float32)
        for i, index in zip(range(batch_size), ids):
            state_batch[i, :] = self.__buffer[index].s
            action_batch[i] = self.__buffer[index].a
            reward_batch[i] = self.__buffer[index].r
            nonterminal_batch[i] = self.__buffer[index].t
            next_state_batch[i, :] = self.__buffer[index].next_s

        return (
            torch.tensor(state_batch, dtype=torch.float, device=self._device),
            torch.tensor(action_batch, dtype=torch.long, device=self._device),
            torch.tensor(reward_batch, dtype=torch.float, device=self._device),
            torch.tensor(next_state_batch,
                         dtype=torch.float,
                         device=self._device),
            torch.tensor(nonterminal_batch,
                         dtype=torch.bool,
                         device=self._device),
        )


class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions

#         self._fc1 = nn.Linear(self._num_states, 16, bias=False)
#         self._fc_final = nn.Linear(16, self._num_actions, bias=False)
        self._fc_final = nn.Linear(self._num_states, self._num_actions, bias=False)

        # Initialize final layer uniformly in [-1e-6, 1e-6] range, according to old Keras implementation
        nn.init.uniform_(self._fc_final.weight, a=-1e-6, b=1e-6)

    def forward(self, state):
        h=state
#         h = self._fc1(state)
#         h = self._relu1(self._fc1(state))
#         h = self._relu2(self._fc2(h))
        q_values = self._fc_final(h)
        return q_values


class DoubleQLearningModel(object):
    def __init__(self, device, num_states, num_actions, learning_rate):
        self._device = device
        self._num_states = num_states
        self._num_actions = num_actions
        self._lr = learning_rate

        # Define the two deep Q-networks
        self.online_model = QNetwork(self._num_states,
                                     self._num_actions).to(device=self._device)
        self.offline_model = QNetwork(
            self._num_states, self._num_actions).to(device=self._device)

        # Define optimizer. Should update online network parameters only.
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(),
                                             lr=self._lr)

        # Define loss function
        self._mse = nn.MSELoss(reduction='mean').to(device=self._device)

    def calc_loss(self, q_online_curr, q_target, a):
        '''
        Calculate loss for given batch
        :param q_online_curr: batch of q values at current state. Shape (N, num actions)
        :param q_target: batch of temporal difference targets. Shape (N,)
        :param a: batch of actions taken at current state. Shape (N,)
        :return:
        '''
        batch_size = q_online_curr.shape[0]
        assert q_online_curr.shape == (batch_size, self._num_actions)
        assert q_target.shape == (batch_size, )
        assert a.shape == (batch_size, )

        # Select only the Q-values corresponding to the actions taken (loss should only be applied for these)
        q_online_curr_allactions = q_online_curr
        q_online_curr = q_online_curr[torch.arange(batch_size),
                                      a]  # New shape: (batch_size,)
        assert q_online_curr.shape == (batch_size, )
        for j in [0, 3, 4]:
            assert q_online_curr_allactions[j, a[j]] == q_online_curr[j]

        # Make sure that gradient is not back-propagated through Q target
        assert not q_target.requires_grad

        loss = self._mse(q_online_curr, q_target)
        assert loss.shape == ()

        return loss

    def update_target_network(self):
        '''
        Update target network parameters, by copying from online network.
        '''
        online_params = self.online_model.state_dict()
        self.offline_model.load_state_dict(online_params)

def eps_greedy_policy(q_values, eps):
    # YOUR CODE HERE
    policy = np.ones([q_values.shape[0]])
    best_action = np.argmax(q_values)
    policy = eps * policy/len(policy)
    policy[best_action] += 1 - eps
    return policy

def sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma):
    '''
    Sample mini-batch from replay buffer, and compute the mini-batch loss
    Inputs:
        ddqn          - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        replay_buffer - Replay buffer object (from which smaples will be drawn)
        batch_size    - Batch size
        gamma         - Discount factor
    Returns:
        Mini-batch loss, on which .backward() will be called to compute gradient.
    '''
    # Sample a minibatch of transitions from replay buffer
    curr_state, curr_action, reward, next_state, nonterminal = replay_buffer.sample_minibatch(batch_size)

    # FYI:
    # ddqn.online_model & ddqn.offline_model are Pytorch modules for online / offline Q-networks, which take the state as input,
    # and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).

    # YOUR CODE HERE
    
    q_online_curr  = ddqn.online_model(curr_state)
    with torch.no_grad():
        q_online_next  = ddqn.online_model(next_state)
        q_offline_next = ddqn.offline_model(next_state)
    
    q_target = calculate_q_targets(q_online_next, q_offline_next, reward, nonterminal, gamma=gamma)
    q_target = q_target.detach()
    loss = ddqn.calc_loss(q_online_curr, q_target, curr_action)

    return loss

def train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=False, batch_size=64, gamma=.94):        
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    eps = 1.
    eps_end = .1 
    eps_decay = .001
    tau = 1000
    cnt_updates = 0
    R_buffer = []
    R_avg = []
    max_height_history = []
    for i in range(num_episodes):
        state = env.reset() # Initial state
        state = state[None,:] # Add singleton dimension, to represent as batch of size 1.
        finish_episode = False # Initialize
        ep_reward = 0 # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
        q_buffer = []
        steps = 0
        max_eps_height=-2
        while not finish_episode:
            if enable_visualization:
                env.render() # comment this line out if you don't want to / cannot render the environment on your system
            steps += 1

            # Take one step in environment. No need to compute gradients,
            # we will just store transition to replay buffer, and later sample a whole batch
            # from the replay buffer to actually take a gradient step.
            q_online_curr, curr_action = calc_q_and_take_action(ddqn, state, eps)
            q_buffer.append(q_online_curr)
            new_state, reward, finish_episode, _ = env.step(curr_action) # take one step in the evironment
            new_state = new_state[None,:]
            next_height = my_qlearning.get_height(new_state[0])
            if max_eps_height < next_height:
                max_eps_height = next_height 
            
            
            
            #++++++++++modified reward++++++++++++
            reward += (3/100)*(np.absolute(new_state[0][4])/(4*np.pi))
            
            # Assess whether terminal state was reached.
            # The episode may end due to having reached 200 steps, but we should not regard this as reaching the terminal state, and hence not disregard Q(s',a) from the Q target.
            # https://arxiv.org/abs/1712.00378
            nonterminal_to_buffer = not finish_episode or steps == 100
            
            # Store experienced transition to replay buffer
            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

            state = new_state
            ep_reward += reward
            
            # If replay buffer contains more than 1000 samples, perform one training step
            if replay_buffer.buffer_length > 1000:
                loss = sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma)
                ddqn.optimizer.zero_grad()
                loss.backward()
                ddqn.optimizer.step()

                cnt_updates += 1
                if cnt_updates % tau == 0:
                    ddqn.update_target_network()
                
        eps = max(eps - eps_decay, eps_end) # decrease epsilon        
        R_buffer.append(ep_reward)
        
        # Running average of episodic rewards (total reward, disregarding discount factor)
        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i-1]) if i > 0 else R_avg.append(R_buffer[i])
        max_height_history.append(max_eps_height)

        print('Episode: {:d}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Avg Q: {:.4g}'.format(
            i, ep_reward, R_avg[-1], eps, np.mean(np.array(q_buffer))))
        
        # If running average > 195 (close to 200), the task is considered solved
        if R_avg[-1] > -250:
            return R_buffer, max_height_history
    return R_buffer, max_height_history

def calc_q_and_take_action(ddqn, state, eps):
    '''
    Calculate Q-values for current state, and take an action according to an epsilon-greedy policy.
    Inputs:
        ddqn   - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        state  - Current state. Numpy array, shape (1, num_states).
        eps    - Exploration parameter.
    Returns:
        q_online_curr   - Q(s,a) for current state s. Numpy array, shape (1, num_actions) or  (num_actions,).
        curr_action     - Selected action (0 or 1, i.e. left or right), sampled from epsilon-greedy policy. Integer.
    '''
    # FYI:
    # ddqn.online_model & ddqn.offline_model are Pytorch modules for online / offline Q-networks, which take the state as input,
    # and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).

    # YOUR CODE HERE
    state = torch.from_numpy(state).float()
    q_online_curr = ddqn.online_model(state)
    curr_action = np.random.choice( np.arange(len(q_online_curr[0])), p = eps_greedy_policy(q_online_curr[0].detach().cpu().numpy(), eps))
    return q_online_curr.detach().cpu().numpy(), curr_action

def calculate_q_targets(q1_batch, q2_batch, r_batch, nonterminal_batch, gamma=.99):
    '''
    Calculates the Q target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network. FloatTensor, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network. FloatTensor, shape (N, num actions)
    : param r_batch: Batch of rewards. FloatTensor, shape (N,)
    : param nonterminal_batch: Batch of booleans, with False elements if state s' is terminal and True otherwise. BoolTensor, shape (N,)
    : param gamma: Discount factor, float.
    : return: Q target. FloatTensor, shape (N,)
    '''
    
    # YOUR CODE HERE
    best_actions = np.argmax(q1_batch.detach().cpu().numpy(), axis=1)
    Y = r_batch + nonterminal_batch.float() * gamma * q2_batch[np.arange(q2_batch.shape[0]), best_actions]

    return Y


def test_run(env, model, n_episodes, n_steps, fps=20, visualization=False):
    #setupvariables
    device = torch.device("cpu")
    fps = 20
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
            #test by using Q-value from offline network (target network)
            Q = model.offline_model(torch.tensor(state, dtype=torch.float, device=device)).cpu().detach().numpy()
            action = np.argmax(Q)
            state, reward, done, _ = env.step(action)
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