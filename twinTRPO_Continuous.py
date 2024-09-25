'''
This file provides a source code for twinTRPO in continuous action space, whose reference is:
[1] Haotian Xu, Junyu Xuan, Guangquan Zhang, Jie Lu. Twin Trust Region Policy Optimization.
submitted to IEEE Transactions on Transactions on Systems, Man and Cybernetics: Systems, 2024.
[2] Haotian Xu, Junyu Xuan, Guangquan Zhang, Jie Lu. Reciprocal trust region policy optimization.
In: World Scientific Proceedings Series on Computer Engineering and Information Science --
Intelligent Management of Data and Information in Decision Making, pp. 187-194,2024.

twinTRPO aggregates TRPO and rTRPO, which has an upper bound and a lower bound of step size,
    and a least objective increments.

Author emails: Haotian.Xu@uts.student.edu.au(H. Xu), Junyu.Xuan@uts.edu.au(J. Xuan),
               Guangquan.Zhang@uts.edu.au (G. Zhang), Jie.Lu@uts.edu.au (J. Lu)
               
If any problem, please concact the primary author.

'''
import os
os.add_dll_directory("C://Users//DELL//.mujoco//mujoco210//bin")
import time
import torch
import numpy as np
import gym
from   tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy

#We list 13 benchmark environments with continuous from classic control and Mujoco
ENV_LIST=[
    #classic control
    'Pendulum-v1',               #(0) states=3, action=1 in (-2,2)
    'MountainCarContinuous-v0',  #(1) states=2, action=1 in (-1,1)
    #Mujoco
    'InvertedDoublePendulum-v2', #(2) states=11,  action=1 in (1.0,1.0)
    'InvertedPendulum-v2',       #(3) states=4,   action=1	in (-3.0,3.0)
    'Reacher-v2',                #(4) states=11,  actions=2 in (-1.0,1.0)
    'Swimmer-v2',                #(5) states=8,   actions=2 in (-1.0,1.0)
    'Hopper-v2',                 #(6) states=11,  actions=3 in (-1.0,+1.0)
    'HalfCheetah-v2',            #(7) states=17,  actions=6 in (-1.0,+1.0)
    'Walker2d-v2',               #(8) states=17,  actions=6 in (-1.0,1.0)
    'Pusher-v2',                 #(9) states =23 action=7 in (-2.0,2.0)
    'Ant-v2',                    #(10) states=27,  actions=8 in (-1.0,+1.0)
    'Humanoid Standup-v2',       #(11) states=376, actions=17 in (-0.4,0.4)
    'Humanoid-v2'                #(12) states=376，actions=17 in (-0.4,0.4)    
    ]


#Please pick up an enviornment
env_id = 0
env_name = ENV_LIST[env_id]

#Some parameter settings:
num_episodes = 5500      #the number of episodes

hidden_dim = 128         #the number of neurons in the hidden layers (64 or 128)

gamma = 0.90             #discount factor(0.90)
lamda = 0.90             #a parameter for computing generalized advantage estimation(0.9)
kl_constraint = 5.0e-5   #KL constraint threshold
obj_constraint = 1.0e-3  #objective increment threshold
alpha = 0.5              #a paramter for step size search

critic_lr = 1.0e-3       #learning rate for critic

rndseed = 0              #random number
win_size = 21            #window size for smoothing return

agent_name ='twinTRPO'

#=======================================================================================================
# To train an on-policy agent
# Input: 
#    env - environment
#    agent -- agent
#    num_episodes -- the number of epsiodes
#    max_horizon -- the maximal length of trajectory
# Output:
#    return_list: a list of returns for each episode
#    time_list: a list of accumulative running time
#    step_List: a list of accumulative steps
#------------------------------------------------------------------------------
def train_an_onpolicy_agent(env, agent, num_episodes, max_horizon=1e5):
    return_list = []
    time_list = [] 
    step_list =[]
    t0 = time.time()
    total_steps = 0
    
    for i in range(10): 
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar: 
            for i_episode in range(int(num_episodes/10)): 
                
                transition_dict = {'states': [], 'actions': [], 'next_states': [],\
                                   'rewards': [], 'dones': []}
                state = env.reset()
                
                done = False
                episode_return = 0
                episode_horizon = 0
                
                # to sample a trajectory
                while (not done) and (episode_horizon <= max_horizon):
                    action = agent.sample_an_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    episode_horizon +=1
                    total_steps += 1
                   
                return_list.append(episode_return)
                time_list.append(time.time() - t0)
                step_list.append(total_steps)
                
                # to train the agent
                agent.update_agent(transition_dict)
                
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                                      'steps': '%d' %total_steps,
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    
    return return_list, time_list, step_list
#===========================================================================================
# To smooth return values
# Input: x-- a list of returns; win_size-- smoothing window size
# Output: a list of smoothing returns
#------------------------------------------------------------------------------
def moving_average(x, wind_size):
    cumulative_sum = np.cumsum(np.insert(x, 0, 0)) 
    middle = (cumulative_sum[win_size:] - cumulative_sum[:-win_size]) / win_size
    r = np.arange(1, win_size-1, 2)
    begin = np.cumsum(x[:win_size-1])[::2] / r
    end = (np.cumsum(x[:-win_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
#==========================================================================================
# To initialize a neural network with a uniform distribution
#------------------------------------------------------------------------------
def init_net_weights(m):
   
    if type(m)==torch.nn.Linear:
        r=0.01
        torch.nn.init.uniform_(m.weight, a=-r, b=r)
        
        #torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        
        m.bias.data.fill_(0.00)
#=========================================================================================
# To compute the generalized advantage estimation (GAE)
# Input:
#      gamma -- discount factor
#      lmbda -- a paramter for GAE
#      td_delta -- temporal difference 
# Output: a list for advantage function 
#------------------------------------------------------------------------------    
def estimate_advantage(gamma, lamda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lamda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

#========================================================================================
# Define a value net with two hidden layers
# Input: 
#    state-dim -- the dimensionality of state space
#    hidden_dim -- the number of neurons for hidden layers
# Output: value function
#------------------------------------------------------------------------------
class ValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.fout = torch.nn.Linear(hidden_dim, 1)
        
        self.apply(init_net_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fout(x)
    
#=========================================================================================
# To define a policy net with two hidden layers
# Input: 
#    state-dim -- the dimensionality of state space
#    hidden_dim -- the number of neurons for hidden layers
#    action_dim -- the dimensionality of action space
#    action_bound -- action bound
# Output:
#    mean and standard deriation for each action
#------------------------------------------------------------------------------
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        
        self.apply(init_net_weights)
        
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std  

#======================================================================================
# To define our twinTRPO agent
class twinTRPO:
    """ twinTRP for continuous action"""
    #---------------------------------------------------------------------------
    def __init__(self, hidden_dim, state_space, action_space, lamda,
                 kl_constraint, obj_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        action_bound = action_space.high[0]
        
        print('State dimension  = ', state_dim)
        print('Action dimension = ', action_dim)
        print('Action bound     = ', action_bound)
        print('-----------------------------------')
        
        
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim, action_bound).to(device)
        self.critic = ValueNetContinuous(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        
        self.gamma = gamma
        self.lamda = lamda
        self.kl_constraint = kl_constraint
        self.obj_constraint = obj_constraint
        self.alpha = alpha
        self.device = device
    
    #--------------------------------------------------------------------------
    # To sample an action
    def sample_an_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        
        return list(np.ravel(action.tolist()))  
    
    #--------------------------------------------------------------------------
    # to compute the product of a Hessian matrix and a vector
    def hessian_matrix_vector_product(self, states, old_action_dists,
                                      vector, damping=0.1):
        mu, std = self.actor(states)
        
        new_action_dists = torch.distributions.Normal(mu, std)
        
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        
        return grad2_vector + damping * vector
    
    #--------------------------------------------------------------------------
    # To solve a set of linear equations with conjugate descent method
    def conjugate_gradient_method(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr < 1e-10:
                break
            
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    #--------------------------------------------------------------------------
    # To compute the surrogate objective function
    def compute_surrogate_objective(self, states, actions, advantage, old_log_probs,
                              actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)
    
    #--------------------------------------------------------------------------
    # To execute a purly linear search for step size
    def pure_line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, dec_vec, min_coef, max_coef):
        
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        
        old_obj = self.compute_surrogate_objective(states, actions, advantage,\
                                             old_log_probs, self.actor)
        
        if (min_coef >= max_coef): #lower bound > upper bound
            #To check the surrogate objective at the lower bound
            coef = min_coef
            new_para = old_para + coef * dec_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            
            new_obj = self.compute_surrogate_objective(states, actions, advantage,
                                             old_log_probs, new_actor)
            
            if (new_obj > old_obj): 
                old_para = new_para
                old_obj = new_obj
            
            #To check the surrogate objective at the upper bound
            coef = max_coef
            new_para = old_para + coef * dec_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_objective(states, actions, advantage,
                                             old_log_probs, new_actor)
            if (new_obj > old_obj): 
                old_para = new_para
                old_obj = new_obj
            
            return old_para
                
        else: #upper bound > lower bound
            nsearch = 15
            dcoef = (max_coef - min_coef)/(nsearch-1)
            for i in range(nsearch):  
                coef = max_coef - i*dcoef
                new_para = old_para + coef * dec_vec
                new_actor = copy.deepcopy(self.actor)
                torch.nn.utils.convert_parameters.vector_to_parameters(
                    new_para, new_actor.parameters())
                
                mu, std = new_actor(states)
                new_action_dists = torch.distributions.Normal(mu, std)
                
                kl_div = torch.mean(
                    torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
                new_obj = self.compute_surrogate_objective(states, actions, advantage,
                                                 old_log_probs, new_actor)
                
                if new_obj > old_obj and kl_div < self.kl_constraint:
                    old_para = new_para
                    old_obj = new_obj
                
            return old_para
    
    #---------------------------------------------------------------------------
    # To train policy net
    def train_policy(self, states, actions, old_action_dists, old_log_probs,
                     advantage):
        
        surrogate_obj = self.compute_surrogate_objective(states, actions, advantage,
                                                   old_log_probs, self.actor)
        
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        
        descent_direction = self.conjugate_gradient_method(obj_grad, states,
                                                    old_action_dists)
        
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        
        max_coef = torch.sqrt(2 * self.kl_constraint / \
                              (torch.dot(descent_direction, Hd) + 1e-8))
        
        min_coef = self.obj_constraint * torch.abs(surrogate_obj)/ \
                              (torch.dot(descent_direction, Hd) + 1e-8)
        
        new_para = self.pure_line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists, descent_direction,
                                    min_coef, max_coef)
        
        torch.nn.utils.convert_parameters.vector_to_parameters(\
            new_para, self.actor.parameters())

    #--------------------------------------------------------------------------
    # To update actor and critic nets
    def update_agent(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)
        advantage = estimate_advantage(self.gamma, self.lamda,
                                      td_delta.cpu()).to(self.device)
        
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      std.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.train_policy(states, actions, old_action_dists, old_log_probs,
                          advantage)

#==========================================================================================
# To run a training procedure with a fixed random number and draw a figure
#------------------------------------------------------------------------------
def main_one_seed():
    
    
    env = gym.make(env_name)
    env.seed(rndseed)
    torch.manual_seed(rndseed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    
    agent = twinTRPO(hidden_dim, env.observation_space, env.action_space,
                       lamda, kl_constraint, obj_constraint, alpha, critic_lr, gamma, device)
    
    return_list,time_list,step_list = train_an_onpolicy_agent(env, agent, num_episodes)
    
    env.close()
    
    episodes_list = list(range(len(return_list)))
    
    mv_return = moving_average(return_list, win_size)

    fig_title = '-'.join([agent_name, env_name])

    plt.plot(episodes_list, return_list, color='b', label='Raw return')
    plt.plot(episodes_list, mv_return, color='r', label='Smoothed return')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.rcParams['savefig.dpi'] = 600
    plt.title(fig_title)
    plt.legend()
    plt.show()

#==========================================================================================
if __name__ == '__main__':

    print('Agent = ', agent_name)
    print('Env = ', env_name)
    
    main_one_seed()
    
    print("end")
#----------------------------------------------------------------------------