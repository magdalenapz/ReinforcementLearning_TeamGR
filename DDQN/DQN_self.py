import hockey.hockey_env as h_env

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


import torch
import numpy as np

#copied this over for easier readability
class Memory():
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dueling: bool = True):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        
        # changed to relu
        self.activations = [ torch.nn.ReLU() for l in  self.layers ]
        self.dueling = dueling
        

        #changes for dueling DQN
        if dueling:
            self.value_stream = torch.nn.Linear(self.hidden_sizes[-1], 1)
            self.advantage_steam = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        else:
            self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        #print(x.dim(), x.shape)
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        
        #changes for dueling DQN
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_steam(x)
            q_value = value + (advantage - advantage.mean(dim = 1, keepdim = True))
            #print("value: ", value)
            #print("advantage: ", advantage)
            #print("q_value: ", q_value)
        else:
            q_value = self.readout(x)
        

        return q_value

    def predict(self, x):
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32))
            #had problems with dimentions, this fixes it
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            return self.forward(x_t).numpy()

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[256,256], 
                 learning_rate = 0.0002, dueling=True):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, 
                         output_size=action_dim, dueling=dueling)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, observations, actions, targets):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        #changed
        acts = torch.from_numpy(actions).long()
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:,None])        
    
    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)
        
    def greedyAction(self, observations):
        q = self.predict(observations)  # (B, A) or (1, A)
        a = np.argmax(q, axis=-1)       # (B,) or (1,)
        if a.size == 1:
            return int(a.item())        # scalar
        return a
    
class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, observation_space, action_space, **userconfig):
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "eps": 0.05,            # Epsilon in epsilon greedy policies                        
            "discount": 0.95,
            "buffer_size": int(5e5),
            "batch_size": 64,
            "learning_rate": 0.0001,
            "update_target_every": 1000,
            "use_target_net":True,
            "dueling": True,
            "double": True
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        
        self.buffer = Memory(max_size=self._config["buffer_size"])
                
        # Q Network
        self.Q = QFunction(observation_dim=self._observation_space, 
                           action_dim=self._action_n,
                           learning_rate = self._config["learning_rate"],
                           dueling = self._config["dueling"])
        # Q Network
        self.Q_target = QFunction(observation_dim=self._observation_space, 
                                  action_dim=self._action_n,
                                  learning_rate = 0,
                                  dueling = self._config["dueling"])
        self._update_target_net()
        self.train_iter = 0
    
    # von mir hinzugefügt
    def save(self, filename):
        # Speichert die Gewichte von Q, Q_target und dem Optimizer
        torch.save({
            'config': self._config,
            'q_state_dict': self.Q.state_dict(),
            'q_target_state_dict': self.Q_target.state_dict(),
            'optimizer_state_dict': self.Q.optimizer.state_dict(),
        }, filename)
    def load(self, filename):
        # Lädt alles wieder in den Agenten
        checkpoint = torch.load(filename)
        self.Q.load_state_dict(checkpoint['q_state_dict'])
        self.Q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.Q.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def _update_target_net(self):        
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()        
        return action
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
            
    def train(self, iter_fit=32):
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()                
        for i in range(iter_fit):

            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            # changed
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            
            if self._config["use_target_net"]:
                if self._config["double"]:
                    
                    # CHANGED: changes to use Double Q-learning
                    # action with highest Q-Value from the current Q network instead of target
                    a_highest = self.Q.greedyAction(s_prime)

                    #conversion to tensor
                    s_prime_t = torch.as_tensor(s_prime, dtype=torch.float32)
                    a_highest_t = torch.as_tensor(a_highest, dtype=torch.int64)

                    v_prime = self.Q_target.Q_value(s_prime_t,a_highest_t).detach().numpy()
                else:    
                    v_prime = self.Q_target.maxQ(s_prime)
            else:
                v_prime = self.Q.maxQ(s_prime)
            # target
            gamma=self._config['discount']                                                
            td_target = rew + gamma * (1.0-done) * v_prime
            
            fit_loss = self.Q.fit(s, a, td_target)
            
            losses.append(fit_loss)
                
        return losses