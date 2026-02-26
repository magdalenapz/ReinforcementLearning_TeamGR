# based on https://github.com/sfujim/TD3/blob/master/utils.py
import numpy as np
import torch

# Implementation of PER extension from https://arxiv.org/pdf/1511.05952
class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu", alpha=0.6):
        self.max_size = max_size
        # added parameter alpha for PER, determining strength of prioritization
        self.alpha = alpha
        self.ptr = 0
        self.size = 0
        self.device = device

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        # added parameter prioritied for PER
        self.priorities = np.zeros(max_size)


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        # new transitions get maximal priority
        self.priorities[self.ptr] = self.priorities.max() if self.size > 0 else 1.0

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, beta=0.4):
        # Sampling probability P(i) = p_i^{alpha} / sum_j p_j^{alpha}
        p_i = self.priorities if self.size == self.max_size else self.priorities[:self.ptr]
        p_i_alpha = p_i ** self.alpha
        P_i = p_i_alpha / p_i_alpha.sum()

        ind = np.random.choice(len(P_i), batch_size, p=P_i)
        # weights = ((1/N * 1/P(i))^beta) / weights_max
        weights = ((1/len(P_i)) * (1/P_i[ind])) ** beta
        weights = weights / weights.max()
    
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            torch.FloatTensor(weights).to(self.device)
        )

    # update priorities
    def update_priorities(self, ind, td_errors):
        self.priorities[ind] = np.abs(td_errors) + 0.000001
