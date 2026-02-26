import numpy as np
import torch
import hockey.hockey_env as h_env

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
# new
class PolicyBuffer(object):
	def __init__(self):
		p1 = h_env.BasicOpponent(weak=True)
		p1.select_action = p1.act
		self.policies = [p1]
		self.score_matrix = np.full((1, 1), 0.5)
		p2 = h_env.BasicOpponent(weak=False)
		p2.select_action = p2.act
		self.add(p2)

	def calculate_elos(self):    
		n = len(self.score_matrix)
		equations = []; values = []
		
		# 2. Build the system of linear equations
		for i in range(n):
			for j in range(i + 1, n):
				# Calculate expected rating difference using the inverted Elo formula
				score = np.clip(self.score_matrix[i, j], 0.001, 0.999)
				diff_ij = 400 * np.log10(score / (1 - score))
				
				# Create an equation row: 1 * R_i - 1 * R_j = diff_ij
				row = np.zeros(n)
				row[i] = 1; row[j] = -1
				
				equations.append(row)
				values.append(diff_ij)
		
		# 4. Solve using Numpy's Least Squares solver
		A = np.array(equations); b = np.array(values)
		elos, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
		
		self.elos = elos - elos[0]

	def add(self, policy, num_games=500):
		self.score_matrix = np.pad(self.score_matrix, ((0, 1), (0, 1)), 
							 mode='constant', constant_values=0.5)
		env = h_env.HockeyEnv()
		for o, opponent in enumerate(self.policies):
			score = 0
			for i in range(num_games):
				state, info = env.reset()
				done = False
				while not done:
					action1 = policy.select_action(np.array(state))
					mirrored_state = env.obs_agent_two()
					action2 = opponent.select_action(np.array(mirrored_state))
					action = np.hstack([action1, action2])
					state, reward, terminated, truncated, info = env.step(action)
					done = terminated or truncated
				score += env._compute_reward()/20 + 0.5
			self.score_matrix[-1, o] = score/num_games
			self.score_matrix[o, -1] = 1 - score/num_games
		self.policies.append(policy)

		self.calculate_elos()
	
	def sample(self, heat=400):
		points = torch.tensor(self.elos/heat)
		probs = torch.softmax(points, dim=0)
		idx = torch.multinomial(probs, num_samples=1).item()
		return self.policies[idx]