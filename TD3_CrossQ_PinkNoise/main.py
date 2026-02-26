import numpy as np
import torch
import gymnasium as gym # old: import gym
import hockey.hockey_env as h_env # new
from pink import PinkNoiseProcess # new
import copy # new

import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import CrossQ
import CrossQhalf

# new
def create_env(env_name):
	if env_name == "Hockey-Weak":
		return gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)
	elif env_name == "Hockey-Strong":
		return gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=False)
	else: return gym.make(env_name)

# new: removed all seeds

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=10):# old: , seed, eval_episodes=10):
	# old: eval_env = gym.make(env_name)
	eval_env = create_env(env_name)
	# old: eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		# old: state, done = eval_env.reset(), False
		state, info = eval_env.reset()#seed = seed + 100)
		done = False
		while not done:
			action = policy.select_action(np.array(state))
			# old: state, reward, done, _ = eval_env.step(action)
			state, reward, terminated, truncated, info = eval_env.step(action)
			done = terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Hockey-Weak")             # OpenAI gym environment name
	#parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="") 				# Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--pink_noise", action="store_true")        # new
	#parser.add_argument("--r_b_size", default=1e6, type=int)        # new
	#parser.add_argument("--p_b_size", default=10, type=int)         # new
	#parser.add_argument("--name", default="")						# new
	parser.add_argument("--policy_buffer", default="")				# new
	parser.add_argument("--heat", default=200, type=int)			# new
	parser.add_argument("--iteration", default=0, type=int)			# new
	parser.add_argument("--eval_episodes", default=10, type=int)	# new
	args = parser.parse_args()

	#if args.name != "": file_name = args.name else: # new 
	file_name = f"{args.env}_{args.policy}_pink={args.pink_noise}_i={args.iteration}" #_seed={args.seed}" #_r_b_size={args.r_b_size}"#_p_b_size={args.p_b_size}"

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}") #, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	# new: Allow Hockey Env
	# old: env = gym.make(args.env)
	if args.env == "Hockey-SelfPlay":
		env = h_env.HockeyEnv()
	else: env = create_env(args.env)

	# Set seeds
	# old: env.seed(args.seed)
	#env.action_space.seed(args.seed)
	#torch.manual_seed(args.seed)
	#np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	if args.env == "Hockey-SelfPlay": action_dim //= 2 # new
	max_action = float(env.action_space.high[0])

	# new
	create_policy = True
	if args.env == "Hockey-SelfPlay":
		policy_buffer = torch.load(f"./models/{args.policy_buffer}", weights_only=False)#, map_location='cpu')
		best_policy = policy_buffer.policies[np.argmax(policy_buffer.elos)]
		if not isinstance(best_policy, h_env.BasicOpponent):
			policy = copy.deepcopy(best_policy)
			create_policy = False
	
	if create_policy: # new
		kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		}
		# Initialize policy # new: switched order to reuse args
		if args.policy == "DDPG":
			policy = DDPG.DDPG(**kwargs)
		elif args.policy == "OurDDPG":
			policy = OurDDPG.DDPG(**kwargs)
		else:
			# Target policy smoothing is scaled wrt the action scale
			kwargs["policy_noise"] = args.policy_noise * max_action
			kwargs["noise_clip"] = args.noise_clip * max_action
			kwargs["policy_freq"] = args.policy_freq
			if args.policy == "TD3":
				policy = TD3.TD3(**kwargs)
			elif args.policy == "CrossQ":
				policy = CrossQ.TD3(**kwargs)
			elif args.policy == "CrossQhalf":
				policy = CrossQhalf.TD3(**kwargs)

		if args.load_model != "":
			policy_file = file_name if args.load_model == "default" else args.load_model
			policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim) #, int(args.r_b_size)) # new: size argument
	
	# Evaluate untrained policy
	if args.env == "Hockey-SelfPlay": # new
		evals1 = [eval_policy(policy, "Hockey-Weak")]#, args.seed)]
		evals2 = [eval_policy(policy, "Hockey-Strong")]#, args.seed)]
	else: evaluations = [eval_policy(policy, args.env)]#, args.seed)]

	# old: state, done = env.reset(), False
	state, info = env.reset()#seed=args.seed)
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# new
	if args.pink_noise:
		noise_process = PinkNoiseProcess((action_dim, 1000), max_action * args.expl_noise)
	if args.env == "Hockey-SelfPlay": opponent = policy_buffer.sample(heat=args.heat) # new


	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		#if t < args.start_timesteps:
		#	action = env.action_space.sample()
		#else:
		# new
		if args.pink_noise: noise = noise_process.sample()
		else: noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			
		action = (
			policy.select_action(np.array(state))
			# old: + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			+ noise
		).clip(-max_action, max_action)

		# new
		if args.env == "Hockey-SelfPlay":
			mirrored_state = env.obs_agent_two()
			other_action = opponent.select_action(np.array(mirrored_state))
			action = np.hstack([action, other_action])

		# Perform action
		# old: next_state, reward, done, _ = env.step(action)
		next_state, reward, terminated, truncated, info = env.step(action)
		done = terminated or truncated
		# old: done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		done_bool = float(terminated)

		# Store data in replay buffer
		if args.env == "Hockey-SelfPlay": # new
			replay_buffer.add(state, action[:action_dim], next_state, reward, done_bool)
		else:
			replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			#print(info['winner'])
			# Reset environment
			# old: state, done = env.reset(), False
			state, info = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			if args.pink_noise: noise_process.reset() # new
			if args.env == "Hockey-SelfPlay": opponent = policy_buffer.sample(heat=args.heat) # new

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			if args.env == "Hockey-SelfPlay": # new
				evals1.append(eval_policy(policy, "Hockey-Weak"))#, args.seed))
				evals2.append(eval_policy(policy, "Hockey-Strong"))#, args.seed))
				np.save(f"./results/{file_name}_Hockey-Weak.npy", evals1)
				np.save(f"./results/{file_name}_Hockey-Strong.npy", evals2)
			else:
				evaluations.append(eval_policy(policy, args.env, args.eval_episodes))#, args.seed))
				np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")

	if args.env == "Hockey-SelfPlay": # new
		policy_buffer.add(policy)
		torch.save(policy_buffer, f"./models/{args.policy_buffer}")