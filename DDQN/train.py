import numpy as np
from argparse import ArgumentParser
from DQN_self import DQNAgent
import hockey.hockey_env as h_env

from gymnasium import spaces

# increase possible actions: [x, y, angle, shoot]
custom_actions = [

    [0,0,0,0], # nothing
    [1,0,0,0], # move right
    [-1,0,0,0], # move left
    [0,1,0,0], # move up
    [0,-1,0,0], # move down
    [0,0,1,0], # turn left
    [0,0,-1,0], # turn right
    [0,0,0,1], # shoot
    [1,1,0,0], # move right and up
    [1,-1,0,0], # move right and down
    [-1,1,0,0], # move left and up
    [-1,-1,0,0], # move left and down
    [1,0,1,0], # move right and turn left
    [1,0,-1,0], # move right and turn right
    [-1,0,1,0], # move left and turn left
    [-1,0,-1,0], # move left and turn right
    [0,1,1,0], # move up and turn left
    [0,1,-1,0], # move up and turn right
    [0,-1,1,0], # move down and turn left
    [0,-1,-1,0] # move down and turn right
]

basic_opponent = h_env.BasicOpponent(weak=True)

env = h_env.HockeyEnv()
def train(
    agent,
    run_name,
    custom_actions=custom_actions,
    max_episodes=2000, 
    max_steps=250, 
    epsilon_start=1.0, 
    epsilon_end=0.05, 
    epsilon_decay=0.0005,
    start_training_after=1000,
    stats_checkpoint_path=None,
):
    epsilon = epsilon_start
    stats = []
    losses = []
    steps = 0
    obs_agent2 = env.obs_agent_two()
    for i in range(max_episodes):
        total_reward = 0
        ob, _info = env.reset()
        epsilon = max(epsilon - epsilon_decay, epsilon_end)
        for t in range(max_steps):
            #env.render(mode="human")
            #env.render()
            obs_agent2 = env.obs_agent_two()
            a1_discrete = agent.act(ob, eps=epsilon)
            a1_continuous = custom_actions[a1_discrete]

            a2_continuous = basic_opponent.act(obs_agent2)
        
            observation_new, reward, done, trunc, info = env.step(np.hstack([a1_continuous, a2_continuous]))    
            
            
            agent.store_transition((ob, a1_discrete, reward, observation_new, done))
            obs_agent2 = env.obs_agent_two()
            total_reward+= reward
            ob = observation_new
            steps +=1
            if steps > start_training_after and steps % 4 == 0:
                losses.extend(agent.train(1))
            if done or trunc: break
            
        stats.append([i,total_reward,t+1])
        if i % 5000 == 0 and stats_checkpoint_path is not None:
            np.save(stats_checkpoint_path, np.asarray(stats, dtype=float))
            agent.save(f"{run_name}_weights_ep{i}.pt")
        if ((i-1) % 20 == 0):
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))
    return stats, losses

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--run",  type=str, default="default")
    arguments = parser.parse_args()

    checkpoint_save_path = f"{arguments.run}_checkpoint_stats.npy"

    q_agent = DQNAgent(
        observation_space=env.observation_space.shape[0], 
        action_space = spaces.Discrete(len(custom_actions)), 
        discount=0.99, 
        eps=1, 
        use_target_net=True,
        dueling = arguments.dueling,
        double = arguments.double
    )
    stats, losses = train(agent=q_agent, max_episodes=arguments.episodes, run_name=arguments.run, stats_checkpoint_path=checkpoint_save_path)
    q_agent.save(f'{arguments.run}_weights.pt')

    np.save(f'{arguments.run}_stats.npy', np.asarray(stats, dtype=float))
    print("finished training")
