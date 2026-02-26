import hockey.hockey_env as h_env

import numpy as np

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



def evaluate(
    agent,
    custom_actions=custom_actions,
    max_episodes=50, 
    max_steps=250,
    render_human= False,
    opponent_weak= True
):
    stats = []
    wins = 0
    losses = 0
    draws = 0
    timeouts = 0
    basic_opponent = h_env.BasicOpponent(weak=opponent_weak)
    winner = None

    env = h_env.HockeyEnv()

    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    _ = env.render()

    for i in range(max_episodes):
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            
            #rendering
            if render_human:
                env.render(mode="human")

            a1_discrete = agent.act(ob, eps=0)
            a1_continuous = custom_actions[a1_discrete]
            
            ob_opponent = env.obs_agent_two()
            a2_continuous = basic_opponent.act(ob_opponent)
            
            
            observation_new, reward, done, trunc, info = env.step(np.hstack([a1_continuous, a2_continuous]))    
            obs_agent2 = env.obs_agent_two()
            
            total_reward+= reward
            ob = observation_new
            
            if done or trunc:
                winner = info["winner"]
                if trunc:
                    timeouts += 1
                break
        
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        elif winner == 0:
            draws += 1
        
        stats.append([i, total_reward, t+1])
        if ((i-1) % 20 == 0):
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))
    return stats, wins, losses, draws, timeouts