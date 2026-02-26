import os
import sys
import torch
import numpy as np
import hockey.hockey_env as h_env
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td3per.td3 import TD3_PER
from td3per.utils import PrioritizedReplayBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"TD3_PER Training Device: {device.upper()}", flush=True)

beta_start = 0.4
beta_end = 1.0
max_episodes = 2000000
start = 25000
batch = 256
exploration_noise = 0.2
save_interval = 1000
snapshot_interval = 5000

save_dir = "TD3/agents/td3_per_neu"
json_dir = "TD3PER/jsons"
log_path = os.path.join(json_dir, "td3_per_sparse_stats.json")
os.makedirs(save_dir, exist_ok=True)

class BasicOpponentWrapper:
    def __init__(self, weak=True):
        self.opponent = h_env.BasicOpponent(weak=weak)
    def select_action(self, obs):
        return self.opponent.act(obs)

def compute_sparse_reward(winner, agent_won, agent_lost):
    if agent_won:
        return 10.0
    elif agent_lost:
        return -10.0
    elif winner == 0:
        return -5.0
    else:
        return -1.0

def train_sparse_td3_per():

    env = h_env.HockeyEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    max_action = float(env.action_space.high[0])

    agent = TD3_PER(state_dim, action_dim, max_action, device=device)
    agent.load("TD3PER/agents/td3_per_neu/td3_per_sparse")
    buffer = PrioritizedReplayBuffer(state_dim, action_dim, max_size=1000000, device=device, alpha=0.6)

    opponents = {
        "strong": BasicOpponentWrapper(weak=False),
        "weak": BasicOpponentWrapper(weak=True)
    }

    total_steps = 0
    episode = 0
    stats = {"wins": 0, "losses": 0, "draws": 0}
    episode_lengths = []
    episode_rewards = []
    win_rates = []
    draw_rates = []
    avg_rewards = []
    betas = []
    prio_mins = []
    prio_maxs = []
    episodes_logged = []

    while episode < max_episodes:
        if np.random.rand() < 0.5:
            opponent, opp_name = opponents["strong"], "strong"
        else:
            opponent, opp_name = opponents["weak"], "weak"

        state, _ = env.reset()
        player_red = np.random.choice([True, False])
        trajectory = []

        done = False
        truncated = False
        step = 0
        ep_reward = 0.0

        while not (done or truncated):
            total_steps += 1
            step += 1

            if player_red:
                obs_agent = np.array(state)
                obs_opp = env.unwrapped.obs_agent_two()
            else:
                obs_agent = env.unwrapped.obs_agent_two()
                obs_opp = np.array(state)

            action_agent = (
                agent.select_action(obs_agent) +
                np.random.normal(0, max_action * exploration_noise, size=4)
            ).clip(-max_action, max_action)

            action_opp = opponent.select_action(obs_opp)

            action_env = (
                np.hstack([action_agent, action_opp])
                if player_red else np.hstack([action_opp, action_agent])
            )

            next_state, reward, done, truncated, info = env.step(action_env)

            next_obs_agent = np.array(next_state) if player_red else env.unwrapped.obs_agent_two()
            trajectory.append((obs_agent, action_agent, next_obs_agent, done))

            state = next_state
            ep_reward += reward

        winner = info.get("winner", 0)
        agent_won = (winner == 1) if player_red else (winner == -1)
        agent_lost = (winner == -1) if player_red else (winner == 1)
        terminal_reward = compute_sparse_reward(winner, agent_won, agent_lost)

        for i, (obs, act, next_obs, d) in enumerate(trajectory):
            reward_to_use = terminal_reward if i == len(trajectory) - 1 else 0.0
            buffer.add(obs, act, next_obs, reward_to_use, d)
        beta = beta_start + (beta_end - beta_start) * min(1.0, episode / (0.8 * max_episodes))
        if buffer.size > start:
            for _ in range(10):
                agent.train(buffer, batch, beta=beta)

        if winner == 0:
            stats["draws"] += 1
            outcome = "D"
        elif agent_won:
            stats["wins"] += 1
            outcome = "W"
        else:
            stats["losses"] += 1
            outcome = "L"

        episode += 1
        episode_lengths.append(step)
        episode_rewards.append(ep_reward)

        if episode % 10 == 0:
            total = stats["wins"] + stats["losses"] + stats["draws"]
            wr = stats["wins"] / total * 100 if total > 0 else 0
            dr = stats["draws"] / total * 100 if total > 0 else 0
            avg_rew = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            prios = buffer.priorities[:buffer.size]
            prio_min = np.min(prios)
            prio_max = np.max(prios)
            episodes_logged.append(episode)
            win_rates.append(wr)
            draw_rates.append(dr)
            avg_rewards.append(avg_rew)
            betas.append(beta)
            prio_mins.append(float(prio_min))
            prio_maxs.append(float(prio_max))
            print(f"Ep {episode} | Opp: {opp_name[:8]} | {outcome} | W:{wr:.0f}% D:{dr:.0f}% | Beta:{beta:.3f} | AvgRew:{avg_rew:.2f} | Prio[min/max]: {prio_min:.3f}/{prio_max:.3f}", flush=True)
            log_entry = {
                "episode": episode,
                "total_steps": total_steps,
                "win_rate": wr,
                "draw_rate": dr,
                "avg_reward_last10": float(avg_rew),
                "beta": float(beta),
                "prio_min": prio_min,
                "prio_max": prio_max
            }
        
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        if episode % save_interval == 0:
            agent.save(f"{save_dir}/td3_per_sparse")
            print(f"--- Saved at {episode} in {save_dir} ---", flush=True)

    print(f"Training complete. Wins:{stats['wins']} Losses:{stats['losses']} Draws:{stats['draws']}")

if __name__ == "__main__":
    train_sparse_td3_per()