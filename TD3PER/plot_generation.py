import json
import matplotlib.pyplot as plt
import json
import numpy as np

episodes = []
win_rates = []
draw_rates = []

with open("TD3PER/jsons/td3_per_sparse_stats.json", "r") as f:
    for line in f:
        data = json.loads(line)
        episodes.append(data["episode"])
        win_rates.append(data["win_rate"])
        draw_rates.append(data["draw_rate"])


# Win/Draw Rate plot
plt.figure()
plt.plot(episodes, win_rates, label="Win Rate")
plt.plot(episodes, draw_rates, label="Draw Rate")
plt.xlabel("Episode")
plt.ylabel("Rate (%)")
plt.legend()
plt.title("Win and Draw Rate over Training")
plt.tight_layout()
plt.savefig("TD3PER/plots/win_draw_rate.png", dpi=300)
plt.close()
print("win draw rate plot saved successfully.")


# Agent against the basic opponents plot
with open("TD3PER/jsons/td3_per_results.json", "r") as f:
    data = json.load(f)

agent_name = list(data.keys())[0]
results = data[agent_name]

labels = list(results.keys())
win_rates = [results[label]["win_rate"] for label in labels]

x = np.arange(len(labels))
plt.figure(figsize=(5, 5))
plt.bar(x, win_rates, width=0.2, color="steelblue")
plt.axhline(y=55, color='red', linestyle='--', linewidth=2, label="55% Threshold")
plt.ylabel("Win Rate (%)")
plt.xticks(x, labels)
plt.ylim(0, 100)
plt.title(f"TD3 + PER – Win Rate against Basic Opponents")
plt.legend()
plt.tight_layout()
plt.savefig("TD3PER/plots/td3_per_winrate.png", dpi=300)
plt.close()
print("Win rate against the basics plot saved successfully.")