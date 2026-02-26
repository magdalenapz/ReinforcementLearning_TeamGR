import numpy as np
import matplotlib.pyplot as plt
import torch

dir = '/home/arneg/Schreibtisch/Reinforcement Learning/Project/Code/TD3-paper/results/'
#dir = '/home/arneg/Schreibtisch/Reinforcement Learning/Project/Backup/results/new/'
env = "Hockey-Strong"
models = ['_TD3_pink=False', '_TD3_pink=True', '_CrossQ_pink=True', '_CrossQhalf_pink=True']
labels = ['TD3 baseline', 'TD3 with pink noise', 'CrossQ (+ pink)', 'CrossQhalf (+ pink)']

for model, label in zip(models, labels):
    min = 0; max = 40
    all_runs = np.empty((0, 21))
    for i in range(min, max):
        try:
            data = np.load(f'{dir}{env}{model}_i={i}.npy')
            if data.shape[0] == 21: all_runs = np.vstack((all_runs, data))
        except FileNotFoundError:
            print(f"File for i={i} not found, skipping.")

    print(len(all_runs))

    mean_performance = np.mean(all_runs, axis=0)
    std_performance = np.std(all_runs, axis=0)
    x_axis = np.arange(mean_performance.shape[0])

    plt.plot(x_axis, mean_performance, label=label)
    plt.fill_between(x_axis, mean_performance - std_performance, mean_performance + std_performance, alpha=0.3)
    
plt.xlabel('evaluation round')
plt.ylabel('total reward per episode')
plt.xticks(np.arange(0, 21, 2))
plt.legend()
plt.show()

# OUTLIER: i=3 CrossQhalf