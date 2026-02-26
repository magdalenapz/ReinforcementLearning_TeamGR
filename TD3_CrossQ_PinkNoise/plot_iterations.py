import numpy as np
import matplotlib.pyplot as plt
import torch

heats = [256, 512, 1024, 2048]

dir = '/home/arneg/Schreibtisch/Reinforcement Learning/Project/Code/TD3-paper/elos/'
for heat in reversed(heats):
    elos = torch.load(f'{dir}elos{heat}', weights_only=False, map_location='cpu')
    plt.plot(elos, label=f'{heat}')
    plt.xlabel('policy index')#, labelpad=5)
    plt.ylabel('Elo')
    plt.xticks([0, 1] + list(range(2, len(elos))), ['0 \n (WBO)', '1 \n (SBO)'] + list(range(2, len(elos))))
plt.legend(title='heat')
plt.tight_layout()
plt.show()