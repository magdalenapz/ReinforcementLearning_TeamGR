import numpy as np
import matplotlib.pyplot as plt
import torch

heats = [256, 512, 1024, 2048]

dir = '/home/arneg/Schreibtisch/Reinforcement Learning/Project/Code/TD3-paper/scores/'
fig, axs = plt.subplots(1, 4, figsize=(6, 2.5), sharex=True, sharey=True, layout='constrained')
#plt.subplots_adjust(top=0.75, bottom=0.2)
for ax, heat in zip(axs, heats):
    scores = torch.load(f'{dir}scores{heat}', weights_only=False, map_location='cpu')
    im = ax.imshow(scores)
    ax.set_title(f'heat={heat}')
    ticks = np.arange(0, scores.shape[0], 5)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
cbar = fig.colorbar(im, ax=axs, location='top', fraction=0.07, pad=0.15)
cbar.set_label('mean score', labelpad=5)
fig.supxlabel('opponent index')
fig.supylabel('agent index', y=0.4)
#plt.tight_layout()

plt.show()