# load in the latest memory from training_data
# Display the board, probabilities, and value of the latest 20 memories

import os
import torch
import numpy as np

from AIZeroConnect4Bot.src.settings import SAVE_PATH

for i in range(200, -1, -1):
    for file in os.listdir(SAVE_PATH):
        if file.startswith(f'memory_{i}_dedu'):
            memory = torch.load(SAVE_PATH + '/' + file)
            for m in memory:  # [:200]:
                if m[2] == 1 or m[2] == -1:
                    continue
                # print board nicer than with 1, 0, -1 use X, O, and empty
                for row in m[0][0]:
                    print(' '.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in row]))
                print(np.round(m[1], 2))
                print(m[2])
                print()
            exit()
