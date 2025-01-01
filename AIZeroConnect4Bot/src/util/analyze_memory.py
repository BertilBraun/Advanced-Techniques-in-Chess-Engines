# load in the latest memory from training_data
# Display the board, probabilities, and value of the latest 20 memories

from collections import Counter
import numpy as np

from torch.optim import Adam

from src.Network import Network
from src.alpha_zero.AlphaZero import AlphaZero
from src.settings import TRAINING_ARGS


model = Network(4, 64)
az = AlphaZero(model, Adam(model.parameters()), TRAINING_ARGS, False)
for i in range(200, -1, -1):
    memory = az._load_all_memories(i)
    if memory:
        memory = az._deduplicate_positions(memory)
        for m in memory[:200]:
            # print board nicer than with 1, 0, -1 use X, O, and empty
            for row in m.state[0]:
                print(' '.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in row]))
            print(np.round(m.policy_targets, 2))
            print(m.value_target)
            print()

        value_counter = Counter()
        for m in memory:
            value_counter[round(m.value_target, 2)] += 1
        print(value_counter)

        exit()
