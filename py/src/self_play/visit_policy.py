from collections.abc import Iterable

import numpy as np

from src.settings import CurrentGame


def action_probabilities(visit_counts: Iterable[tuple[int, int]]) -> np.ndarray:
    probabilities = np.zeros(CurrentGame.action_size, dtype=np.float32)
    for move, visit_count in visit_counts:
        probabilities[move] = visit_count
    total_visits = float(np.sum(probabilities))
    if total_visits <= 0:
        raise ValueError('Visit counts must contain at least one positive visit.')
    probabilities /= total_visits
    return probabilities
