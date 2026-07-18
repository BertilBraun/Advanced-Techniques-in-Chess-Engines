from collections.abc import Iterator

import numpy as np
import torch

from src.Encoding import BINARY_CHANNELS, C, H, SCALAR_CHANNELS, W, decode_board_states, encode_board_state
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.train.Trainer import TrainingBatch, prefetch_training_batches


class FixedBatchLoader:
    def __init__(self, batches: tuple[TrainingBatch, ...]) -> None:
        self.batches = batches

    def __iter__(self) -> Iterator[TrainingBatch]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def encoded_state(seed: int) -> bytes:
    generator = np.random.default_rng(seed)
    state = np.zeros((C, H, W), dtype=np.int8)
    state[list(BINARY_CHANNELS)] = generator.integers(
        0,
        2,
        size=(len(BINARY_CHANNELS), H, W),
        dtype=np.int8,
    )
    state[list(SCALAR_CHANNELS)] = generator.integers(
        -1,
        2,
        size=(len(SCALAR_CHANNELS), 1, 1),
        dtype=np.int8,
    )
    return encode_board_state(state)


def dataset() -> SelfPlayDataset:
    result = SelfPlayDataset()
    result.encoded_states = [encoded_state(seed) for seed in range(4)]
    result.visit_counts = [np.asarray(((seed, seed + 1), (seed + 10, seed + 2)), dtype=np.uint16) for seed in range(4)]
    result.value_targets = [-1.0, -0.25, 0.5, 1.0]
    return result


def test_vectorized_board_decode_matches_individual_decode() -> None:
    samples = dataset()

    individual_states = torch.stack([samples[index][0] for index in range(len(samples))])
    batched_states = torch.from_numpy(decode_board_states(samples.encoded_states)).to(dtype=torch.float32)

    torch.testing.assert_close(batched_states, individual_states)


def test_prebatched_dataset_matches_individual_samples() -> None:
    samples = dataset()
    expected = tuple(
        torch.stack([samples[index][component] for index in range(len(samples))]) for component in range(3)
    )

    batch = samples.__getitems__(list(range(len(samples))))

    for actual_component, expected_component in zip(batch, expected):
        torch.testing.assert_close(actual_component, expected_component)


def test_background_prefetch_preserves_batch_order_and_values() -> None:
    samples = dataset()
    first_batch = samples.__getitems__([2, 0])
    second_batch = samples.__getitems__([3, 1])
    loader = FixedBatchLoader((first_batch, second_batch))

    prefetched = list(prefetch_training_batches(loader))

    assert len(prefetched) == 2
    for actual_batch, expected_batch in zip(prefetched, (first_batch, second_batch)):
        for actual_component, expected_component in zip(actual_batch, expected_batch):
            torch.testing.assert_close(actual_component, expected_component)
