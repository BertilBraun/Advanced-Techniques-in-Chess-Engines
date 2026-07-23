from pathlib import Path

import h5py
import numpy as np

from src.Encoding import BINARY_CHANNELS, C, H, SCALAR_CHANNELS, W, encode_board_state
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.value_target import REPLAY_SCHEMA_VERSION, FinalOutcome, TerminationReason
from src.settings import CurrentGame


PRODUCTION_GLOBAL_BATCH_SIZE = 1024


def encoded_state_templates(count: int, seed: int) -> tuple[bytes, ...]:
    generator = np.random.default_rng(seed)
    templates: list[bytes] = []
    for _ in range(count):
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
        templates.append(encode_board_state(state))
    return tuple(templates)


def write_replay_fixture(
    path: Path,
    sample_count: int,
    seed: int,
    state_template_count: int = 4096,
) -> None:
    if state_template_count <= 0:
        raise ValueError('State template count must be positive.')
    path.parent.mkdir(parents=True, exist_ok=True)
    templates = encoded_state_templates(min(sample_count, state_template_count), seed)
    states = np.asarray([templates[index % len(templates)] for index in range(sample_count)])
    visit_counts = np.zeros((sample_count, 1, 2), dtype=np.int32)
    visit_counts[:, 0, 0] = np.arange(sample_count) % CurrentGame.action_size
    visit_counts[:, 0, 1] = 600
    mcts_root_values = np.linspace(-1.0, 1.0, num=sample_count, dtype=np.float32)
    final_outcomes = np.where(
        mcts_root_values > 0.0,
        int(FinalOutcome.WIN),
        np.where(
            mcts_root_values < 0.0,
            int(FinalOutcome.LOSS),
            int(FinalOutcome.DRAW),
        ),
    ).astype(np.uint8)
    game_count = max(1, sample_count // 50)
    stats = SelfPlayDatasetStats(
        num_samples=sample_count,
        num_games=game_count,
        game_lengths=[50] * game_count,
        total_generation_time=float(game_count),
    )
    with h5py.File(path, 'w') as file:
        file.create_dataset('states', data=states)
        file.create_dataset('visit_counts', data=visit_counts)
        file.create_dataset('final_outcomes', data=final_outcomes)
        file.create_dataset('mcts_root_values', data=mcts_root_values)
        file.create_dataset('outcome_target_eligible', data=np.ones(sample_count, dtype=np.bool_))
        file.create_dataset(
            'termination_reasons',
            data=np.full(sample_count, int(TerminationReason.NATURAL), dtype=np.uint8),
        )
        file.create_dataset('plies', data=np.arange(sample_count, dtype=np.int32) % 512)
        file.create_dataset(
            'current_player_piece_counts',
            data=np.full(sample_count, 8, dtype=np.uint8),
        )
        file.create_dataset(
            'opponent_piece_counts',
            data=np.full(sample_count, 8, dtype=np.uint8),
        )
        file.attrs['replay_schema_version'] = REPLAY_SCHEMA_VERSION
        file.attrs['metadata'] = str(SelfPlayDataset._get_current_metadata())
        file.attrs['stats'] = str(stats._asdict())
