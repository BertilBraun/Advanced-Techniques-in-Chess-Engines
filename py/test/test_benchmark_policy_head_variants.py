from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.games.contracts import WdlTarget
from src.games.representation import encode_packed_planes
from src.replay.contracts import IneligibleSearchBudgetTarget, ReplaySample, SparsePolicyTarget
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts
from tools.benchmark_policy_head_variants import (
    POLICY_HEAD_VARIANTS,
    Arguments,
    replay_layout_for,
    resolve_variants,
    run_bake_off,
    split_rows,
)

GO_CONFIGURATION = Path(__file__).parent / 'configs' / 'go-7x7-experiment.yaml'
SYNTHETIC_ROWS = 48


def _synthetic_store(path: Path) -> None:
    configuration = load_experiment_configuration(GO_CONFIGURATION)
    game = create_game_implementation(configuration)
    layout = replay_layout_for(configuration, game)
    action_size = game.state.action_size
    representation = game.state.representation
    generator = np.random.default_rng(11)
    store = ReplayStore.create(path, layout, maximum_capacity=SYNTHETIC_ROWS, logical_capacity=SYNTHETIC_ROWS)
    try:
        for row in range(SYNTHETIC_ROWS):
            legal_action_ids = tuple(int(action) for action in generator.choice(action_size, size=6, replace=False))
            policy = SparsePolicyTarget(
                visits=SearchVisitCounts(
                    action_ids=legal_action_ids[:3],
                    visit_counts=(5, 3, 2),
                ),
                legal_action_ids=legal_action_ids,
            )
            store.append(
                ReplaySample(
                    encoded_state=encode_packed_planes(
                        generator.integers(
                            0,
                            2,
                            size=(representation.channels, representation.rows, representation.columns),
                            dtype=np.int8,
                        ),
                        layout.packed_planes,
                        representation.binary_channels,
                        representation.scalar_channels,
                    ),
                    policy=policy,
                    wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
                    root_value=0.0,
                    auxiliary_targets=(IneligibleSearchBudgetTarget(),),
                    sample_weight=1.0,
                    source_model_generation=row,
                    source_created_at_seconds=float(row),
                )
            )
        store.flush()
    finally:
        store.close()
        game.close()


def _arguments(store: Path, output: Path) -> Arguments:
    return Arguments(
        store=store,
        configuration=GO_CONFIGURATION,
        device_id=0,
        steps=4,
        batch_size=8,
        holdout_rows=16,
        evaluation_interval=2,
        learning_rate=0.002,
        trunk_layers=2,
        trunk_hidden_size=16,
        variants=('a-ch4-baseline', 'e-ch4-rank96'),
        random_seed=5,
        output_path=output,
    )


def test_row_split_keeps_the_holdout_out_of_the_training_pool() -> None:
    split = split_rows(100, holdout_rows=20, random_seed=3)

    assert len(split.holdout) == 20
    assert len(split.training) == 80
    assert not set(split.holdout.tolist()) & set(split.training.tolist())
    assert split_rows(100, holdout_rows=20, random_seed=3).holdout.tolist() == split.holdout.tolist()


def test_unknown_variant_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match='Unknown policy head variants'):
        resolve_variants(('z-nonexistent',))


def test_default_variant_selection_covers_the_documented_grid() -> None:
    assert tuple(variant_id for variant_id, _ in resolve_variants(())) == tuple(
        variant_id for variant_id, _ in POLICY_HEAD_VARIANTS
    )


def test_bake_off_smoke_runs_on_a_synthetic_store(tmp_path: Path) -> None:
    store_path = tmp_path / 'replay.bin'
    output_path = tmp_path / 'report.json'
    _synthetic_store(store_path)

    report = run_bake_off(_arguments(store_path, output_path))

    assert tuple(result.variant_id for result in report.results) == ('a-ch4-baseline', 'e-ch4-rank96')
    assert report.training_rows == SYNTHETIC_ROWS - report.holdout_rows
    assert all(result.completed_steps == 4 for result in report.results)
    assert all(result.head_parameter_count > 0 for result in report.results)
    assert json.loads(output_path.read_text())['results'][0]['variant_id'] == 'a-ch4-baseline'

    repeated = run_bake_off(_arguments(store_path, output_path))

    assert [result.final_holdout_policy_cross_entropy for result in repeated.results] == [
        result.final_holdout_policy_cross_entropy for result in report.results
    ]
