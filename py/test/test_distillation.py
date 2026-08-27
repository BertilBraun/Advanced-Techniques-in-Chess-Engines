from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from src.distillation.dataset import (
    MAXIMUM_LEGAL_ACTIONS,
    MAXIMUM_POLICY_ENTRIES,
    DistillationDatasetManifest,
    build_training_batch,
    open_dataset,
    record_dtype,
    write_dataset,
)
from src.distillation.teacher import normalize_state_dict_keys
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.batch import TrainingBatch
from src.training.checkpoint.persistence import create_model, create_optimizer
from tools.benchmark_training_overfit import LossValues, achievable_loss_floor
from tools.distill_train_student import (
    distillation_objective,
    learning_rate_at,
    observed_losses,
    parameter_counts,
    student_architecture,
)

PAYLOAD_BYTES = CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes
ACTION_SIZE = CHESS_NETWORK_DIMENSIONS.actions


@dataclass(frozen=True)
class OverfitObservation:
    initial: LossValues
    final: LossValues
    floor: LossValues


def _manifest(position_count: int) -> DistillationDatasetManifest:
    return DistillationDatasetManifest(
        game='chess',
        position_count=position_count,
        action_size=ACTION_SIZE,
        payload_bytes=PAYLOAD_BYTES,
        maximum_policy_entries=MAXIMUM_POLICY_ENTRIES,
        maximum_legal_actions=MAXIMUM_LEGAL_ACTIONS,
        teacher_generation=94,
        teacher_weights_sha256='a' * 64,
        teacher_parameter_count=1_000_000,
        random_seed=7,
        random_opening_plies=6,
        sampling_temperature=1.0,
        sample_one_position_in=14,
        random_perturbation_probability=0.1,
        maximum_game_plies=300,
        builder_source_revision='0123456789abcdef',
    )


def _records(
    policy_entries: tuple[tuple[tuple[int, float], ...], ...],
    legal_counts: tuple[int, ...],
) -> npt.NDArray:
    records = np.zeros(len(policy_entries), dtype=record_dtype(PAYLOAD_BYTES))
    for row_index, entries in enumerate(policy_entries):
        records['policy_count'][row_index] = len(entries)
        for entry_index, (action_id, probability) in enumerate(entries):
            records['policy_action_ids'][row_index, entry_index] = action_id
            records['policy_probabilities'][row_index, entry_index] = probability
        legal_count = legal_counts[row_index]
        records['legal_count'][row_index] = legal_count
        records['legal_action_ids'][row_index, :legal_count] = np.arange(legal_count) + row_index
        records['wdl'][row_index] = (0.5, 0.3, 0.2)
    return records


def _synthetic_records(row_count: int, seed: int, legal_count: int = 8) -> npt.NDArray:
    generator = np.random.default_rng(seed)
    records = np.zeros(row_count, dtype=record_dtype(PAYLOAD_BYTES))
    for row_index in range(row_count):
        packed = generator.integers(0, 256, size=PAYLOAD_BYTES, dtype=np.uint8)
        records['packed_state'][row_index] = packed.tobytes()
        legal_action_ids = generator.choice(ACTION_SIZE, size=legal_count, replace=False)
        records['legal_count'][row_index] = legal_count
        records['legal_action_ids'][row_index, :legal_count] = legal_action_ids
        probabilities = generator.dirichlet(np.full(legal_count, 0.6))
        records['policy_count'][row_index] = legal_count
        records['policy_action_ids'][row_index, :legal_count] = legal_action_ids
        records['policy_probabilities'][row_index, :legal_count] = probabilities
        wdl = generator.dirichlet(np.ones(3))
        records['wdl'][row_index] = wdl
    return records


def _batch(records: npt.NDArray) -> TrainingBatch:
    return build_training_batch(records, CHESS_STATE_CONTRACT, ACTION_SIZE, torch.device('cpu'))


def test_written_dataset_reopens_with_its_manifest(tmp_path: Path) -> None:
    records = _synthetic_records(5, seed=1)
    dataset_path = tmp_path / 'positions.bin'

    write_dataset(dataset_path, records, _manifest(len(records)))
    _, manifest = open_dataset(dataset_path)

    assert manifest == _manifest(len(records))


@pytest.mark.parametrize(
    'field',
    ('packed_state', 'legal_count', 'legal_action_ids', 'policy_count', 'policy_action_ids', 'policy_probabilities'),
)
def test_dataset_round_trip_preserves_field(tmp_path: Path, field: str) -> None:
    records = _synthetic_records(5, seed=2)
    dataset_path = tmp_path / 'positions.bin'

    write_dataset(dataset_path, records, _manifest(len(records)))
    reopened, _ = open_dataset(dataset_path)

    assert np.array_equal(reopened[field], records[field])


def test_dataset_round_trip_preserves_teacher_wdl(tmp_path: Path) -> None:
    records = _synthetic_records(5, seed=3)
    dataset_path = tmp_path / 'positions.bin'

    write_dataset(dataset_path, records, _manifest(len(records)))
    reopened, _ = open_dataset(dataset_path)

    assert np.array_equal(reopened['wdl'], records['wdl'])


def test_open_dataset_rejects_a_row_count_its_manifest_denies(tmp_path: Path) -> None:
    records = _synthetic_records(5, seed=4)
    dataset_path = tmp_path / 'positions.bin'

    write_dataset(dataset_path, records, _manifest(4))

    with pytest.raises(ValueError, match='manifest declares'):
        open_dataset(dataset_path)


def test_dense_policy_targets_carry_the_stored_probabilities() -> None:
    batch = _batch(_records((((17, 0.25), (900, 0.75)), ((3, 1.0),)), legal_counts=(2, 1)))

    assert batch.policy_targets[0, 17] == pytest.approx(0.25)
    assert batch.policy_targets[0, 900] == pytest.approx(0.75)
    assert batch.policy_targets[1, 3] == pytest.approx(1.0)


def test_dense_policy_targets_are_zero_outside_the_stored_entries() -> None:
    batch = _batch(_records((((17, 0.25), (900, 0.75)), ((3, 1.0),)), legal_counts=(2, 1)))

    stored = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    stored[[17, 900]] = True

    assert torch.count_nonzero(batch.policy_targets[0, ~stored]) == 0


def test_dense_policy_targets_sum_to_one() -> None:
    batch = _batch(_records((((17, 0.25), (900, 0.75)), ((3, 0.5), (4, 0.5))), legal_counts=(2, 2)))

    assert batch.policy_targets.sum(dim=1).tolist() == pytest.approx((1.0, 1.0))


def test_dense_policy_targets_have_the_action_space_width() -> None:
    batch = _batch(_records((((17, 1.0),), ((3, 1.0),)), legal_counts=(1, 1)))

    assert batch.policy_targets.shape == (2, ACTION_SIZE)


@pytest.mark.parametrize('legal_count', (1, 7, MAXIMUM_LEGAL_ACTIONS))
def test_legal_action_ids_keep_the_stored_prefix(legal_count: int) -> None:
    batch = _batch(_records((((17, 1.0),), ((3, 1.0),)), legal_counts=(legal_count, 1)))

    assert torch.equal(batch.policy_legal_action_ids[0, :legal_count], torch.arange(legal_count))


@pytest.mark.parametrize('legal_count', (1, 7))
def test_legal_action_ids_pad_the_remainder_with_minus_one(legal_count: int) -> None:
    batch = _batch(_records((((17, 1.0),), ((3, 1.0),)), legal_counts=(legal_count, 1)))

    padding = batch.policy_legal_action_ids[0, legal_count:]

    assert torch.equal(padding, torch.full((MAXIMUM_LEGAL_ACTIONS - legal_count,), -1))


def test_legal_action_ids_have_the_fixed_layout_width() -> None:
    batch = _batch(_records((((17, 1.0),), ((3, 1.0),)), legal_counts=(2, 5)))

    assert batch.policy_legal_action_ids.shape == (2, MAXIMUM_LEGAL_ACTIONS)


def test_teacher_wdl_reaches_the_batch_unblended() -> None:
    batch = _batch(_records((((17, 1.0),), ((3, 1.0),)), legal_counts=(1, 1)))

    assert batch.wdl_targets[0].tolist() == pytest.approx((0.5, 0.3, 0.2))
    assert batch.root_values[0] == 0.0


@pytest.fixture(scope='module')
def overfit_observation() -> OverfitObservation:
    torch.manual_seed(20260826)
    batch = _batch(_synthetic_records(8, seed=5))
    objective = distillation_objective()
    model = create_model(student_architecture(1, 16, 32), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    optimizer = create_optimizer(model, 'adamw')
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = 0.05

    model.train()
    initial = observed_losses(objective.calculate_loss(model.training_output(batch.states), batch))
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = objective.calculate_loss(model.training_output(batch.states), batch)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return OverfitObservation(
        initial=initial,
        final=observed_losses(loss),
        floor=achievable_loss_floor(batch, objective),
    )


def test_training_loss_falls_well_below_its_starting_value(overfit_observation: OverfitObservation) -> None:
    assert overfit_observation.final.total < 0.8 * overfit_observation.initial.total


def test_training_loss_converges_onto_the_target_entropy_floor(overfit_observation: OverfitObservation) -> None:
    initial_gap = overfit_observation.initial.total - overfit_observation.floor.total
    final_gap = overfit_observation.final.total - overfit_observation.floor.total

    assert final_gap < 0.05 * initial_gap


def test_the_entropy_floor_bounds_the_reachable_policy_loss() -> None:
    batch = _batch(_synthetic_records(8, seed=6))

    floor = achievable_loss_floor(batch, distillation_objective())

    assert floor.policy > 0.0
    assert floor.wdl > 0.0


@pytest.mark.parametrize(
    ('step', 'expected'),
    ((1, 0.005), (10, 0.05), (20, 0.1)),
)
def test_warmup_scales_the_learning_rate_linearly(step: int, expected: float) -> None:
    assert learning_rate_at(step, total_steps=100, peak_learning_rate=0.1, warmup_steps=20) == pytest.approx(expected)


def test_cosine_decay_reaches_zero_at_the_final_step() -> None:
    assert learning_rate_at(100, total_steps=100, peak_learning_rate=0.1, warmup_steps=20) == pytest.approx(0.0)


def test_cosine_decay_halves_the_peak_at_the_decay_midpoint() -> None:
    assert learning_rate_at(60, total_steps=100, peak_learning_rate=0.1, warmup_steps=20) == pytest.approx(0.05)


def test_the_policy_bottleneck_keeps_the_student_heads_small() -> None:
    bottlenecked = create_model(student_architecture(6, 64, 16), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    unbottlenecked = create_model(student_architecture(6, 64, 1880), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)

    assert parameter_counts(bottlenecked).heads < 0.2 * parameter_counts(unbottlenecked).heads


def test_parameter_counts_split_the_whole_student() -> None:
    model = create_model(student_architecture(4, 32, 16), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)

    counts = parameter_counts(model)

    assert counts.total == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize(
    ('stored', 'expected'),
    (
        ('_orig_mod.backBone.0.x', 'backbone.0.x'),
        ('_orig_mod.startBlock.0.weight', 'start_block.0.weight'),
        ('startBlock.0.weight', 'start_block.0.weight'),
        ('policyHead.3.bias', 'policy_head.3.bias'),
        ('valueHead.1.weight', 'value_head.1.weight'),
        ('auxiliaryHeads.0.2.bias', 'auxiliary_head_modules.0.2.bias'),
    ),
)
def test_legacy_checkpoint_keys_normalize_to_current_module_names(stored: str, expected: str) -> None:
    assert tuple(normalize_state_dict_keys({stored: torch.zeros(1)})) == (expected,)


def test_current_checkpoint_keys_survive_normalization_unchanged() -> None:
    keys = {'backbone.0.weight': torch.zeros(1), 'value_head.1.bias': torch.zeros(1)}

    assert tuple(normalize_state_dict_keys(keys)) == tuple(keys)
