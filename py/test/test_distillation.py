from __future__ import annotations

from dataclasses import dataclass, replace
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
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout
from tools.benchmark_training_overfit import LossValues, achievable_loss_floor
from tools.distill_train_student import (
    AUXILIARY_LOSS_WEIGHT,
    Arguments,
    AttentionBiasKind,
    LearningRateSchedule,
    NetworkKind,
    OptimizerKind,
    PolicyHeadKind,
    auxiliary_head_layouts,
    dataset_split,
    distillation_objective,
    held_out_batches,
    learning_rate_at,
    observed_losses,
    parameter_counts,
    student_architecture,
)

STUDENT_ARGUMENTS = Arguments(
    dataset=Path('unused.bin'),
    output_run_state=Path('unused'),
    network_kind=NetworkKind.CONVOLUTIONAL,
    layers=6,
    hidden_size=64,
    heads=8,
    feedforward=128,
    policy_head_kind=PolicyHeadKind.DENSE,
    policy_key_size=128,
    attention_bias_kind=AttentionBiasKind.DISABLED,
    smolgen_compressed_size=8,
    smolgen_hidden_size=32,
    smolgen_generated_size=32,
    optimizer_kind=OptimizerKind.ADAMW,
    floor_fraction=0.1,
    policy_bottleneck_rank=16,
    batch_size=64,
    steps=1000,
    learning_rate=0.002,
    learning_rate_schedule=LearningRateSchedule.COSINE,
    anneal_fraction=0.2,
    warmup_steps=200,
    max_grad_norm=0.5,
    holdout_fraction=0.02,
    training_fraction=1.0,
    evaluate_every=200,
    checkpoint_every=0,
    distil_auxiliary_heads=(),
    device_id=0,
    random_seed=1,
    generation=0,
)


def convolutional_student(layers: int, hidden_size: int, policy_bottleneck_rank: int) -> Arguments:
    return replace(
        STUDENT_ARGUMENTS,
        layers=layers,
        hidden_size=hidden_size,
        policy_bottleneck_rank=policy_bottleneck_rank,
    )


PAYLOAD_BYTES = CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes
ACTION_SIZE = CHESS_NETWORK_DIMENSIONS.actions
DISTILLED_AUXILIARY_HEADS = ('next_policy', 'remaining_game_length')


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
    model = create_model(
        student_architecture(convolutional_student(1, 16, 32)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS
    )
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


@pytest.mark.parametrize('schedule', tuple(LearningRateSchedule))
@pytest.mark.parametrize(
    ('step', 'expected'),
    ((1, 0.005), (10, 0.05), (20, 0.1)),
)
def test_warmup_scales_the_learning_rate_linearly(schedule: LearningRateSchedule, step: int, expected: float) -> None:
    rate = learning_rate_at(step, total_steps=100, peak_learning_rate=0.1, warmup_steps=20, schedule=schedule)

    assert rate == pytest.approx(expected)


def test_cosine_decay_reaches_zero_at_the_final_step() -> None:
    assert learning_rate_at(100, total_steps=100, peak_learning_rate=0.1, warmup_steps=20) == pytest.approx(0.0)


def test_cosine_decay_halves_the_peak_at_the_decay_midpoint() -> None:
    assert learning_rate_at(60, total_steps=100, peak_learning_rate=0.1, warmup_steps=20) == pytest.approx(0.05)


def test_the_policy_bottleneck_keeps_the_student_heads_small() -> None:
    bottlenecked = create_model(
        student_architecture(convolutional_student(6, 64, 16)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS
    )
    unbottlenecked = create_model(
        student_architecture(convolutional_student(6, 64, 1880)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS
    )

    assert parameter_counts(bottlenecked).heads < 0.2 * parameter_counts(unbottlenecked).heads


def test_parameter_counts_split_the_whole_student() -> None:
    model = create_model(
        student_architecture(convolutional_student(4, 32, 16)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS
    )

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


def test_the_cosine_schedule_is_the_default() -> None:
    explicit = learning_rate_at(
        60,
        total_steps=100,
        peak_learning_rate=0.1,
        warmup_steps=20,
        schedule=LearningRateSchedule.COSINE,
    )

    assert learning_rate_at(60, total_steps=100, peak_learning_rate=0.1, warmup_steps=20) == explicit


@pytest.mark.parametrize('anneal_fraction', (0.1, 0.2, 0.5))
def test_the_cosine_schedule_ignores_the_anneal_fraction(anneal_fraction: float) -> None:
    rate = learning_rate_at(
        60,
        total_steps=100,
        peak_learning_rate=0.1,
        warmup_steps=20,
        schedule=LearningRateSchedule.COSINE,
        anneal_fraction=anneal_fraction,
    )

    assert rate == pytest.approx(0.05)


@pytest.mark.parametrize('step', (21, 50, 79, 80))
def test_the_plateau_schedule_holds_the_peak_until_the_anneal_begins(step: int) -> None:
    rate = learning_rate_at(
        step,
        total_steps=100,
        peak_learning_rate=0.1,
        warmup_steps=20,
        schedule=LearningRateSchedule.PLATEAU,
        anneal_fraction=0.2,
    )

    assert rate == pytest.approx(0.1)


def test_the_plateau_schedule_halves_the_peak_at_the_anneal_midpoint() -> None:
    rate = learning_rate_at(
        90,
        total_steps=100,
        peak_learning_rate=0.1,
        warmup_steps=20,
        schedule=LearningRateSchedule.PLATEAU,
        anneal_fraction=0.2,
    )

    assert rate == pytest.approx(0.05)


@pytest.mark.parametrize('anneal_fraction', (0.1, 0.2, 0.5, 1.0))
def test_the_plateau_schedule_reaches_zero_at_the_final_step(anneal_fraction: float) -> None:
    rate = learning_rate_at(
        100,
        total_steps=100,
        peak_learning_rate=0.1,
        warmup_steps=20,
        schedule=LearningRateSchedule.PLATEAU,
        anneal_fraction=anneal_fraction,
    )

    assert rate == pytest.approx(0.0)


def _plateau_rate_at(step: int, anneal_fraction: float) -> float:
    return learning_rate_at(
        step,
        total_steps=100,
        peak_learning_rate=0.1,
        warmup_steps=20,
        schedule=LearningRateSchedule.PLATEAU,
        anneal_fraction=anneal_fraction,
    )


@pytest.mark.parametrize(('anneal_fraction', 'anneal_start'), ((0.1, 90), (0.25, 75), (0.5, 50)))
def test_the_plateau_still_holds_the_peak_on_the_step_the_anneal_starts(
    anneal_fraction: float, anneal_start: int
) -> None:
    assert _plateau_rate_at(anneal_start, anneal_fraction) == pytest.approx(0.1)


@pytest.mark.parametrize(('anneal_fraction', 'anneal_start'), ((0.1, 90), (0.25, 75), (0.5, 50)))
def test_the_plateau_drops_below_the_peak_on_the_first_annealing_step(
    anneal_fraction: float, anneal_start: int
) -> None:
    assert _plateau_rate_at(anneal_start + 1, anneal_fraction) < 0.1


@pytest.mark.parametrize(
    ('heads', 'expected'),
    (
        ((), ()),
        (('next_policy',), (NextPolicyHeadLayout(kind='next_policy', action_size=ACTION_SIZE, ply_offset=1),)),
        (
            ('remaining_game_length',),
            (RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=400.0),),
        ),
        (
            DISTILLED_AUXILIARY_HEADS,
            (
                NextPolicyHeadLayout(kind='next_policy', action_size=ACTION_SIZE, ply_offset=1),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=400.0),
            ),
        ),
    ),
)
def test_requested_head_names_build_the_matching_layouts(heads: tuple[str, ...], expected: tuple[object, ...]) -> None:
    assert auxiliary_head_layouts(heads, ACTION_SIZE) == expected


def test_every_distilled_auxiliary_head_carries_the_production_loss_weight() -> None:
    objective = distillation_objective(DISTILLED_AUXILIARY_HEADS)

    assert [loss.weight for loss in objective.auxiliary_losses] == [AUXILIARY_LOSS_WEIGHT, AUXILIARY_LOSS_WEIGHT]


def test_an_objective_without_auxiliary_heads_carries_no_auxiliary_losses() -> None:
    assert distillation_objective().auxiliary_losses == ()


def _auxiliary_records(
    next_policy_entries: tuple[tuple[tuple[int, float], ...], ...],
    remaining_game_lengths: tuple[float, ...],
    legal_counts: tuple[int, ...],
) -> npt.NDArray:
    records = _records(tuple(((0, 1.0),) for _ in next_policy_entries), legal_counts)
    for row_index, entries in enumerate(next_policy_entries):
        records['next_policy_count'][row_index] = len(entries)
        for entry_index, (action_id, probability) in enumerate(entries):
            records['next_policy_action_ids'][row_index, entry_index] = action_id
            records['next_policy_probabilities'][row_index, entry_index] = probability
        records['remaining_game_length'][row_index] = remaining_game_lengths[row_index]
    return records


def _auxiliary_batch(records: npt.NDArray) -> TrainingBatch:
    return build_training_batch(
        records,
        CHESS_STATE_CONTRACT,
        ACTION_SIZE,
        torch.device('cpu'),
        DISTILLED_AUXILIARY_HEADS,
    )


@pytest.fixture(scope='module')
def auxiliary_batch() -> TrainingBatch:
    return _auxiliary_batch(
        _auxiliary_records(
            next_policy_entries=(((11, 0.25), (802, 0.75)), ((7, 1.0),)),
            remaining_game_lengths=(0.125, 0.5),
            legal_counts=(2, 1),
        )
    )


def test_requesting_two_auxiliary_heads_builds_one_target_each(auxiliary_batch: TrainingBatch) -> None:
    assert len(auxiliary_batch.auxiliary_targets) == len(DISTILLED_AUXILIARY_HEADS)


def test_next_policy_targets_have_the_action_space_width(auxiliary_batch: TrainingBatch) -> None:
    assert auxiliary_batch.auxiliary_targets[0].shape == (2, ACTION_SIZE)


def test_remaining_game_length_targets_are_one_column_wide(auxiliary_batch: TrainingBatch) -> None:
    assert auxiliary_batch.auxiliary_targets[1].shape == (2, 1)


def test_next_policy_targets_carry_the_stored_probabilities(auxiliary_batch: TrainingBatch) -> None:
    dense = auxiliary_batch.auxiliary_targets[0]

    assert (dense[0, 11], dense[0, 802], dense[1, 7]) == pytest.approx((0.25, 0.75, 1.0))


def test_next_policy_targets_are_zero_outside_the_stored_entries(auxiliary_batch: TrainingBatch) -> None:
    stored = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    stored[[11, 802]] = True

    assert torch.count_nonzero(auxiliary_batch.auxiliary_targets[0][0, ~stored]) == 0


def test_next_policy_targets_sum_to_one(auxiliary_batch: TrainingBatch) -> None:
    assert auxiliary_batch.auxiliary_targets[0].sum(dim=1).tolist() == pytest.approx((1.0, 1.0))


def test_remaining_game_length_targets_carry_the_stored_scalars(auxiliary_batch: TrainingBatch) -> None:
    assert auxiliary_batch.auxiliary_targets[1].flatten().tolist() == pytest.approx((0.125, 0.5))


@pytest.mark.parametrize('head_index', range(len(DISTILLED_AUXILIARY_HEADS)))
def test_every_row_is_eligible_for_every_auxiliary_head(auxiliary_batch: TrainingBatch, head_index: int) -> None:
    assert bool(auxiliary_batch.auxiliary_eligibility[head_index].all())


def test_the_next_policy_head_is_masked_by_its_own_entries(auxiliary_batch: TrainingBatch) -> None:
    stored = auxiliary_batch.auxiliary_targets[0][0].nonzero().flatten()
    mask = auxiliary_batch.auxiliary_legal_action_ids[0][0]

    assert torch.equal(mask[mask >= 0].sort().values, stored.sort().values)


def test_the_next_policy_head_does_not_reuse_the_primary_legal_actions(auxiliary_batch: TrainingBatch) -> None:
    assert not torch.equal(auxiliary_batch.auxiliary_legal_action_ids[0], auxiliary_batch.policy_legal_action_ids)


def test_the_remaining_game_length_head_gets_padding_only_legal_actions(auxiliary_batch: TrainingBatch) -> None:
    padding = auxiliary_batch.auxiliary_legal_action_ids[1]

    assert torch.equal(padding, torch.full((2, MAXIMUM_LEGAL_ACTIONS), -1))


def _synthetic_auxiliary_records(row_count: int, seed: int, legal_count: int = 8) -> npt.NDArray:
    records = _synthetic_records(row_count, seed, legal_count)
    generator = np.random.default_rng(seed + 1)
    for row_index in range(row_count):
        # The head predicts the following ply, whose action ids are in the opponent's frame, so this support is
        # deliberately disjoint from the current position's legal actions.
        action_ids = (
            records['legal_action_ids'][row_index, :legal_count].astype(np.int64) + ACTION_SIZE // 2
        ) % ACTION_SIZE
        records['next_policy_count'][row_index] = legal_count
        records['next_policy_action_ids'][row_index, :legal_count] = action_ids
        records['next_policy_probabilities'][row_index, :legal_count] = generator.dirichlet(np.full(legal_count, 0.6))
        records['remaining_game_length'][row_index] = generator.integers(1, 400) / 400.0
    return records


@pytest.fixture(scope='module')
def auxiliary_overfit_observation() -> OverfitObservation:
    torch.manual_seed(20260827)
    batch = _auxiliary_batch(_synthetic_auxiliary_records(8, seed=21))
    objective = distillation_objective(DISTILLED_AUXILIARY_HEADS)
    model = create_model(
        student_architecture(convolutional_student(1, 16, 32)),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        auxiliary_head_layouts(DISTILLED_AUXILIARY_HEADS, ACTION_SIZE),
    )
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


def test_the_auxiliary_objective_reports_one_loss_per_head(
    auxiliary_overfit_observation: OverfitObservation,
) -> None:
    assert len(auxiliary_overfit_observation.final.auxiliary) == len(DISTILLED_AUXILIARY_HEADS)


def test_auxiliary_training_loss_falls_well_below_its_starting_value(
    auxiliary_overfit_observation: OverfitObservation,
) -> None:
    assert auxiliary_overfit_observation.final.total < 0.8 * auxiliary_overfit_observation.initial.total


def test_auxiliary_training_policy_loss_falls_below_its_starting_value(
    auxiliary_overfit_observation: OverfitObservation,
) -> None:
    assert auxiliary_overfit_observation.final.policy < auxiliary_overfit_observation.initial.policy


def test_the_next_policy_loss_falls_toward_its_own_entropy_floor(
    auxiliary_overfit_observation: OverfitObservation,
) -> None:
    initial_gap = auxiliary_overfit_observation.initial.auxiliary[0] - auxiliary_overfit_observation.floor.auxiliary[0]
    final_gap = auxiliary_overfit_observation.final.auxiliary[0] - auxiliary_overfit_observation.floor.auxiliary[0]

    assert final_gap < 0.5 * initial_gap


def test_auxiliary_heads_are_counted_apart_from_the_primary_heads() -> None:
    layouts = auxiliary_head_layouts(DISTILLED_AUXILIARY_HEADS, ACTION_SIZE)
    plain = create_model(
        student_architecture(convolutional_student(6, 64, 16)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS
    )
    distilling = create_model(
        student_architecture(convolutional_student(6, 64, 16)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS, layouts
    )

    assert parameter_counts(plain).heads == parameter_counts(distilling).heads


def test_auxiliary_heads_enlarge_the_reported_total() -> None:
    layouts = auxiliary_head_layouts(DISTILLED_AUXILIARY_HEADS, ACTION_SIZE)
    plain = create_model(
        student_architecture(convolutional_student(6, 64, 16)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS
    )
    distilling = create_model(
        student_architecture(convolutional_student(6, 64, 16)), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS, layouts
    )

    assert parameter_counts(distilling).total > parameter_counts(plain).total


def test_parameter_counts_split_the_whole_student_with_auxiliary_heads() -> None:
    model = create_model(
        student_architecture(convolutional_student(4, 32, 16)),
        torch.device('cpu'),
        CHESS_NETWORK_DIMENSIONS,
        auxiliary_head_layouts(DISTILLED_AUXILIARY_HEADS, ACTION_SIZE),
    )

    counts = parameter_counts(model)

    assert counts.total == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize(
    ('training_fraction', 'expected_training_rows'),
    ((1.0, 30), (0.5, 15), (0.25, 8)),
)
def test_the_training_fraction_shortens_the_training_prefix(
    training_fraction: float, expected_training_rows: int
) -> None:
    split = dataset_split(40, holdout_fraction=0.25, training_fraction=training_fraction)

    assert split.training_row_count == expected_training_rows


@pytest.mark.parametrize('training_fraction', (1.0, 0.5, 0.25))
def test_the_training_fraction_leaves_the_held_out_split_where_it_was(training_fraction: float) -> None:
    split = dataset_split(40, holdout_fraction=0.25, training_fraction=training_fraction)

    assert (split.held_out_start_row, split.held_out_row_count) == (30, 10)


def test_halving_the_training_fraction_leaves_the_held_out_floor_bit_identical() -> None:
    records = _synthetic_records(40, seed=31)
    objective = distillation_objective()

    def floor_at(training_fraction: float) -> LossValues:
        split = dataset_split(len(records), holdout_fraction=0.25, training_fraction=training_fraction)
        batches = held_out_batches(records, split.held_out_start_row, 5, ACTION_SIZE, torch.device('cpu'))
        return achievable_loss_floor(batches[0], objective)

    assert floor_at(0.5) == floor_at(1.0)
