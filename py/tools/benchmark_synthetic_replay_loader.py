from __future__ import annotations

import argparse
import hashlib
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import (
    PackedPlaneLayout,
    PackedPlanePayload,
    RepresentationDimensions,
    decode_packed_planes,
)
from src.replay.batch_loader import build_dense_targets, build_training_batch, decode_states
from src.replay.columnar import ReplayColumnViews
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    IneligibleNextPolicyTarget,
    IneligibleRemainingGameLengthTarget,
    IneligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts, TerminationReason
from src.training.batch import TrainingBatch
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    TrainingTargetLayout,
    auxiliary_head_output_size,
)
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel

ACTION_SIZE = 4_864
MAXIMUM_LEGAL_ACTIONS = 218
MAXIMUM_POLICY_ENTRIES = 60


class BenchmarkTiming(FrozenModel):
    median_seconds: float
    median_rows_per_second: float
    trial_seconds: tuple[float, ...]


class SyntheticReplayBenchmarkReport(FrozenModel):
    evidence_scope: str
    python_version: str
    torch_version: str
    cpu: str
    appended_rows: int
    maximum_capacity: int
    logical_capacity: int
    live_rows: int
    batch_size: int
    iterations_per_trial: int
    repeats: int
    measured_rows_per_trial: int
    wrapped_store: bool
    duplicate_indices_per_batch: bool
    exact_equivalence: bool
    semantic_checksum: str
    index_generation: BenchmarkTiming
    logical_to_physical: BenchmarkTiming
    column_gather: BenchmarkTiming
    packed_decode: BenchmarkTiming
    decoded_augmentation: BenchmarkTiming
    dense_target_build: BenchmarkTiming
    full_object_reference: BenchmarkTiming
    full_vectorized: BenchmarkTiming
    full_build_speedup: float


class _SyntheticChessState(GameStateContract[int]):
    def __init__(self) -> None:
        self._representation = RepresentationDimensions(
            channels=29,
            rows=8,
            columns=8,
            binary_channels=tuple(range(22)),
            scalar_channels=tuple(range(22, 29)),
            packed_planes=PackedPlaneLayout(board_size=8, binary_plane_count=22, scalar_count=7),
        )

    @property
    def name(self) -> str:
        return 'synthetic_chess_shaped'

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def maximum_legal_action_count(self) -> int:
        return MAXIMUM_LEGAL_ACTIONS

    @property
    def representation(self) -> RepresentationDimensions:
        return self._representation

    def initial_position(self) -> int:
        return 0

    def legal_action_ids(self, position: int) -> tuple[int, ...]:
        del position
        return tuple(range(MAXIMUM_LEGAL_ACTIONS))

    def child_position(self, position: int, action_id: int) -> int:
        return position + action_id + 1

    def is_irreversible_transition(self, position: int, action_id: int, child: int) -> bool:
        del position, action_id, child
        return False

    def current_player(self, position: int) -> Player:
        return Player.FIRST if position % 2 == 0 else Player.SECOND

    def natural_terminal_wdl(self, position: int) -> WdlTarget | None:
        del position
        return None

    def adjudicated_wdl(self, position: int, reason: TerminationReason) -> WdlTarget:
        del position, reason
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def encode_network_input(self, position: int) -> PackedPlanePayload:
        del position
        return self.packed_plane_layout.empty_value()

    @property
    def augmentation_count(self) -> int:
        return 2

    def transform_decoded_states(
        self,
        states: npt.NDArray[np.float32],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> None:
        mirrored = augmentation_indices == 1
        states[mirrored] = states[mirrored, :, :, ::-1].copy()

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        if augmentation_index == 0:
            return action_id
        if augmentation_index == 1:
            return ACTION_SIZE - 1 - action_id
        raise ValueError('Synthetic augmentation index is outside the fixed layout.')


def _layout(state: _SyntheticChessState) -> ReplayLayout:
    targets = TrainingTargetLayout(
        action_size=ACTION_SIZE,
        wdl_size=3,
        auxiliary_heads=(
            NextPolicyHeadLayout(kind='next_policy', action_size=ACTION_SIZE, ply_offset=1),
            RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=400.0),
            FutureSearchValueHeadLayout(kind='future_search_value', ply_offset=4, smooth_l1_beta=0.5),
            IrreversibleProgressHeadLayout(kind='irreversible_progress', horizon_plies=8),
            LegalMovesHeadLayout(kind='legal_moves', action_size=ACTION_SIZE),
        ),
    )
    return ReplayLayout(
        packed_planes=state.packed_plane_layout,
        targets=targets,
        maximum_policy_entries=MAXIMUM_POLICY_ENTRIES,
        maximum_legal_actions=MAXIMUM_LEGAL_ACTIONS,
    )


def _policy(row: int, offset: int) -> SparsePolicyTarget:
    legal = tuple((row * 223 + offset + index) % ACTION_SIZE for index in range(MAXIMUM_LEGAL_ACTIONS))
    action_ids = legal[:16]
    return SparsePolicyTarget(
        visits=SearchVisitCounts(
            action_ids=action_ids,
            visit_counts=tuple((index + row) % 31 + 1 for index in range(len(action_ids))),
        ),
        legal_action_ids=legal,
    )


def _sample(state: _SyntheticChessState, row: int) -> ReplaySample:
    payload = (
        np.random.default_rng(row)
        .integers(
            0,
            256,
            size=state.packed_plane_layout.payload_bytes,
            dtype=np.uint8,
        )
        .tobytes()
    )
    eligible = row % 3 != 0
    next_policy = _policy(row, 997)
    return ReplaySample(
        encoded_state=state.packed_plane_layout.value(payload),
        policy=_policy(row, 0),
        wdl_target=(
            WdlTarget(win=1.0, draw=0.0, loss=0.0)
            if row % 3 == 0
            else WdlTarget(win=0.0, draw=1.0, loss=0.0)
            if row % 3 == 1
            else WdlTarget(win=0.0, draw=0.0, loss=1.0)
        ),
        root_value=float((row % 21) - 10) / 10.0,
        auxiliary_targets=(
            EligibleNextPolicyTarget(policy=next_policy) if eligible else IneligibleNextPolicyTarget(),
            EligibleRemainingGameLengthTarget(normalized_length=float(row % 100) / 100.0)
            if eligible
            else IneligibleRemainingGameLengthTarget(),
            EligibleScalarAuxiliaryTarget(kind='future_search_value', value=float((row % 9) - 4) / 4.0)
            if eligible
            else IneligibleScalarAuxiliaryTarget(kind='future_search_value'),
            EligibleScalarAuxiliaryTarget(kind='irreversible_progress', value=float(row % 8) / 8.0)
            if eligible
            else IneligibleScalarAuxiliaryTarget(kind='irreversible_progress'),
            EligibleLegalMovesTarget(),
        ),
        sample_weight=float(row % 4 + 1),
        source_model_generation=row % 100,
        source_created_at_seconds=float(1_700_000_000 + row),
    )


def _create_wrapped_store(
    path: Path,
    state: _SyntheticChessState,
    maximum_rows: int,
    logical_rows: int,
) -> ReplayStore:
    if not 1 < logical_rows < maximum_rows:
        raise ValueError('Synthetic benchmark requires logical rows strictly below maximum rows.')
    store = ReplayStore.create(path, _layout(state), maximum_capacity=maximum_rows, logical_capacity=logical_rows)
    try:
        appended_rows = maximum_rows + maximum_rows // 4
        for start in range(0, appended_rows, 64):
            store.extend(tuple(_sample(state, row) for row in range(start, min(start + 64, appended_rows))))
        store.flush()
        if store.state.head == 0 or store.state.head + store.state.size <= store.state.maximum_capacity:
            raise RuntimeError('Synthetic benchmark failed to construct a wrapped replay FIFO.')
    except BaseException:
        store.close()
        raise
    return store


def _plans(
    seed: int,
    iterations: int,
    replay_size: int,
    batch_size: int,
) -> tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...]:
    generator = np.random.default_rng(seed)
    plans = []
    for _ in range(iterations):
        indices = generator.choice(replay_size, size=batch_size, replace=False).astype(np.int64, copy=False)
        indices[1] = indices[0]
        indices[-2:] = (0, replay_size - 1)
        augmentations = generator.integers(0, 2, size=batch_size, dtype=np.int64)
        plans.append((indices, augmentations))
    return tuple(plans)


def _object_reference_batch(
    store: ReplayStore,
    state: _SyntheticChessState,
    indices: npt.NDArray[np.int64],
    augmentations: npt.NDArray[np.int64],
) -> TrainingBatch:
    samples = tuple(store.sample_at(int(index)) for index in indices)
    row_count = len(samples)
    states = np.empty((row_count, 29, 8, 8), dtype=np.float32)
    policy = np.zeros((row_count, ACTION_SIZE), dtype=np.float32)
    policy_legal = np.full((row_count, MAXIMUM_LEGAL_ACTIONS), -1, dtype=np.int64)
    auxiliary = tuple(
        np.zeros((row_count, auxiliary_head_output_size(head)), dtype=np.float32)
        for head in store.layout.targets.auxiliary_heads
    )
    auxiliary_legal = tuple(
        np.full((row_count, MAXIMUM_LEGAL_ACTIONS), -1, dtype=np.int64) for _ in store.layout.targets.auxiliary_heads
    )
    auxiliary_eligibility = tuple(np.zeros(row_count, dtype=np.bool_) for _ in store.layout.targets.auxiliary_heads)
    for output_row, (sample, augmentation) in enumerate(zip(samples, augmentations, strict=True)):
        decoded = decode_packed_planes(
            sample.encoded_state,
            state.packed_plane_layout,
            state.representation.binary_channels,
            state.representation.scalar_channels,
        ).astype(np.float32)[np.newaxis]
        state.transform_decoded_states(decoded, np.asarray((augmentation,), dtype=np.int64))
        states[output_row] = decoded[0]
        permutation = state.action_permutations[int(augmentation)]
        _write_policy_reference(policy[output_row], policy_legal[output_row], sample.policy, permutation)
        for target_index, target in enumerate(sample.auxiliary_targets):
            match target:
                case EligibleNextPolicyTarget(policy=next_policy):
                    _write_policy_reference(
                        auxiliary[target_index][output_row],
                        auxiliary_legal[target_index][output_row],
                        next_policy,
                        permutation,
                    )
                    auxiliary_eligibility[target_index][output_row] = True
                case EligibleRemainingGameLengthTarget(normalized_length=value):
                    auxiliary[target_index][output_row, 0] = value
                    auxiliary_eligibility[target_index][output_row] = True
                case EligibleScalarAuxiliaryTarget(value=value):
                    auxiliary[target_index][output_row, 0] = value
                    auxiliary_eligibility[target_index][output_row] = True
                case EligibleLegalMovesTarget():
                    transformed = permutation[np.asarray(sample.policy.legal_action_ids, dtype=np.uint16)]
                    auxiliary[target_index][output_row, transformed] = 1.0
                    auxiliary_legal[target_index][output_row, : len(transformed)] = transformed
                    auxiliary_eligibility[target_index][output_row] = True
                case (
                    IneligibleNextPolicyTarget()
                    | IneligibleRemainingGameLengthTarget()
                    | IneligibleScalarAuxiliaryTarget()
                ):
                    pass
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policy),
        policy_legal_action_ids=torch.from_numpy(policy_legal),
        wdl_targets=torch.tensor(
            tuple((sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss) for sample in samples),
            dtype=torch.float32,
        ),
        root_values=torch.tensor(tuple(sample.root_value for sample in samples), dtype=torch.float32),
        auxiliary_targets=tuple(torch.from_numpy(target) for target in auxiliary),
        auxiliary_legal_action_ids=tuple(torch.from_numpy(actions) for actions in auxiliary_legal),
        auxiliary_eligibility=tuple(torch.from_numpy(mask) for mask in auxiliary_eligibility),
        sample_weights=torch.tensor(tuple(sample.sample_weight for sample in samples), dtype=torch.float32),
        source_model_generations=torch.tensor(
            tuple(sample.source_model_generation for sample in samples), dtype=torch.int64
        ),
        source_created_at_seconds=torch.tensor(
            tuple(sample.source_created_at_seconds for sample in samples), dtype=torch.float64
        ),
    )


def _write_policy_reference(
    dense: npt.NDArray[np.float32],
    legal_output: npt.NDArray[np.int64],
    target: SparsePolicyTarget,
    permutation: npt.NDArray[np.uint16],
) -> None:
    actions = permutation[np.asarray(target.visits.action_ids, dtype=np.uint16)]
    visits = np.asarray(target.visits.visit_counts, dtype=np.float32)
    dense[actions] = visits / visits.sum()
    legal = permutation[np.asarray(target.legal_action_ids, dtype=np.uint16)]
    legal_output[: len(legal)] = legal


def _batch_tensors(batch: TrainingBatch) -> tuple[torch.Tensor, ...]:
    return (
        batch.states,
        batch.policy_targets,
        batch.policy_legal_action_ids,
        batch.wdl_targets,
        batch.root_values,
        *batch.auxiliary_targets,
        *batch.auxiliary_legal_action_ids,
        *batch.auxiliary_eligibility,
        batch.sample_weights,
        batch.source_model_generations,
        batch.source_created_at_seconds,
    )


def _validate_equivalence(
    store: ReplayStore,
    state: _SyntheticChessState,
    plans: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
) -> str:
    digest = hashlib.sha256()
    for indices, augmentations in plans:
        reference = _object_reference_batch(store, state, indices, augmentations)
        vectorized = build_training_batch(store, state, indices, augmentations)
        for expected, actual in zip(_batch_tensors(reference), _batch_tensors(vectorized), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            digest.update(actual.numpy().tobytes())
    return digest.hexdigest()


def _elapsed(operation: Callable[[], object]) -> float:
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started


def _timing(trials: list[float], rows: int) -> BenchmarkTiming:
    median = statistics.median(trials)
    return BenchmarkTiming(
        median_seconds=median,
        median_rows_per_second=rows / median,
        trial_seconds=tuple(trials),
    )


def _consume_physical_maps(
    store: ReplayStore,
    plans: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
) -> None:
    for indices, _ in plans:
        store.logical_to_physical(indices)


def _consume_gathers(
    store: ReplayStore,
    physical: tuple[npt.NDArray[np.int64], ...],
) -> None:
    for indices in physical:
        store.gather_physical(indices)


def _consume_decodes(
    gathered: tuple[ReplayColumnViews, ...],
    state: _SyntheticChessState,
) -> None:
    for columns in gathered:
        decode_states(columns.encoded_state, state)


def _consume_augmentations(
    decoded: tuple[npt.NDArray[np.float32], ...],
    plans: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
    state: _SyntheticChessState,
) -> None:
    for values, (_, augmentations) in zip(decoded, plans, strict=True):
        state.transform_decoded_states(values, augmentations)


def _consume_dense_targets(
    gathered: tuple[ReplayColumnViews, ...],
    plans: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
    store: ReplayStore,
    state: _SyntheticChessState,
) -> None:
    for columns, (_, augmentations) in zip(gathered, plans, strict=True):
        build_dense_targets(columns, store.layout, state, augmentations)


def _consume_reference_batches(
    store: ReplayStore,
    state: _SyntheticChessState,
    plans: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
) -> None:
    for plan in plans:
        _object_reference_batch(store, state, *plan)


def _consume_vectorized_batches(
    store: ReplayStore,
    state: _SyntheticChessState,
    plans: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
) -> None:
    for plan in plans:
        build_training_batch(store, state, *plan)


def run_benchmark(
    output: Path,
    maximum_rows: int = 1_024,
    logical_rows: int = 768,
    batch_size: int = 128,
    iterations: int = 6,
    repeats: int = 5,
    seed: int = 20_260_822,
) -> SyntheticReplayBenchmarkReport:
    if batch_size < 4 or batch_size > logical_rows:
        raise ValueError('Batch size must be at least four and no larger than logical replay size.')
    if iterations <= 0 or repeats <= 0:
        raise ValueError('Iterations and repeats must be positive.')
    state = _SyntheticChessState()
    with tempfile.TemporaryDirectory(prefix='az-synthetic-replay-benchmark-') as directory:
        store = _create_wrapped_store(Path(directory) / 'replay.bin', state, maximum_rows, logical_rows)
        try:
            plans = _plans(seed, iterations, store.state.size, batch_size)
            checksum = _validate_equivalence(store, state, plans)
            physical = tuple(store.logical_to_physical(indices) for indices, _ in plans)
            gathered = tuple(store.gather_physical(indices) for indices in physical)
            decoded = tuple(decode_states(columns.encoded_state, state) for columns in gathered)
            trial_values: dict[str, list[float]] = {
                name: []
                for name in (
                    'index',
                    'physical',
                    'gather',
                    'decode',
                    'augmentation',
                    'dense',
                    'reference',
                    'vectorized',
                )
            }
            for trial in range(repeats):
                trial_values['index'].append(_elapsed(lambda: _plans(seed, iterations, store.state.size, batch_size)))
                trial_values['physical'].append(_elapsed(lambda: _consume_physical_maps(store, plans)))
                trial_values['gather'].append(_elapsed(lambda: _consume_gathers(store, physical)))
                trial_values['decode'].append(_elapsed(lambda: _consume_decodes(gathered, state)))
                trial_values['augmentation'].append(_elapsed(lambda: _consume_augmentations(decoded, plans, state)))
                trial_values['dense'].append(_elapsed(lambda: _consume_dense_targets(gathered, plans, store, state)))
                builders = (
                    ('reference', lambda: _consume_reference_batches(store, state, plans)),
                    ('vectorized', lambda: _consume_vectorized_batches(store, state, plans)),
                )
                if trial % 2:
                    builders = tuple(reversed(builders))
                for name, builder in builders:
                    trial_values[name].append(_elapsed(builder))
        finally:
            store.close()
    rows = batch_size * iterations
    reference = _timing(trial_values['reference'], rows)
    vectorized = _timing(trial_values['vectorized'], rows)
    report = SyntheticReplayBenchmarkReport(
        evidence_scope='Synthetic CPU-only chess-shaped schema-4 replay; not CUDA, DDP, or live-run acceptance evidence.',
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cpu=platform.processor() or platform.machine(),
        appended_rows=maximum_rows + maximum_rows // 4,
        maximum_capacity=maximum_rows,
        logical_capacity=logical_rows,
        live_rows=logical_rows,
        batch_size=batch_size,
        iterations_per_trial=iterations,
        repeats=repeats,
        measured_rows_per_trial=rows,
        wrapped_store=True,
        duplicate_indices_per_batch=True,
        exact_equivalence=True,
        semantic_checksum=checksum,
        index_generation=_timing(trial_values['index'], rows),
        logical_to_physical=_timing(trial_values['physical'], rows),
        column_gather=_timing(trial_values['gather'], rows),
        packed_decode=_timing(trial_values['decode'], rows),
        decoded_augmentation=_timing(trial_values['augmentation'], rows),
        dense_target_build=_timing(trial_values['dense'], rows),
        full_object_reference=reference,
        full_vectorized=vectorized,
        full_build_speedup=reference.median_seconds / vectorized.median_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(output, report.model_dump_json(indent=2) + '\n')
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a bounded synthetic CPU replay-loader benchmark.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maximum-rows', type=_positive_int, default=1_024)
    parser.add_argument('--logical-rows', type=_positive_int, default=768)
    parser.add_argument('--batch-size', type=_positive_int, default=128)
    parser.add_argument('--iterations', type=_positive_int, default=6)
    parser.add_argument('--repeats', type=_positive_int, default=5)
    parser.add_argument('--seed', type=int, default=20_260_822)
    arguments = parser.parse_args()
    run_benchmark(
        arguments.output,
        arguments.maximum_rows,
        arguments.logical_rows,
        arguments.batch_size,
        arguments.iterations,
        arguments.repeats,
        arguments.seed,
    )


if __name__ == '__main__':
    main()
