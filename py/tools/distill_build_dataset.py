from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.distillation.dataset import (
    CHESS_PAYLOAD_BYTES,
    MAXIMUM_LEGAL_ACTIONS,
    MAXIMUM_POLICY_ENTRIES,
    DistillationDatasetManifest,
    record_dtype,
    write_dataset,
)
from src.distillation.teacher import LoadedTeacher, load_teacher, read_network_definition
from src.evaluation.inference import decode_packed_inputs
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT, ChessPosition
from src.games.representation import PackedPlanePayload
from src.training.checkpoint.paths import checkpoint_manifest_path, model_save_path
from src.training.network import (
    DensePolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    NetworkDefinition,
    NetworkParams,
    ResidualContextPlacement,
)
from src.training.targets import AuxiliaryHeadLayout, NextPolicyHeadLayout, RemainingGameLengthHeadLayout
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision

PROGRESS_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class BuilderArguments:
    teacher_run_state: Path
    teacher_generation: int
    teacher_layers: int
    teacher_hidden_size: int
    output: Path
    positions: int
    parallel_games: int
    random_opening_plies: int
    sampling_temperature: float
    sample_one_position_in: int
    random_perturbation_probability: float
    maximum_game_plies: int
    random_seed: int
    device_id: int


@dataclass
class GameSlot:
    position: ChessPosition
    ply: int


@dataclass(frozen=True)
class AuxiliaryHeadIndices:
    next_policy: int | None
    remaining_game_length: int | None


@dataclass(frozen=True)
class AuxiliaryRowOutputs:
    next_policy_logits: npt.NDArray[np.float32] | None
    remaining_game_length: float | None


@dataclass(frozen=True)
class AuxiliaryBatchOutputs:
    next_policy_logits: npt.NDArray[np.float32] | None
    remaining_game_length: npt.NDArray[np.float32] | None

    def row(self, index: int) -> AuxiliaryRowOutputs:
        return AuxiliaryRowOutputs(
            next_policy_logits=None if self.next_policy_logits is None else self.next_policy_logits[index],
            remaining_game_length=(
                None if self.remaining_game_length is None else float(self.remaining_game_length[index])
            ),
        )


@dataclass(frozen=True)
class PolicyEntries:
    action_ids: npt.NDArray[np.int64]
    probabilities: npt.NDArray[np.float64]


def build_teacher_architecture(layers: int, hidden_size: int) -> NetworkParams:
    return NetworkParams(
        num_layers=layers,
        hidden_size=hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        policy_head=DensePolicyHeadConfiguration(channels=4),
        num_value_channels=2,
        value_fc_size=48,
    )


def resolve_teacher_definition(arguments: BuilderArguments) -> NetworkDefinition:
    manifest_path = checkpoint_manifest_path(arguments.teacher_generation, arguments.teacher_run_state)
    definition = read_network_definition(manifest_path) if manifest_path.exists() else None
    if definition is None:
        print(
            f'{manifest_path} carries no network definition; falling back to --teacher-layers '
            f'{arguments.teacher_layers} --teacher-hidden-size {arguments.teacher_hidden_size} and no auxiliary heads.',
            flush=True,
        )
        return NetworkDefinition(
            architecture=build_teacher_architecture(arguments.teacher_layers, arguments.teacher_hidden_size),
            dimensions=CHESS_NETWORK_DIMENSIONS,
            auxiliary_heads=(),
        )
    if definition.dimensions != CHESS_NETWORK_DIMENSIONS:
        raise ValueError(
            f'{manifest_path} describes a network over {definition.dimensions}, but this builder writes chess records '
            f'over {CHESS_NETWORK_DIMENSIONS}.'
        )
    head_kinds = tuple(head.kind for head in definition.auxiliary_heads)
    print(
        f'Teacher architecture from {manifest_path} (--teacher-layers and --teacher-hidden-size ignored): '
        f'{definition.architecture.model_dump_json()}; auxiliary heads {head_kinds}',
        flush=True,
    )
    return definition


def locate_auxiliary_heads(heads: tuple[AuxiliaryHeadLayout, ...]) -> AuxiliaryHeadIndices:
    next_policy: int | None = None
    remaining_game_length: int | None = None
    for index, head in enumerate(heads):
        match head:
            case NextPolicyHeadLayout():
                next_policy = index
            case RemainingGameLengthHeadLayout():
                remaining_game_length = index
            case _:
                # Generation cannot be repeated cheaply, so an unstorable head stops the run instead of being dropped.
                raise ValueError(
                    f'The teacher carries the auxiliary head {head.kind!r}, for which the distillation record has no '
                    f'column. Extend the record before generating from this teacher.'
                )
    return AuxiliaryHeadIndices(next_policy=next_policy, remaining_game_length=remaining_game_length)


def open_game(generator: np.random.Generator, random_opening_plies: int) -> ChessPosition:
    while True:
        position = CHESS_STATE_CONTRACT.initial_position()
        for _ in range(random_opening_plies):
            legal_action_ids = CHESS_STATE_CONTRACT.legal_action_ids(position)
            action_id = int(legal_action_ids[generator.integers(len(legal_action_ids))])
            position = CHESS_STATE_CONTRACT.child_position(position, action_id)
            if CHESS_STATE_CONTRACT.natural_terminal_wdl(position) is not None:
                break
        else:
            return position


def softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    exponentials = np.exp(logits - logits.max())
    return exponentials / exponentials.sum()


def top_policy_entries(
    legal_action_ids: npt.NDArray[np.int64], policy_over_legal: npt.NDArray[np.float64]
) -> PolicyEntries:
    entry_count = min(len(legal_action_ids), MAXIMUM_POLICY_ENTRIES)
    selected = np.argsort(-policy_over_legal, kind='stable')[:entry_count]
    probabilities = policy_over_legal[selected]
    return PolicyEntries(action_ids=legal_action_ids[selected], probabilities=probabilities / probabilities.sum())


def capture_auxiliary_outputs(
    auxiliary_logits: tuple[torch.Tensor, ...], head_indices: AuxiliaryHeadIndices
) -> AuxiliaryBatchOutputs:
    next_policy_logits = (
        None if head_indices.next_policy is None else auxiliary_logits[head_indices.next_policy].float().cpu().numpy()
    )
    remaining_game_length = (
        None
        if head_indices.remaining_game_length is None
        else auxiliary_logits[head_indices.remaining_game_length].float().cpu().numpy().reshape(-1)
    )
    return AuxiliaryBatchOutputs(next_policy_logits=next_policy_logits, remaining_game_length=remaining_game_length)


def write_record(
    record: np.void,
    packed_state: PackedPlanePayload,
    legal_action_ids: npt.NDArray[np.int64],
    teacher_policy: npt.NDArray[np.float64],
    wdl_probabilities: npt.NDArray[np.float32],
    auxiliary: AuxiliaryRowOutputs,
) -> None:
    legal_count = len(legal_action_ids)
    entries = top_policy_entries(legal_action_ids, teacher_policy)
    entry_count = len(entries.action_ids)
    record['packed_state'] = np.void(packed_state.payload)
    record['legal_count'] = legal_count
    record['legal_action_ids'][:legal_count] = legal_action_ids
    record['policy_count'] = entry_count
    record['policy_action_ids'][:entry_count] = entries.action_ids
    record['policy_probabilities'][:entry_count] = entries.probabilities
    record['wdl'] = wdl_probabilities
    if auxiliary.next_policy_logits is not None:
        next_policy = softmax(auxiliary.next_policy_logits[legal_action_ids].astype(np.float64))
        next_entries = top_policy_entries(legal_action_ids, next_policy)
        next_entry_count = len(next_entries.action_ids)
        record['next_policy_count'] = next_entry_count
        record['next_policy_action_ids'][:next_entry_count] = next_entries.action_ids
        record['next_policy_probabilities'][:next_entry_count] = next_entries.probabilities
    if auxiliary.remaining_game_length is not None:
        record['remaining_game_length'] = auxiliary.remaining_game_length


def report_progress(recorded: int, total: int, completed_games: int, elapsed_seconds: float) -> None:
    rate = recorded / elapsed_seconds
    remaining_minutes = (total - recorded) / rate / 60.0 if recorded else float('inf')
    print(
        f'{recorded}/{total} positions | {completed_games} games | '
        f'{rate:.0f} positions/s | eta {remaining_minutes:.1f} min',
        flush=True,
    )


def generate_records(
    teacher: LoadedTeacher,
    arguments: BuilderArguments,
    device: torch.device,
    head_indices: AuxiliaryHeadIndices,
) -> npt.NDArray:
    generator = np.random.default_rng(arguments.random_seed)
    records = np.zeros(arguments.positions, dtype=record_dtype(CHESS_PAYLOAD_BYTES))
    slots = [
        GameSlot(position=open_game(generator, arguments.random_opening_plies), ply=arguments.random_opening_plies)
        for _ in range(arguments.parallel_games)
    ]

    recorded = 0
    completed_games = 0
    started_at = time.monotonic()
    reported_at = started_at

    while recorded < arguments.positions:
        packed_states = tuple(CHESS_STATE_CONTRACT.encode_network_input(slot.position) for slot in slots)
        legal_action_ids = tuple(
            np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(slot.position), dtype=np.int64) for slot in slots
        )
        decoded = decode_packed_inputs(CHESS_STATE_CONTRACT, packed_states)
        with torch.inference_mode():
            output = teacher.network.training_output(torch.from_numpy(decoded).to(device))
            policy_logits = output.policy_logits.float().cpu().numpy()
            wdl_probabilities = torch.softmax(output.wdl_logits.float(), dim=1).cpu().numpy()
            auxiliary = capture_auxiliary_outputs(output.auxiliary_logits, head_indices)

        for row, slot in enumerate(slots):
            if recorded == arguments.positions:
                break
            legal = legal_action_ids[row]
            legal_logits = policy_logits[row, legal].astype(np.float64)
            teacher_policy = softmax(legal_logits)
            # A fixed stride from a fixed opening length only ever lands on one side to move, so an exchange
            # is seen from one end and never the other. Retaining each ply independently removes that.
            if generator.random() < 1.0 / arguments.sample_one_position_in:
                write_record(
                    records[recorded],
                    packed_states[row],
                    legal,
                    teacher_policy,
                    wdl_probabilities[row],
                    auxiliary.row(row),
                )
                recorded += 1
            if generator.random() < arguments.random_perturbation_probability:
                chosen = int(generator.integers(len(legal)))
            else:
                chosen = int(generator.choice(len(legal), p=softmax(legal_logits / arguments.sampling_temperature)))
            slot.position = CHESS_STATE_CONTRACT.child_position(slot.position, int(legal[chosen]))
            slot.ply += 1
            game_over = CHESS_STATE_CONTRACT.natural_terminal_wdl(slot.position) is not None
            if game_over or slot.ply >= arguments.maximum_game_plies:
                completed_games += 1
                slot.position = open_game(generator, arguments.random_opening_plies)
                slot.ply = arguments.random_opening_plies

        now = time.monotonic()
        if now - reported_at >= PROGRESS_INTERVAL_SECONDS:
            report_progress(recorded, arguments.positions, completed_games, now - started_at)
            reported_at = now

    report_progress(recorded, arguments.positions, completed_games, time.monotonic() - started_at)
    return records


def parse_arguments() -> BuilderArguments:
    parser = argparse.ArgumentParser(
        description='Generate diverse chess positions and label them with the raw head outputs of a teacher network.'
    )
    parser.add_argument('--teacher-run-state', type=Path, required=True, help='Run-state directory holding model_N.pt.')
    parser.add_argument('--teacher-generation', type=int, required=True, help='Generation of the teacher checkpoint.')
    parser.add_argument(
        '--teacher-layers',
        type=int,
        required=True,
        help='Residual block count; used only when the checkpoint manifest carries no network definition.',
    )
    parser.add_argument(
        '--teacher-hidden-size',
        type=int,
        required=True,
        help='Trunk width; used only when the checkpoint manifest carries no network definition.',
    )
    parser.add_argument('--output', type=Path, required=True, help='Dataset file to write; manifest sits beside it.')
    parser.add_argument('--positions', type=int, required=True, help='Number of labelled positions to collect.')
    parser.add_argument('--parallel-games', type=int, default=512, help='Games kept in flight per teacher batch.')
    parser.add_argument('--random-opening-plies', type=int, default=4, help='Uniformly random plies opening a game.')
    parser.add_argument('--sampling-temperature', type=float, default=1.0, help='Temperature of the move sampling.')
    parser.add_argument(
        '--sample-one-position-in',
        type=int,
        default=14,
        help='Retain each ply with probability 1/N, so a game contributes about its length over N positions.',
    )
    parser.add_argument(
        '--random-perturbation-probability',
        type=float,
        default=0.05,
        help='Probability of playing a uniformly random legal move at any ply.',
    )
    parser.add_argument('--maximum-game-plies', type=int, default=300, help='Ply cap after which a game is restarted.')
    parser.add_argument('--random-seed', type=int, required=True, help='Seed of the position-generation sampler.')
    parser.add_argument('--device-id', type=int, default=0, help='CUDA device index; CPU when CUDA is unavailable.')
    parsed = parser.parse_args()
    return BuilderArguments(
        teacher_run_state=parsed.teacher_run_state,
        teacher_generation=parsed.teacher_generation,
        teacher_layers=parsed.teacher_layers,
        teacher_hidden_size=parsed.teacher_hidden_size,
        output=parsed.output,
        positions=parsed.positions,
        parallel_games=parsed.parallel_games,
        random_opening_plies=parsed.random_opening_plies,
        sampling_temperature=parsed.sampling_temperature,
        sample_one_position_in=parsed.sample_one_position_in,
        random_perturbation_probability=parsed.random_perturbation_probability,
        maximum_game_plies=parsed.maximum_game_plies,
        random_seed=parsed.random_seed,
        device_id=parsed.device_id,
    )


def main() -> None:
    arguments = parse_arguments()
    device = torch.device('cuda', arguments.device_id) if torch.cuda.is_available() else torch.device('cpu')
    definition = resolve_teacher_definition(arguments)
    head_indices = locate_auxiliary_heads(definition.auxiliary_heads)
    weights_path = model_save_path(arguments.teacher_generation, arguments.teacher_run_state)
    teacher = load_teacher(
        weights_path=weights_path,
        architecture=definition.architecture,
        dimensions=definition.dimensions,
        auxiliary_heads=definition.auxiliary_heads,
        device=device,
        generation=arguments.teacher_generation,
    )
    print(f'Teacher {weights_path} on {device}: {teacher.parameter_count} parameters', flush=True)

    records = generate_records(teacher, arguments, device, head_indices)

    revision = read_source_revision()
    manifest = DistillationDatasetManifest(
        game=CHESS_STATE_CONTRACT.name,
        position_count=arguments.positions,
        action_size=CHESS_STATE_CONTRACT.action_size,
        payload_bytes=CHESS_PAYLOAD_BYTES,
        maximum_policy_entries=MAXIMUM_POLICY_ENTRIES,
        maximum_legal_actions=MAXIMUM_LEGAL_ACTIONS,
        teacher_generation=arguments.teacher_generation,
        teacher_weights_sha256=file_sha256(weights_path),
        teacher_parameter_count=teacher.parameter_count,
        random_seed=arguments.random_seed,
        random_opening_plies=arguments.random_opening_plies,
        sampling_temperature=arguments.sampling_temperature,
        sample_one_position_in=arguments.sample_one_position_in,
        random_perturbation_probability=arguments.random_perturbation_probability,
        maximum_game_plies=arguments.maximum_game_plies,
        builder_source_revision=revision.commit + ('-dirty' if revision.dirty else ''),
        captured_auxiliary_heads=tuple(head.kind for head in definition.auxiliary_heads),
    )
    write_dataset(arguments.output, records, manifest)
    print(f'Wrote {arguments.positions} positions to {arguments.output}', flush=True)


if __name__ == '__main__':
    main()
