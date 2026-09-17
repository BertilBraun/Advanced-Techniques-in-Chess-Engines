"""Verify that a live TorchScript bootstrap switches to the TensorRT checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.self_play.configuration import TensorRtInferenceBackend
from src.self_play.native_configuration import native_inference_backend
from src.training.checkpoint.contracts import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


@dataclass(frozen=True)
class Arguments:
    configuration_path: Path
    checkpoint_directory: Path
    bootstrap_generation: int
    candidate_generation: int
    device_id: int
    maximum_policy_absolute_error: float
    maximum_value_absolute_error: float
    output_path: Path


class PositionTransitionComparison(FrozenModel):
    moves_uci: tuple[str, ...]
    bootstrap_candidate_policy_difference: float = Field(ge=0.0)
    bootstrap_candidate_value_difference: float = Field(ge=0.0)
    transition_candidate_policy_error: float = Field(ge=0.0)
    transition_candidate_value_error: float = Field(ge=0.0)
    selected_action_matches: bool


class TensorRtBackendTransitionReport(FrozenModel):
    schema_version: Literal[1] = 1
    bootstrap_generation: int = Field(ge=0)
    candidate_generation: int = Field(gt=0)
    maximum_transition_policy_error: float = Field(ge=0.0)
    maximum_transition_value_error: float = Field(ge=0.0)
    all_selected_actions_match: bool
    bootstrap_and_candidate_differ: bool
    passed: bool
    positions: tuple[PositionTransitionComparison, ...] = Field(min_length=1)


HISTORIES = (
    (),
    ('e2e4',),
    ('d2d4', 'g8f6'),
    ('e2e4', 'e7e5', 'g1f3', 'b8c6'),
    ('d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3'),
    ('c2c4', 'e7e5', 'b1c3', 'g8f6', 'g2g3', 'd7d5'),
)
STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def _policy_difference(reference: list[tuple[int, float]], candidate: list[tuple[int, float]]) -> float:
    reference_policy = dict(reference)
    candidate_policy = dict(candidate)
    if reference_policy.keys() != candidate_policy.keys():
        raise ValueError('Inference backends returned different legal action IDs.')
    return max(abs(reference_policy[action] - candidate_policy[action]) for action in reference_policy)


def _selected_action(policy: list[tuple[int, float]]) -> int:
    return min(policy, key=lambda action: (-action[1], action[0]))[0]


def run(arguments: Arguments) -> TensorRtBackendTransitionReport:
    configuration = load_chess_experiment_configuration(arguments.configuration_path)
    game = ChessImplementation(configuration)
    match game.self_play_configuration.inference.backend:
        case TensorRtInferenceBackend(bootstrap_with_torchscript=True):
            pass
        case _:
            raise ValueError('Transition verification requires a TensorRT backend with a TorchScript bootstrap.')
    bootstrap_checkpoint = CheckpointReference.load_for_inference(
        arguments.checkpoint_directory,
        arguments.bootstrap_generation,
    )
    candidate_checkpoint = CheckpointReference.load_for_inference(
        arguments.checkpoint_directory,
        arguments.candidate_generation,
    )
    deployed_candidate = game.deployment_checkpoint(candidate_checkpoint)
    bootstrap_parameters = game.self_play_parameters_at(arguments.bootstrap_generation)
    candidate_parameters = game.self_play_parameters_at(arguments.candidate_generation)
    transitioned = game.create_native_search(arguments.device_id, bootstrap_checkpoint, bootstrap_parameters)
    bootstrap_reference = game.create_native_search(arguments.device_id, bootstrap_checkpoint, bootstrap_parameters)
    candidate_reference = game.create_native_search(arguments.device_id, deployed_candidate, candidate_parameters)
    transitioned.refresh_model(
        arguments.candidate_generation,
        str(deployed_candidate.inference_model_path),
        native_inference_backend(
            game.self_play_configuration.inference.backend,
            arguments.candidate_generation,
        ),
    )
    native_histories = [(STARTING_FEN, list(moves)) for moves in HISTORIES]
    bootstrap_outputs = bootstrap_reference.inference_with_history(native_histories)
    candidate_outputs = candidate_reference.inference_with_history(native_histories)
    transitioned_outputs = transitioned.inference_with_history(native_histories)
    comparisons: list[PositionTransitionComparison] = []
    for moves, bootstrap, candidate, refreshed in zip(
        HISTORIES,
        bootstrap_outputs,
        candidate_outputs,
        transitioned_outputs,
        strict=True,
    ):
        bootstrap_policy, bootstrap_value = bootstrap
        candidate_policy, candidate_value = candidate
        refreshed_policy, refreshed_value = refreshed
        comparisons.append(
            PositionTransitionComparison(
                moves_uci=moves,
                bootstrap_candidate_policy_difference=_policy_difference(bootstrap_policy, candidate_policy),
                bootstrap_candidate_value_difference=abs(bootstrap_value - candidate_value),
                transition_candidate_policy_error=_policy_difference(refreshed_policy, candidate_policy),
                transition_candidate_value_error=abs(refreshed_value - candidate_value),
                selected_action_matches=_selected_action(refreshed_policy) == _selected_action(candidate_policy),
            )
        )
    positions = tuple(comparisons)
    maximum_policy_error = max(item.transition_candidate_policy_error for item in positions)
    maximum_value_error = max(item.transition_candidate_value_error for item in positions)
    all_selected_actions_match = all(item.selected_action_matches for item in positions)
    bootstrap_and_candidate_differ = any(
        item.bootstrap_candidate_policy_difference > arguments.maximum_policy_absolute_error
        or item.bootstrap_candidate_value_difference > arguments.maximum_value_absolute_error
        for item in positions
    )
    report = TensorRtBackendTransitionReport(
        bootstrap_generation=arguments.bootstrap_generation,
        candidate_generation=arguments.candidate_generation,
        maximum_transition_policy_error=maximum_policy_error,
        maximum_transition_value_error=maximum_value_error,
        all_selected_actions_match=all_selected_actions_match,
        bootstrap_and_candidate_differ=bootstrap_and_candidate_differ,
        passed=(
            bootstrap_and_candidate_differ
            and all_selected_actions_match
            and maximum_policy_error <= arguments.maximum_policy_absolute_error
            and maximum_value_error <= arguments.maximum_value_absolute_error
        ),
        positions=positions,
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--checkpoint-directory', required=True, type=Path)
    parser.add_argument('--bootstrap-generation', type=int, default=0)
    parser.add_argument('--candidate-generation', required=True, type=int)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--maximum-policy-absolute-error', type=float, default=1e-6)
    parser.add_argument('--maximum-value-absolute-error', type=float, default=1e-6)
    parser.add_argument('--output', required=True, type=Path)
    parsed = parser.parse_args()
    arguments = Arguments(
        configuration_path=parsed.configuration.resolve(),
        checkpoint_directory=parsed.checkpoint_directory.resolve(),
        bootstrap_generation=parsed.bootstrap_generation,
        candidate_generation=parsed.candidate_generation,
        device_id=parsed.device_id,
        maximum_policy_absolute_error=parsed.maximum_policy_absolute_error,
        maximum_value_absolute_error=parsed.maximum_value_absolute_error,
        output_path=parsed.output.resolve(),
    )
    if not arguments.configuration_path.is_file() or not arguments.checkpoint_directory.is_dir():
        raise ValueError('Configuration and checkpoint directory must exist.')
    if (
        arguments.bootstrap_generation < 0
        or arguments.candidate_generation <= arguments.bootstrap_generation
        or arguments.device_id < 0
    ):
        raise ValueError('Generations and device ID must define a forward nonnegative transition.')
    if arguments.maximum_policy_absolute_error < 0.0 or arguments.maximum_value_absolute_error < 0.0:
        raise ValueError('Error limits must be nonnegative.')
    return arguments


def main() -> None:
    report = run(parse_arguments())
    print(report.model_dump_json(indent=2))
    if not report.passed:
        raise ValueError('TensorRT backend transition did not serve the candidate engine.')


if __name__ == '__main__':
    main()
