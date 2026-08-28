from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from src.evaluation.contracts import OPENING_SUITE_MANIFEST_ADAPTER
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.search_budget.allocation import production_parallel_searches
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision

if TYPE_CHECKING:
    from AlphaZeroCpp import ChessPosition

_MINIMUM_BRANCHING_FACTOR = 2


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    model: Path
    openings: Path
    output: Path
    generation: int
    device: int
    games: int
    positions: int
    rollout_visits: int
    maximum_plies: int
    move_sampling_temperature: float
    inference_batch_size: int
    random_seed: int


class SampledPosition(FrozenModel):
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)
    legal_action_count: int = Field(ge=_MINIMUM_BRANCHING_FACTOR)
    fen: str = Field(min_length=1)


class PositionSample(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    configuration_path: Path
    model_path: Path
    model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    generation: int = Field(ge=0)
    opening_manifest_path: Path
    opening_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    games: int = Field(gt=0)
    rollout_visits: int = Field(gt=0)
    maximum_plies: int = Field(gt=0)
    move_sampling_temperature: float = Field(gt=0.0)
    random_seed: int = Field(ge=0)
    observed_position_count: int = Field(gt=0)
    rollout_seconds: float = Field(ge=0.0)
    rollout_simulations: int = Field(ge=0)
    positions: tuple[SampledPosition, ...] = Field(min_length=1)


def _opening_positions(game: ChessImplementation, manifest_path: Path, games: int) -> tuple[ChessPosition, ...]:
    manifest = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(manifest_path.read_text(encoding='utf-8'))
    if manifest.game != 'chess':
        raise ValueError('Position sampling requires a chess opening manifest.')
    state = game.state
    positions: list[ChessPosition] = []
    for index in range(games):
        opening = manifest.openings[index % len(manifest.openings)]
        position = state.initial_position()
        for action_id in opening.action_ids:
            position = state.child_position(position, action_id)
        positions.append(position)
    return tuple(positions)


def _rollout_parameters(game: ChessImplementation, arguments: Arguments) -> ResolvedSelfPlayParameters:
    baseline = game.self_play_parameters_at(arguments.generation, 0.0)
    # Sampling only needs a plausible move distribution, so it runs without root noise or forced playouts.
    return replace(
        baseline,
        baseline_visits=arguments.rollout_visits,
        forced_playout_coefficient=0.0,
        dirichlet_alpha=1.0,
        dirichlet_epsilon=0.0,
    )


def _sampled_action_id(
    visits: list[tuple[int, int]],
    temperature: float,
    generator: random.Random,
) -> int:
    weights = [count ** (1.0 / temperature) for _, count in visits]
    return generator.choices([action_id for action_id, _ in visits], weights=weights, k=1)[0]


def collect_positions(arguments: Arguments) -> PositionSample:
    from AlphaZeroCpp import (
        BatchedInferenceParameters,
        ChessSelfPlaySearch,
    )

    configuration = load_chess_experiment_configuration(arguments.configuration)
    game = ChessImplementation(configuration)
    parameters = _rollout_parameters(game, arguments)
    search = ChessSelfPlaySearch(
        game.native_inference_configuration(arguments.device, arguments.model),
        game.native_search_parameters(parameters),
        BatchedInferenceParameters(
            workers=1,
            batch_size=arguments.inference_batch_size,
            outstanding_batches_per_worker=1,
        ),
        arguments.generation,
    )
    generator = random.Random(arguments.random_seed)
    roots = [search.new_root(position) for position in _opening_positions(game, arguments.openings, arguments.games)]
    live = list(enumerate(roots))
    observed: list[SampledPosition] = []
    simulations = 0
    started = time.perf_counter()
    for ply in range(arguments.maximum_plies):
        if not live:
            break
        batch = search.search(
            [
                search.request(
                    root,
                    assigned_additional_visits=arguments.rollout_visits,
                    parallel_searches=production_parallel_searches(arguments.rollout_visits),
                    add_root_noise=False,
                )
                for _, root in live
            ]
        )
        simulations += sum(result.final_visits - result.starting_visits for result in batch.results)
        advanced: list[tuple[int, object]] = []
        for (game_index, root), result in zip(live, batch.results, strict=True):
            visits = [(visit.action_id, visit.visit_count) for visit in result.search_visits]
            if len(visits) < _MINIMUM_BRANCHING_FACTOR:
                # A forced move carries no policy signal, so it is played but never sampled.
                if visits:
                    root.play(visits[0][0])
                    if not root.is_terminal:
                        advanced.append((game_index, root))
                continue
            observed.append(
                SampledPosition(
                    game_index=game_index,
                    ply=ply,
                    legal_action_count=len(visits),
                    fen=root.position.fen,
                )
            )
            root.play(_sampled_action_id(visits, arguments.move_sampling_temperature, generator))
            if not root.is_terminal:
                advanced.append((game_index, root))
        live = advanced
    rollout_seconds = time.perf_counter() - started
    game.close()
    if len(observed) < arguments.positions:
        raise ValueError(f'Rollout produced {len(observed)} positions, fewer than the requested {arguments.positions}.')
    selected = sorted(
        generator.sample(observed, arguments.positions),
        key=lambda position: (position.game_index, position.ply),
    )
    return PositionSample(
        source_revision=read_source_revision().commit,
        tool_sha256=file_sha256(Path(__file__)),
        configuration_path=arguments.configuration.resolve(),
        model_path=arguments.model.resolve(),
        model_sha256=file_sha256(arguments.model),
        generation=arguments.generation,
        opening_manifest_path=arguments.openings.resolve(),
        opening_manifest_sha256=file_sha256(arguments.openings),
        games=arguments.games,
        rollout_visits=arguments.rollout_visits,
        maximum_plies=arguments.maximum_plies,
        move_sampling_temperature=arguments.move_sampling_temperature,
        random_seed=arguments.random_seed,
        observed_position_count=len(observed),
        rollout_seconds=rollout_seconds,
        rollout_simulations=simulations,
        positions=tuple(selected),
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Sample chess positions from frozen-model rollouts for target study.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--openings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--positions', type=int, default=3000)
    parser.add_argument('--rollout-visits', type=int, default=100)
    parser.add_argument('--maximum-plies', type=int, default=200)
    parser.add_argument('--move-sampling-temperature', type=float, default=1.0)
    parser.add_argument('--inference-batch-size', type=int, default=256)
    parser.add_argument('--random-seed', type=int, default=20260826)
    parsed = parser.parse_args()
    if min(parsed.generation, parsed.device, parsed.random_seed) < 0:
        parser.error('--generation, --device and --random-seed must be nonnegative')
    positive = (
        parsed.games,
        parsed.positions,
        parsed.rollout_visits,
        parsed.maximum_plies,
        parsed.inference_batch_size,
    )
    if min(positive) <= 0 or parsed.move_sampling_temperature <= 0.0:
        parser.error('game, position, visit, ply, batch and temperature values must be positive')
    if parsed.rollout_visits < 2:
        parser.error('--rollout-visits must exceed one so the search can expand a child')
    return Arguments(
        configuration=parsed.configuration,
        model=parsed.model,
        openings=parsed.openings,
        output=parsed.output,
        generation=parsed.generation,
        device=parsed.device,
        games=parsed.games,
        positions=parsed.positions,
        rollout_visits=parsed.rollout_visits,
        maximum_plies=parsed.maximum_plies,
        move_sampling_temperature=parsed.move_sampling_temperature,
        inference_batch_size=parsed.inference_batch_size,
        random_seed=parsed.random_seed,
    )


def main() -> None:
    arguments = parse_arguments()
    sample = collect_positions(arguments)
    write_text_atomically(arguments.output, sample.model_dump_json(indent=2) + '\n')
    print(arguments.output)


if __name__ == '__main__':
    main()
