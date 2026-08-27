from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.self_play.parameters import FixedFullSearchBudget, ResolvedSelfPlayParameters
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision
from tools.sample_chess_search_positions import PositionSample


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    model: Path
    positions: Path
    output: Path
    generation: int
    device: int
    trees: int
    visits: int
    parallel_searches: tuple[int, ...]
    virtual_loss_weights: tuple[float, ...]
    inference_batch_size: int


class CollisionMeasurement(FrozenModel):
    parallel_searches: int = Field(gt=0)
    virtual_loss_weight: float = Field(ge=0.0, le=1.0)
    trees: int = Field(gt=0)
    simulations: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    simulations_per_second: float = Field(gt=0.0)
    average_batch_size: float = Field(gt=0.0)
    # With no collisions a tree offers one leaf per in-flight descent, so this approaches 1.0.
    batch_occupancy: float = Field(gt=0.0)
    inference_calls: int = Field(gt=0)


class CollisionReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    generation: int = Field(ge=0)
    visits: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    exploration_constant: float = Field(gt=0.0)
    forced_playout_coefficient: float = Field(ge=0.0)
    measurements: tuple[CollisionMeasurement, ...] = Field(min_length=1)


def _parameters(
    game: ChessImplementation,
    arguments: Arguments,
    parallel_searches: int,
    virtual_loss_weight: float,
) -> ResolvedSelfPlayParameters:
    baseline = game.self_play_parameters_at(arguments.generation)
    return replace(
        baseline,
        parallel_searches=parallel_searches,
        virtual_loss_weight=virtual_loss_weight,
        full_search_budget=FixedFullSearchBudget(kind='fixed', visits=arguments.visits),
        fast_searches=arguments.visits,
        dirichlet_alpha=1.0,
        dirichlet_epsilon=0.0,
    )


def measure_collisions(arguments: Arguments) -> CollisionReport:
    from AlphaZeroCpp import (
        BatchedInferenceParameters,
        ChessPosition,
        ChessSelfPlaySearch,
        ChessSelfPlaySearchRequest,
    )

    sample = PositionSample.model_validate_json(arguments.positions.read_text(encoding='utf-8'))
    fens = tuple(position.fen for position in sample.positions)[: arguments.trees]
    configuration = load_chess_experiment_configuration(arguments.configuration)
    game = ChessImplementation(configuration)
    measurements: list[CollisionMeasurement] = []
    reference = game.self_play_parameters_at(arguments.generation)

    for parallel_searches in arguments.parallel_searches:
        for virtual_loss_weight in arguments.virtual_loss_weights:
            if parallel_searches == 1 and virtual_loss_weight != arguments.virtual_loss_weights[0]:
                # A single descent in flight never collides, so the weight cannot matter.
                continue
            search = ChessSelfPlaySearch(
                game.native_inference_configuration(arguments.device, arguments.model),
                game.native_search_parameters(_parameters(game, arguments, parallel_searches, virtual_loss_weight)),
                BatchedInferenceParameters(
                    workers=1,
                    batch_size=arguments.inference_batch_size,
                    outstanding_batches_per_worker=1,
                ),
                arguments.generation,
            )
            roots = [search.new_root(ChessPosition(fen)) for fen in fens]
            started = time.perf_counter()
            batch = search.search([ChessSelfPlaySearchRequest(root, True) for root in roots])
            elapsed = time.perf_counter() - started
            statistics = search.inference_statistics()
            simulations = sum(result.final_visits - result.starting_visits for result in batch.results)
            average_batch_size = statistics.averageNumberOfPositionsInInferenceCall
            measurements.append(
                CollisionMeasurement(
                    parallel_searches=parallel_searches,
                    virtual_loss_weight=virtual_loss_weight,
                    trees=len(roots),
                    simulations=simulations,
                    elapsed_seconds=elapsed,
                    simulations_per_second=simulations / elapsed,
                    average_batch_size=average_batch_size,
                    batch_occupancy=average_batch_size
                    / min(arguments.inference_batch_size, parallel_searches * len(roots)),
                    inference_calls=statistics.modelInferenceCalls,
                )
            )
            print(
                f'parallel {parallel_searches:3d}  vlw {virtual_loss_weight:.2f}  '
                f'batch {average_batch_size:7.2f}  occupancy {measurements[-1].batch_occupancy:.3f}  '
                f'{measurements[-1].simulations_per_second:8.0f} sims/s',
                flush=True,
            )
    game.close()
    return CollisionReport(
        source_revision=read_source_revision().commit,
        tool_sha256=file_sha256(Path(__file__)),
        model_sha256=file_sha256(arguments.model),
        generation=arguments.generation,
        visits=arguments.visits,
        inference_batch_size=arguments.inference_batch_size,
        exploration_constant=reference.exploration_constant,
        forced_playout_coefficient=reference.forced_playout_coefficient,
        measurements=tuple(measurements),
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Measure inference-batch occupancy against virtual loss weight.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--trees', type=int, default=1)
    parser.add_argument('--visits', type=int, default=600)
    parser.add_argument('--parallel-searches', type=int, nargs='+', default=(1, 2, 4, 8, 16))
    parser.add_argument('--virtual-loss-weights', type=float, nargs='+', default=(0.25, 0.5, 1.0))
    parser.add_argument('--inference-batch-size', type=int, default=256)
    parsed = parser.parse_args()
    if min(parsed.generation, parsed.device) < 0:
        parser.error('--generation and --device must be nonnegative')
    if min(parsed.trees, parsed.visits, parsed.inference_batch_size) <= 0:
        parser.error('--trees, --visits and --inference-batch-size must be positive')
    if any(value <= 0 for value in parsed.parallel_searches):
        parser.error('--parallel-searches values must be positive')
    if any(not 0.0 <= value <= 1.0 for value in parsed.virtual_loss_weights):
        parser.error('--virtual-loss-weights must lie in [0, 1]')
    if parsed.visits <= max(parsed.parallel_searches):
        parser.error('--visits must exceed every parallel-search count')
    return Arguments(
        configuration=parsed.configuration,
        model=parsed.model,
        positions=parsed.positions,
        output=parsed.output,
        generation=parsed.generation,
        device=parsed.device,
        trees=parsed.trees,
        visits=parsed.visits,
        parallel_searches=tuple(parsed.parallel_searches),
        virtual_loss_weights=tuple(parsed.virtual_loss_weights),
        inference_batch_size=parsed.inference_batch_size,
    )


def main() -> None:
    arguments = parse_arguments()
    report = measure_collisions(arguments)
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    print(arguments.output)


if __name__ == '__main__':
    main()
