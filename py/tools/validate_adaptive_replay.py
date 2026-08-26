from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.self_play.parameters import AdaptiveFullSearchBudget, ResolvedSelfPlayParameters
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.provenance import read_source_revision
from tools.measure_policy_target_fidelity import (
    STOPPING_RULE_GRID_ADAPTER,
    AdaptiveStoppingRule,
    _adaptive_stop,
    _checkpoints,
    _reference_parameters,
)
from tools.measure_policy_target_fidelity import (
    Arguments as FidelityArguments,
)
from tools.sample_chess_search_positions import PositionSample


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    model: Path
    positions: Path
    grid: Path
    rule_label: str
    output: Path
    generation: int
    device: int
    reference_visits: int
    observation_interval: int
    parallel_searches: int
    chunk_positions: int
    inference_batch_size: int
    position_limit: int


class ReplayDisagreement(FrozenModel):
    fen: str = Field(min_length=1)
    native_visits: int = Field(gt=0)
    replayed_visits: int = Field(gt=0)
    native_stop_reason: str = Field(min_length=1)


class ReplayValidation(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    rule_label: str = Field(min_length=1)
    positions: int = Field(gt=0)
    exact_agreement: float = Field(ge=0.0, le=1.0)
    mean_absolute_visit_difference: float = Field(ge=0.0)
    native_mean_visits: float = Field(gt=0.0)
    replayed_mean_visits: float = Field(gt=0.0)
    disagreements: tuple[ReplayDisagreement, ...]


def _native_budget(rule: AdaptiveStoppingRule) -> AdaptiveFullSearchBudget:
    return AdaptiveFullSearchBudget(
        kind='adaptive',
        minimum_visits=rule.minimum_visits,
        maximum_visits=rule.maximum_visits,
        observation_interval=rule.observation_interval,
        leader_stability_window=rule.leader_stability_window,
        root_value_tolerance=rule.root_value_tolerance,
        initial_top_visit_share=rule.initial_top_visit_share,
        final_top_visit_share=rule.final_top_visit_share,
        initial_top_two_margin=rule.initial_top_two_margin,
        final_top_two_margin=rule.final_top_two_margin,
        threshold_relaxation_visits=rule.threshold_relaxation_visits,
        minimum_search_correction_to_unlock_tail=None,
    )


def _native_parameters(
    game: ChessImplementation,
    arguments: Arguments,
    rule: AdaptiveStoppingRule,
) -> ResolvedSelfPlayParameters:
    baseline = game.self_play_parameters_at(arguments.generation)
    return replace(
        baseline,
        parallel_searches=arguments.parallel_searches,
        full_search_budget=_native_budget(rule),
        fast_searches=rule.maximum_visits,
        dirichlet_alpha=1.0,
        dirichlet_epsilon=0.0,
    )


def _fidelity_arguments(arguments: Arguments) -> FidelityArguments:
    return FidelityArguments(
        configuration=arguments.configuration,
        model=arguments.model,
        positions=arguments.positions,
        grid=arguments.grid,
        output=arguments.output,
        generation=arguments.generation,
        device=arguments.device,
        reference_visits=arguments.reference_visits,
        observation_interval=arguments.observation_interval,
        parallel_searches=arguments.parallel_searches,
        chunk_positions=arguments.chunk_positions,
        inference_batch_size=arguments.inference_batch_size,
        position_limit=arguments.position_limit,
    )


def validate_replay(arguments: Arguments) -> ReplayValidation:
    from AlphaZeroCpp import (
        BatchedInferenceParameters,
        ChessPosition,
        ChessSelfPlaySearch,
        ChessSelfPlaySearchRequest,
        SearchCheckpointDetail,
    )

    grid = STOPPING_RULE_GRID_ADAPTER.validate_json(arguments.grid.read_text(encoding='utf-8'))
    rules = tuple(rule for rule in grid.rules if rule.label == arguments.rule_label)
    if len(rules) != 1 or rules[0].kind != 'adaptive':
        raise ValueError(f'Grid must contain exactly one adaptive rule labelled {arguments.rule_label!r}.')
    rule = rules[0]
    sample = PositionSample.model_validate_json(arguments.positions.read_text(encoding='utf-8'))
    fens = tuple(position.fen for position in sample.positions)[: arguments.position_limit]

    configuration = load_chess_experiment_configuration(arguments.configuration)
    game = ChessImplementation(configuration)
    inference = BatchedInferenceParameters(
        workers=1,
        batch_size=arguments.inference_batch_size,
        outstanding_batches_per_worker=1,
    )

    native_visits: list[int] = []
    native_reasons: list[str] = []
    native_search = ChessSelfPlaySearch(
        game.native_inference_configuration(arguments.device, arguments.model),
        game.native_search_parameters(_native_parameters(game, arguments, rule)),
        inference,
        arguments.generation,
    )
    for start in range(0, len(fens), arguments.chunk_positions):
        roots = [native_search.new_root(ChessPosition(fen)) for fen in fens[start : start + arguments.chunk_positions]]
        batch = native_search.search([ChessSelfPlaySearchRequest(root, True) for root in roots])
        native_visits.extend(result.final_visits for result in batch.results)
        native_reasons.extend(result.stop_reason.name.lower() for result in batch.results)

    replayed_visits: list[int] = []
    reference_search = ChessSelfPlaySearch(
        game.native_inference_configuration(arguments.device, arguments.model),
        game.native_search_parameters(_reference_parameters(game, _fidelity_arguments(arguments))),
        inference,
        arguments.generation,
    )
    for start in range(0, len(fens), arguments.chunk_positions):
        roots = [
            reference_search.new_root(ChessPosition(fen)) for fen in fens[start : start + arguments.chunk_positions]
        ]
        batch = reference_search.search(
            [ChessSelfPlaySearchRequest(root, True, SearchCheckpointDetail.POLICIES) for root in roots]
        )
        replayed_visits.extend(
            _adaptive_stop(_checkpoints(result.checkpoints), rule, arguments.observation_interval).visits
            for result in batch.results
        )
    game.close()

    differences = [abs(native - replayed) for native, replayed in zip(native_visits, replayed_visits, strict=True)]
    return ReplayValidation(
        source_revision=read_source_revision().commit,
        rule_label=rule.label,
        positions=len(fens),
        exact_agreement=sum(difference == 0 for difference in differences) / len(differences),
        mean_absolute_visit_difference=sum(differences) / len(differences),
        native_mean_visits=sum(native_visits) / len(native_visits),
        replayed_mean_visits=sum(replayed_visits) / len(replayed_visits),
        disagreements=tuple(
            ReplayDisagreement(
                fen=fen,
                native_visits=native,
                replayed_visits=replayed,
                native_stop_reason=reason,
            )
            for fen, native, replayed, reason in zip(fens, native_visits, replayed_visits, native_reasons, strict=True)
            if native != replayed
        ),
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Check the replayed adaptive stop against the native search stop.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--rule-label', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--reference-visits', type=int, default=5000)
    parser.add_argument('--observation-interval', type=int, default=50)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--chunk-positions', type=int, default=64)
    parser.add_argument('--inference-batch-size', type=int, default=128)
    parser.add_argument('--position-limit', type=int, default=200)
    parsed = parser.parse_args()
    if parsed.reference_visits % parsed.observation_interval:
        parser.error('--reference-visits must be a whole number of observation intervals')
    return Arguments(
        configuration=parsed.configuration,
        model=parsed.model,
        positions=parsed.positions,
        grid=parsed.grid,
        rule_label=parsed.rule_label,
        output=parsed.output,
        generation=parsed.generation,
        device=parsed.device,
        reference_visits=parsed.reference_visits,
        observation_interval=parsed.observation_interval,
        parallel_searches=parsed.parallel_searches,
        chunk_positions=parsed.chunk_positions,
        inference_batch_size=parsed.inference_batch_size,
        position_limit=parsed.position_limit,
    )


def main() -> None:
    arguments = parse_arguments()
    validation = validate_replay(arguments)
    write_text_atomically(arguments.output, validation.model_dump_json(indent=2) + '\n')
    print(
        f'{validation.rule_label}: exact agreement {validation.exact_agreement:.3f} over '
        f'{validation.positions} positions, mean |difference| {validation.mean_absolute_visit_difference:.1f} visits'
    )


if __name__ == '__main__':
    main()
