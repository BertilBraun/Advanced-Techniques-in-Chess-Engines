from __future__ import annotations

import argparse
import math
import statistics
import time
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import Field, TypeAdapter
from src.experiment.configuration import load_chess_experiment_configuration
from src.games.chess.training import ChessImplementation
from src.self_play.parameters import AdaptiveFullSearchBudget, ResolvedSelfPlayParameters
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision
from tools.sample_chess_search_positions import PositionSample

if TYPE_CHECKING:
    from AlphaZeroCpp import SearchCheckpoint

# Floors the candidate policy so a target that never visited a reference action keeps a finite penalty.
_POLICY_PROBABILITY_FLOOR = 1e-6
_FIRST_CHECKPOINT_VALUE_DELTA = 2.0


class FixedStoppingRule(FrozenModel):
    kind: Literal['fixed'] = 'fixed'
    label: str = Field(min_length=1)
    visits: int = Field(gt=0)


class AdaptiveStoppingRule(FrozenModel):
    kind: Literal['adaptive'] = 'adaptive'
    label: str = Field(min_length=1)
    minimum_visits: int = Field(gt=0)
    maximum_visits: int = Field(gt=0)
    observation_interval: int = Field(gt=0)
    leader_stability_window: int = Field(gt=0)
    root_value_tolerance: float = Field(ge=0.0, le=1.0)
    initial_top_visit_share: float = Field(ge=0.0, le=1.0)
    final_top_visit_share: float = Field(ge=0.0, le=1.0)
    initial_top_two_margin: float = Field(ge=0.0, le=1.0)
    final_top_two_margin: float = Field(ge=0.0, le=1.0)
    threshold_relaxation_visits: int = Field(gt=0)


StoppingRule: TypeAlias = Annotated[FixedStoppingRule | AdaptiveStoppingRule, Field(discriminator='kind')]


class StoppingRuleGrid(FrozenModel):
    schema_version: Literal[1] = 1
    description: str = Field(min_length=1)
    rules: tuple[StoppingRule, ...] = Field(min_length=1)


STOPPING_RULE_GRID_ADAPTER = TypeAdapter(StoppingRuleGrid)


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    model: Path
    positions: Path
    grid: Path
    output: Path
    generation: int
    device: int
    reference_visits: int
    observation_interval: int
    parallel_searches: int
    chunk_positions: int
    inference_batch_size: int
    position_limit: int | None


@dataclass(frozen=True)
class _Checkpoint:
    visits: int
    leader_action_id: int
    most_visited_action_id: int
    top_visit_share: float
    top_two_margin: float
    root_value: float
    policy: tuple[tuple[int, int], ...]


@dataclass
class _RuleAccumulator:
    positions: int = 0
    stop_visits: int = 0
    maximum_reached: int = 0
    leader_agreements: int = 0
    most_visited_agreements: int = 0
    kullback_leibler: float = 0.0
    total_variation: float = 0.0
    root_value_error: float = 0.0
    stop_visit_samples: list[int] = field(default_factory=list)


class RuleFidelity(FrozenModel):
    label: str = Field(min_length=1)
    kind: Literal['fixed', 'adaptive']
    positions: int = Field(gt=0)
    mean_stop_visits: float = Field(gt=0.0)
    median_stop_visits: float = Field(gt=0.0)
    maximum_reached_fraction: float = Field(ge=0.0, le=1.0)
    policy_leader_agreement: float = Field(ge=0.0, le=1.0)
    most_visited_agreement: float = Field(ge=0.0, le=1.0)
    mean_policy_kullback_leibler: float = Field(ge=0.0)
    mean_policy_total_variation: float = Field(ge=0.0, le=1.0)
    mean_root_value_absolute_error: float = Field(ge=0.0)


class EqualComputeComparison(FrozenModel):
    label: str = Field(min_length=1)
    mean_stop_visits: float = Field(gt=0.0)
    fixed_kullback_leibler_at_equal_compute: float | None
    kullback_leibler_advantage: float | None
    equivalent_fixed_visits: float | None
    visit_saving: float | None


class FidelityReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    configuration_path: Path
    model_path: Path
    model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    generation: int = Field(ge=0)
    position_sample_path: Path
    position_sample_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    grid_path: Path
    grid_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    positions: int = Field(gt=0)
    reference_visits: int = Field(gt=0)
    observation_interval: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    exploration_constant: float = Field(gt=0.0)
    virtual_loss_weight: float = Field(ge=0.0, le=1.0)
    forced_playout_coefficient: float = Field(ge=0.0)
    search_value_discount_per_ply: float = Field(gt=0.0, le=1.0)
    reference_seconds: float = Field(gt=0.0)
    reference_simulations: int = Field(gt=0)
    reference_simulations_per_second: float = Field(gt=0.0)
    inference_average_batch_size: float = Field(gt=0.0)
    policy_probability_floor: float = Field(gt=0.0)
    rules: tuple[RuleFidelity, ...] = Field(min_length=1)
    equal_compute_comparisons: tuple[EqualComputeComparison, ...]


def _reference_budget(arguments: Arguments) -> AdaptiveFullSearchBudget:
    # Minimum equal to maximum makes the deterministic stop unreachable, so the trace always runs to the end.
    return AdaptiveFullSearchBudget(
        kind='adaptive',
        minimum_visits=arguments.reference_visits,
        maximum_visits=arguments.reference_visits,
        observation_interval=arguments.observation_interval,
        leader_stability_window=arguments.observation_interval,
        root_value_tolerance=0.0,
        initial_top_visit_share=1.0,
        final_top_visit_share=1.0,
        initial_top_two_margin=1.0,
        final_top_two_margin=1.0,
        threshold_relaxation_visits=arguments.reference_visits,
        minimum_search_correction_to_unlock_tail=None,
    )


def _reference_parameters(game: ChessImplementation, arguments: Arguments) -> ResolvedSelfPlayParameters:
    baseline = game.self_play_parameters_at(arguments.generation)
    # Root noise is exploration, not target shape, so the fidelity comparison runs without it.
    return replace(
        baseline,
        parallel_searches=arguments.parallel_searches,
        full_search_budget=_reference_budget(arguments),
        fast_searches=arguments.reference_visits,
        dirichlet_alpha=1.0,
        dirichlet_epsilon=0.0,
    )


def _checkpoints(checkpoints: list[SearchCheckpoint]) -> tuple[_Checkpoint, ...]:
    return tuple(
        _Checkpoint(
            visits=checkpoint.visits,
            leader_action_id=checkpoint.leader_action_id,
            most_visited_action_id=checkpoint.most_visited_action_id,
            top_visit_share=checkpoint.top_visit_share,
            top_two_margin=checkpoint.top_two_margin,
            root_value=checkpoint.root_value,
            policy=tuple((visit.action_id, visit.visit_count) for visit in checkpoint.policy_target_visits),
        )
        for checkpoint in checkpoints
    )


def _probabilities(policy: tuple[tuple[int, int], ...]) -> dict[int, float]:
    total = sum(count for _, count in policy)
    return {action_id: count / total for action_id, count in policy}


def _kullback_leibler(reference: dict[int, float], candidate: dict[int, float]) -> float:
    divergence = 0.0
    for action_id, reference_probability in reference.items():
        candidate_probability = max(candidate.get(action_id, 0.0), _POLICY_PROBABILITY_FLOOR)
        divergence += reference_probability * math.log(reference_probability / candidate_probability)
    return max(divergence, 0.0)


def _total_variation(reference: dict[int, float], candidate: dict[int, float]) -> float:
    return 0.5 * sum(
        abs(reference.get(action_id, 0.0) - candidate.get(action_id, 0.0))
        for action_id in reference.keys() | candidate.keys()
    )


def _fixed_stop(trace: tuple[_Checkpoint, ...], visits: int) -> _Checkpoint:
    for checkpoint in trace:
        if checkpoint.visits >= visits:
            return checkpoint
    return trace[-1]


def _adaptive_stop(trace: tuple[_Checkpoint, ...], rule: AdaptiveStoppingRule, interval: int) -> _Checkpoint:
    if rule.observation_interval % interval:
        raise ValueError(
            f'Rule {rule.label!r} observes every {rule.observation_interval} visits, which the reference '
            f'interval {interval} cannot reproduce; use a multiple of the reference interval.'
        )
    stride = rule.observation_interval // interval
    observed = trace[stride - 1 :: stride]
    history_count = rule.leader_stability_window // rule.observation_interval
    previous_value: float | None = None
    for index, checkpoint in enumerate(observed):
        value_delta = (
            _FIRST_CHECKPOINT_VALUE_DELTA if previous_value is None else abs(checkpoint.root_value - previous_value)
        )
        previous_value = checkpoint.root_value
        if checkpoint.visits >= rule.maximum_visits:
            return checkpoint
        stable = index >= history_count and all(
            earlier.leader_action_id == checkpoint.leader_action_id
            for earlier in observed[index - history_count : index]
        )
        if checkpoint.visits < rule.minimum_visits or not stable or value_delta > rule.root_value_tolerance:
            continue
        progress = min(
            1.0,
            max(0.0, (checkpoint.visits - rule.minimum_visits) / rule.threshold_relaxation_visits),
        )
        required_share = rule.initial_top_visit_share + progress * (
            rule.final_top_visit_share - rule.initial_top_visit_share
        )
        required_margin = rule.initial_top_two_margin + progress * (
            rule.final_top_two_margin - rule.initial_top_two_margin
        )
        if checkpoint.top_visit_share >= required_share or checkpoint.top_two_margin >= required_margin:
            return checkpoint
    return observed[-1] if observed else trace[-1]


def _accumulate(
    accumulator: _RuleAccumulator,
    candidate: _Checkpoint,
    reference: _Checkpoint,
    maximum_visits: int,
) -> None:
    reference_policy = _probabilities(reference.policy)
    candidate_policy = _probabilities(candidate.policy)
    accumulator.positions += 1
    accumulator.stop_visits += candidate.visits
    accumulator.stop_visit_samples.append(candidate.visits)
    accumulator.maximum_reached += candidate.visits >= maximum_visits
    accumulator.leader_agreements += candidate.leader_action_id == reference.leader_action_id
    accumulator.most_visited_agreements += candidate.most_visited_action_id == reference.most_visited_action_id
    accumulator.kullback_leibler += _kullback_leibler(reference_policy, candidate_policy)
    accumulator.total_variation += _total_variation(reference_policy, candidate_policy)
    accumulator.root_value_error += abs(candidate.root_value - reference.root_value)


def _rule_fidelity(rule: StoppingRule, accumulator: _RuleAccumulator) -> RuleFidelity:
    positions = accumulator.positions
    return RuleFidelity(
        label=rule.label,
        kind=rule.kind,
        positions=positions,
        mean_stop_visits=accumulator.stop_visits / positions,
        median_stop_visits=statistics.median(accumulator.stop_visit_samples),
        maximum_reached_fraction=accumulator.maximum_reached / positions,
        policy_leader_agreement=accumulator.leader_agreements / positions,
        most_visited_agreement=accumulator.most_visited_agreements / positions,
        mean_policy_kullback_leibler=accumulator.kullback_leibler / positions,
        mean_policy_total_variation=accumulator.total_variation / positions,
        mean_root_value_absolute_error=accumulator.root_value_error / positions,
    )


def _interpolate(points: tuple[tuple[float, float], ...], at: float) -> float | None:
    if len(points) < 2 or at < points[0][0] or at > points[-1][0]:
        return None
    index = bisect_left([x for x, _ in points], at)
    if index == 0:
        return points[0][1]
    left_x, left_y = points[index - 1]
    right_x, right_y = points[index]
    if right_x == left_x:
        return left_y
    return left_y + (right_y - left_y) * (at - left_x) / (right_x - left_x)


def _equal_compute_comparisons(rules: tuple[RuleFidelity, ...]) -> tuple[EqualComputeComparison, ...]:
    fixed = sorted(
        ((rule.mean_stop_visits, rule.mean_policy_kullback_leibler) for rule in rules if rule.kind == 'fixed'),
    )
    # Divergence falls as visits rise, so the inverse curve is read from the reversed pairs.
    inverse = tuple(sorted((divergence, visits) for visits, divergence in fixed))
    comparisons: list[EqualComputeComparison] = []
    for rule in rules:
        if rule.kind != 'adaptive':
            continue
        fixed_divergence = _interpolate(tuple(fixed), rule.mean_stop_visits)
        equivalent_visits = _interpolate(inverse, rule.mean_policy_kullback_leibler)
        comparisons.append(
            EqualComputeComparison(
                label=rule.label,
                mean_stop_visits=rule.mean_stop_visits,
                fixed_kullback_leibler_at_equal_compute=fixed_divergence,
                kullback_leibler_advantage=(
                    None if fixed_divergence is None else fixed_divergence - rule.mean_policy_kullback_leibler
                ),
                equivalent_fixed_visits=equivalent_visits,
                visit_saving=(None if equivalent_visits is None else equivalent_visits - rule.mean_stop_visits),
            )
        )
    return tuple(comparisons)


def measure_fidelity(arguments: Arguments) -> FidelityReport:
    sample = PositionSample.model_validate_json(arguments.positions.read_text(encoding='utf-8'))
    grid = STOPPING_RULE_GRID_ADAPTER.validate_json(arguments.grid.read_text(encoding='utf-8'))
    fens = tuple(position.fen for position in sample.positions)
    if arguments.position_limit is not None:
        fens = fens[: arguments.position_limit]
    for rule in grid.rules:
        if rule.kind == 'fixed' and rule.visits > arguments.reference_visits:
            raise ValueError(f'Rule {rule.label!r} asks for more visits than the reference trace holds.')
        if rule.kind == 'adaptive' and rule.maximum_visits > arguments.reference_visits:
            raise ValueError(f'Rule {rule.label!r} allows more visits than the reference trace holds.')

    from AlphaZeroCpp import (
        BatchedInferenceParameters,
        ChessPosition,
        ChessSelfPlaySearch,
        ChessSelfPlaySearchRequest,
        SearchCheckpointDetail,
    )

    configuration = load_chess_experiment_configuration(arguments.configuration)
    game = ChessImplementation(configuration)
    parameters = _reference_parameters(game, arguments)
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
    accumulators = {rule.label: _RuleAccumulator() for rule in grid.rules}
    simulations = 0
    elapsed = 0.0
    for start in range(0, len(fens), arguments.chunk_positions):
        chunk = fens[start : start + arguments.chunk_positions]
        roots = [search.new_root(ChessPosition(fen)) for fen in chunk]
        started = time.perf_counter()
        batch = search.search(
            [ChessSelfPlaySearchRequest(root, True, SearchCheckpointDetail.POLICIES) for root in roots]
        )
        elapsed += time.perf_counter() - started
        simulations += sum(result.final_visits - result.starting_visits for result in batch.results)
        for result in batch.results:
            trace = _checkpoints(result.checkpoints)
            if not trace:
                continue
            reference = trace[-1]
            for rule in grid.rules:
                match rule:
                    case FixedStoppingRule(visits=visits):
                        candidate = _fixed_stop(trace, visits)
                        maximum_visits = visits
                    case AdaptiveStoppingRule():
                        candidate = _adaptive_stop(trace, rule, arguments.observation_interval)
                        maximum_visits = rule.maximum_visits
                _accumulate(accumulators[rule.label], candidate, reference, maximum_visits)
    inference_statistics = search.inference_statistics()
    game.close()

    rules = tuple(_rule_fidelity(rule, accumulators[rule.label]) for rule in grid.rules)
    return FidelityReport(
        source_revision=read_source_revision().commit,
        tool_sha256=file_sha256(Path(__file__)),
        configuration_path=arguments.configuration.resolve(),
        model_path=arguments.model.resolve(),
        model_sha256=file_sha256(arguments.model),
        generation=arguments.generation,
        position_sample_path=arguments.positions.resolve(),
        position_sample_sha256=file_sha256(arguments.positions),
        grid_path=arguments.grid.resolve(),
        grid_sha256=file_sha256(arguments.grid),
        positions=len(fens),
        reference_visits=arguments.reference_visits,
        observation_interval=arguments.observation_interval,
        parallel_searches=arguments.parallel_searches,
        exploration_constant=parameters.exploration_constant,
        virtual_loss_weight=parameters.virtual_loss_weight,
        forced_playout_coefficient=parameters.forced_playout_coefficient,
        search_value_discount_per_ply=parameters.value_discount_per_ply,
        reference_seconds=elapsed,
        reference_simulations=simulations,
        reference_simulations_per_second=simulations / elapsed,
        inference_average_batch_size=inference_statistics.averageNumberOfPositionsInInferenceCall,
        policy_probability_floor=_POLICY_PROBABILITY_FLOOR,
        rules=rules,
        equal_compute_comparisons=_equal_compute_comparisons(rules),
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Measure policy-target fidelity per visit for search stopping rules.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--reference-visits', type=int, default=10000)
    parser.add_argument('--observation-interval', type=int, default=50)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--chunk-positions', type=int, default=128)
    parser.add_argument('--inference-batch-size', type=int, default=256)
    parser.add_argument('--position-limit', type=int)
    parsed = parser.parse_args()
    if min(parsed.generation, parsed.device) < 0:
        parser.error('--generation and --device must be nonnegative')
    positive = (
        parsed.reference_visits,
        parsed.observation_interval,
        parsed.parallel_searches,
        parsed.chunk_positions,
        parsed.inference_batch_size,
    )
    if min(positive) <= 0:
        parser.error('visit, interval, parallel, chunk and batch values must be positive')
    if parsed.reference_visits % parsed.observation_interval:
        parser.error('--reference-visits must be a whole number of observation intervals')
    if parsed.reference_visits <= parsed.parallel_searches:
        parser.error('--reference-visits must exceed --parallel-searches')
    if parsed.position_limit is not None and parsed.position_limit <= 0:
        parser.error('--position-limit must be positive')
    return Arguments(
        configuration=parsed.configuration,
        model=parsed.model,
        positions=parsed.positions,
        grid=parsed.grid,
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
    report = measure_fidelity(arguments)
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    print(arguments.output)


if __name__ == '__main__':
    main()
