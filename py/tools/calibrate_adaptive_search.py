from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from AlphaZeroCpp import (
    BatchedInferenceParameters,
    ChessPosition,
    ChessSelfPlaySearch,
    ChessSelfPlaySearchRequest,
    InferenceConfiguration,
    SearchCheckpoint,
)
from pydantic import Field
from src.experiment.configuration import load_chess_experiment_configuration
from src.experiment.generation_schedule import defined_schedule_values
from src.games.chess.training import ChessImplementation
from src.self_play.configuration import AdaptiveFullSearchBudgetConfiguration, FixedFullSearchBudgetConfiguration
from src.self_play.parameters import AdaptiveFullSearchBudget, FixedFullSearchBudget
from src.util.frozen_model import FrozenModel


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    model: Path
    openings: Path
    output: Path
    generation: int
    device: int
    positions: int
    batch_size: int
    reference_visits: int
    thresholds: tuple[float, ...]


class PolicyEntry(FrozenModel):
    action_id: int = Field(ge=0)
    probability: float = Field(ge=0.0, le=1.0)


class SearchSnapshot(FrozenModel):
    visits: int = Field(gt=0)
    selected_action_id: int = Field(ge=0)
    policy_leader_action_id: int = Field(ge=0)
    root_value: float = Field(ge=-1.0, le=1.0)
    predicted_search_correction: float = Field(ge=0.0, le=1.0)
    policy: tuple[PolicyEntry, ...]


class PositionAudit(FrozenModel):
    fen: str = Field(min_length=1)
    snapshots: tuple[SearchSnapshot, ...] = Field(min_length=1)


class CandidateMetrics(FrozenModel):
    maximum_visits: int = Field(gt=0)
    minimum_search_correction: float = Field(ge=0.0, le=1.0)
    mean_visits: float = Field(gt=0.0)
    maximum_visits_reached_fraction: float = Field(ge=0.0, le=1.0)
    selected_move_agreement: float = Field(ge=0.0, le=1.0)
    mean_policy_total_variation: float = Field(ge=0.0, le=1.0)
    mean_root_value_error: float = Field(ge=0.0, le=2.0)


class StopReasonCount(FrozenModel):
    reason: str = Field(min_length=1)
    count: int = Field(gt=0)


class LiveCanaryMetrics(FrozenModel):
    mean_visits: float = Field(gt=0.0)
    simulations_per_second: float = Field(gt=0.0)
    inference_average_batch_size: float = Field(gt=0.0)
    stop_reasons: tuple[StopReasonCount, ...]


class CalibrationReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    model: str = Field(min_length=1)
    generation: int = Field(ge=0)
    positions: int = Field(gt=0)
    reference_visits: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    simulations_per_second: float = Field(gt=0.0)
    inference_average_batch_size: float = Field(gt=0.0)
    live_canary: LiveCanaryMetrics
    candidates: tuple[CandidateMetrics, ...]
    audits: tuple[PositionAudit, ...]


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Audit adaptive Chess stopping against continuous reference searches.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--openings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--positions', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--reference-visits', type=int, default=3200)
    parser.add_argument('--thresholds', type=_parse_thresholds, default=(0.4, 0.5, 0.6, 0.7))
    parsed = parser.parse_args()
    if min(parsed.generation, parsed.device) < 0:
        parser.error('--generation and --device must be nonnegative')
    if min(parsed.positions, parsed.batch_size, parsed.reference_visits) <= 0:
        parser.error('--positions, --batch-size, and --reference-visits must be positive')
    return Arguments(
        configuration=parsed.configuration,
        model=parsed.model,
        openings=parsed.openings,
        output=parsed.output,
        generation=parsed.generation,
        device=parsed.device,
        positions=parsed.positions,
        batch_size=parsed.batch_size,
        reference_visits=parsed.reference_visits,
        thresholds=parsed.thresholds,
    )


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('thresholds must be comma-separated numbers') from error
    if not thresholds or any(item < 0.0 or item > 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError('thresholds must be comma-separated values in [0, 1]')
    return thresholds


def _load_openings(path: Path, count: int) -> tuple[str, ...]:
    openings = tuple(
        line.rsplit('\t', maxsplit=1)[1]
        for line in path.read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#')
    )
    if len(openings) < count:
        raise ValueError(f'Opening suite contains {len(openings)} positions, fewer than requested {count}.')
    return openings[:count]


def _snapshot(checkpoint: SearchCheckpoint, predicted_search_correction: float) -> SearchSnapshot:
    policy_visits = checkpoint.policy_target_visits
    total = sum(visit.visit_count for visit in policy_visits)
    leader = max(policy_visits, key=lambda visit: (visit.visit_count, -visit.action_id))
    return SearchSnapshot(
        visits=checkpoint.visits,
        selected_action_id=checkpoint.leader_action_id,
        policy_leader_action_id=leader.action_id,
        root_value=checkpoint.root_value,
        predicted_search_correction=predicted_search_correction,
        policy=tuple(
            PolicyEntry(action_id=visit.action_id, probability=visit.visit_count / total) for visit in policy_visits
        ),
    )


def collect_audits(arguments: Arguments) -> tuple[tuple[PositionAudit, ...], float, float, float]:
    configuration = load_chess_experiment_configuration(arguments.configuration)
    game = ChessImplementation(configuration)
    parameters = game.self_play_parameters_at(arguments.generation)
    match parameters.full_search_budget:
        case AdaptiveFullSearchBudget() as adaptive:
            pass
        case FixedFullSearchBudget():
            raise ValueError('Calibration requires an adaptive full-search configuration.')
    if arguments.reference_visits < adaptive.maximum_visits:
        raise ValueError('Reference visits must cover the configured adaptive maximum.')
    if (arguments.reference_visits - adaptive.minimum_visits) % adaptive.observation_interval:
        raise ValueError('Reference visits must align with the adaptive observation interval.')
    audit_budget = replace(
        adaptive,
        maximum_visits=arguments.reference_visits,
        leader_stability_window=arguments.reference_visits + adaptive.observation_interval,
        root_value_tolerance=0.0,
        initial_top_visit_share=1.0,
        final_top_visit_share=1.0,
        initial_top_two_margin=1.0,
        final_top_two_margin=1.0,
        minimum_search_correction_to_unlock_tail=None,
    )
    audit_parameters = replace(parameters, full_search_budget=audit_budget, fast_searches=arguments.reference_visits)
    openings = _load_openings(arguments.openings, arguments.positions)
    search = ChessSelfPlaySearch(
        InferenceConfiguration(arguments.device, str(arguments.model)),
        game.native_search_parameters(audit_parameters),
        BatchedInferenceParameters(1, arguments.batch_size, 1),
    )
    roots = [search.new_root(ChessPosition(fen)) for fen in openings]
    started = time.perf_counter()
    batch = search.search([ChessSelfPlaySearchRequest(root, True) for root in roots])
    elapsed = time.perf_counter() - started
    statistics = search.inference_statistics()
    game.close()
    audits = tuple(
        PositionAudit(
            fen=fen,
            snapshots=tuple(
                _snapshot(checkpoint, result.predicted_search_correction) for checkpoint in result.checkpoints
            ),
        )
        for fen, result in zip(openings, batch.results, strict=True)
    )
    simulations = sum(result.final_visits - result.starting_visits for result in batch.results)
    return audits, elapsed, simulations / elapsed, statistics.averageNumberOfPositionsInInferenceCall


def run_live_canary(arguments: Arguments) -> LiveCanaryMetrics:
    configuration = load_chess_experiment_configuration(arguments.configuration)
    game = ChessImplementation(configuration)
    openings = _load_openings(arguments.openings, arguments.positions)
    search = ChessSelfPlaySearch(
        InferenceConfiguration(arguments.device, str(arguments.model)),
        game.native_search_parameters(game.self_play_parameters_at(arguments.generation)),
        BatchedInferenceParameters(1, arguments.batch_size, 1),
    )
    roots = [search.new_root(ChessPosition(fen)) for fen in openings]
    started = time.perf_counter()
    batch = search.search([ChessSelfPlaySearchRequest(root, True) for root in roots])
    elapsed = time.perf_counter() - started
    statistics = search.inference_statistics()
    game.close()
    simulations = sum(result.final_visits - result.starting_visits for result in batch.results)
    reason_names = tuple(result.stop_reason.name.lower() for result in batch.results)
    return LiveCanaryMetrics(
        mean_visits=sum(result.final_visits for result in batch.results) / len(batch.results),
        simulations_per_second=simulations / elapsed,
        inference_average_batch_size=statistics.averageNumberOfPositionsInInferenceCall,
        stop_reasons=tuple(
            StopReasonCount(reason=reason, count=reason_names.count(reason)) for reason in sorted(set(reason_names))
        ),
    )


def _deterministic_stop(
    snapshots: tuple[SearchSnapshot, ...],
    index: int,
    adaptive: AdaptiveFullSearchBudget,
) -> bool:
    history_count = adaptive.leader_stability_window // adaptive.observation_interval
    if index < history_count:
        return False
    current = snapshots[index]
    leaders = {snapshot.policy_leader_action_id for snapshot in snapshots[index - history_count : index + 1]}
    if len(leaders) != 1 or abs(current.root_value - snapshots[index - 1].root_value) > adaptive.root_value_tolerance:
        return False
    progress = min(1.0, (current.visits - adaptive.minimum_visits) / adaptive.threshold_relaxation_visits)
    top_share, margin = _top_statistics(current.policy)
    required_share = adaptive.initial_top_visit_share + progress * (
        adaptive.final_top_visit_share - adaptive.initial_top_visit_share
    )
    required_margin = adaptive.initial_top_two_margin + progress * (
        adaptive.final_top_two_margin - adaptive.initial_top_two_margin
    )
    return top_share >= required_share or margin >= required_margin


def _top_statistics(policy: tuple[PolicyEntry, ...]) -> tuple[float, float]:
    probabilities = sorted((entry.probability for entry in policy), reverse=True)
    second = probabilities[1] if len(probabilities) > 1 else 0.0
    return probabilities[0], probabilities[0] - second


def _selected_snapshot(
    audit: PositionAudit,
    adaptive: AdaptiveFullSearchBudget,
    maximum_visits: int,
    threshold: float,
) -> SearchSnapshot:
    gate_visit = adaptive.minimum_visits + (maximum_visits - adaptive.minimum_visits) // 2
    for index, snapshot in enumerate(audit.snapshots):
        if snapshot.visits > maximum_visits:
            break
        if _deterministic_stop(audit.snapshots, index, adaptive):
            return snapshot
        if snapshot.visits >= gate_visit and snapshot.predicted_search_correction < threshold:
            return snapshot
        if snapshot.visits >= maximum_visits:
            return snapshot
    raise ValueError(f'Audit omitted required maximum visit snapshot {maximum_visits}.')


def _policy_total_variation(left: SearchSnapshot, right: SearchSnapshot) -> float:
    left_policy = {entry.action_id: entry.probability for entry in left.policy}
    right_policy = {entry.action_id: entry.probability for entry in right.policy}
    return 0.5 * sum(
        abs(left_policy.get(action, 0.0) - right_policy.get(action, 0.0)) for action in left_policy | right_policy
    )


def candidate_metrics(
    audits: tuple[PositionAudit, ...],
    adaptive: AdaptiveFullSearchBudget,
    maximum_visits: int,
    threshold: float,
) -> CandidateMetrics:
    selected = tuple(_selected_snapshot(audit, adaptive, maximum_visits, threshold) for audit in audits)
    references = tuple(audit.snapshots[-1] for audit in audits)
    return CandidateMetrics(
        maximum_visits=maximum_visits,
        minimum_search_correction=threshold,
        mean_visits=sum(snapshot.visits for snapshot in selected) / len(selected),
        maximum_visits_reached_fraction=sum(snapshot.visits >= maximum_visits for snapshot in selected) / len(selected),
        selected_move_agreement=sum(
            snapshot.selected_action_id == reference.selected_action_id
            for snapshot, reference in zip(selected, references, strict=True)
        )
        / len(selected),
        mean_policy_total_variation=sum(
            _policy_total_variation(snapshot, reference)
            for snapshot, reference in zip(selected, references, strict=True)
        )
        / len(selected),
        mean_root_value_error=sum(
            abs(snapshot.root_value - reference.root_value)
            for snapshot, reference in zip(selected, references, strict=True)
        )
        / len(selected),
    )


def main() -> None:
    arguments = parse_arguments()
    configuration = load_chess_experiment_configuration(arguments.configuration)
    match configuration.chess.self_play.search.full_search_budget:
        case AdaptiveFullSearchBudgetConfiguration() as budget_configuration:
            adaptive = budget_configuration.resolve(arguments.generation)
        case FixedFullSearchBudgetConfiguration():
            raise ValueError('Calibration requires an adaptive full-search configuration.')
    audits, elapsed, simulations_per_second, average_batch_size = collect_audits(arguments)
    maxima = tuple(
        maximum
        for maximum in sorted(set(defined_schedule_values(budget_configuration.maximum_visits)))
        if adaptive.minimum_visits < maximum <= arguments.reference_visits
        and (maximum - adaptive.minimum_visits) % (2 * adaptive.observation_interval) == 0
    )
    report = CalibrationReport(
        source_revision=subprocess.run(
            ('git', 'rev-parse', 'HEAD'),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        model=str(arguments.model),
        generation=arguments.generation,
        positions=len(audits),
        reference_visits=arguments.reference_visits,
        elapsed_seconds=elapsed,
        simulations_per_second=simulations_per_second,
        inference_average_batch_size=average_batch_size,
        live_canary=run_live_canary(arguments),
        candidates=tuple(
            candidate_metrics(audits, adaptive, maximum, threshold)
            for maximum in maxima
            for threshold in arguments.thresholds
        ),
        audits=audits,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')
    print(arguments.output)


if __name__ == '__main__':
    main()
