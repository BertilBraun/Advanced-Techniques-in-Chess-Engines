from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean, median

from src.replay.manager import IngestedCompletedGame, ReplayDescription
from src.self_play.completed_game import SearchObservation, SearchStopReason, TerminationReason
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedgerState


@dataclass(frozen=True)
class TrainingLifecycleTelemetry:
    configured_replay_ratio: float
    observed_replay_ratio: float
    materialized_samples: int
    consumed_presentations: float
    available_presentations: float
    required_presentations_per_quantum: int
    available_quantum_fraction: float
    live_replay_rows: int
    logical_replay_capacity: int
    replay_fill_fraction: float


@dataclass(frozen=True)
class TerminationGameLengthTelemetry:
    reason: TerminationReason
    completed_games: int
    fraction: float
    mean_plies: float | None


@dataclass(frozen=True)
class CompletedGameLengthTelemetry:
    lengths_plies: tuple[int, ...]
    mean_plies: float
    median_plies: float
    p90_plies: float
    p99_plies: float
    maximum_plies: int
    terminations: tuple[TerminationGameLengthTelemetry, ...]


@dataclass(frozen=True)
class AdaptiveSearchTelemetry:
    final_visits: tuple[int, ...]
    new_simulations: tuple[int, ...]
    search_correction_targets: tuple[float, ...]
    search_correction_predictions: tuple[float, ...]
    policy_corrections: tuple[float, ...]
    value_corrections: tuple[float, ...]
    stop_reasons: tuple[tuple[SearchStopReason, int], ...]


def adaptive_search_telemetry(
    games: tuple[IngestedCompletedGame, ...],
) -> AdaptiveSearchTelemetry | None:
    observations = tuple(
        observation
        for game in games
        for observation in game.observations
        if observation.full_search
    )
    if not observations:
        return None
    return AdaptiveSearchTelemetry(
        final_visits=tuple(observation.final_visits for observation in observations),
        new_simulations=tuple(_new_simulations(observation) for observation in observations),
        search_correction_targets=tuple(observation.search_correction_target for observation in observations),
        search_correction_predictions=tuple(
            observation.predicted_search_correction for observation in observations
        ),
        policy_corrections=tuple(observation.policy_correction for observation in observations),
        value_corrections=tuple(observation.value_correction for observation in observations),
        stop_reasons=tuple(
            (reason, sum(observation.stop_reason is reason for observation in observations))
            for reason in SearchStopReason
        ),
    )


def _new_simulations(observation: SearchObservation) -> int:
    return observation.final_visits - observation.starting_visits


def completed_game_length_telemetry(
    games: tuple[IngestedCompletedGame, ...],
) -> CompletedGameLengthTelemetry | None:
    if not games:
        return None
    lengths = tuple(game.length_plies for game in games)
    terminations: list[TerminationGameLengthTelemetry] = []
    for reason in TerminationReason:
        reason_lengths = tuple(game.length_plies for game in games if game.termination_reason is reason)
        terminations.append(
            TerminationGameLengthTelemetry(
                reason=reason,
                completed_games=len(reason_lengths),
                fraction=len(reason_lengths) / len(games),
                mean_plies=fmean(reason_lengths) if reason_lengths else None,
            )
        )
    return CompletedGameLengthTelemetry(
        lengths_plies=lengths,
        mean_plies=fmean(lengths),
        median_plies=median(lengths),
        p90_plies=_percentile(lengths, 0.90),
        p99_plies=_percentile(lengths, 0.99),
        maximum_plies=max(lengths),
        terminations=tuple(terminations),
    )


def _percentile(values: tuple[int, ...], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def training_lifecycle_telemetry(
    state: CreditLedgerState,
    credit: CreditTrainingParams,
    replay: ReplayDescription,
    global_batch_size: int,
) -> TrainingLifecycleTelemetry:
    materialized_samples = state.earned_credits / credit.replay_ratio
    assert materialized_samples == materialized_samples.to_integral_value()
    materialized_sample_count = int(materialized_samples)
    observed_replay_ratio = (
        float(state.consumed_credits / materialized_samples) if materialized_samples > Decimal(0) else 0.0
    )
    required_presentations = credit.presentation_credits_per_quantum(global_batch_size)
    return TrainingLifecycleTelemetry(
        configured_replay_ratio=float(credit.replay_ratio),
        observed_replay_ratio=observed_replay_ratio,
        materialized_samples=materialized_sample_count,
        consumed_presentations=float(state.consumed_credits),
        available_presentations=float(state.available_credits),
        required_presentations_per_quantum=required_presentations,
        available_quantum_fraction=float(state.available_credits / Decimal(required_presentations)),
        live_replay_rows=replay.size,
        logical_replay_capacity=replay.logical_capacity,
        replay_fill_fraction=replay.size / replay.logical_capacity,
    )
