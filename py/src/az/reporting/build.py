from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.az.calibration.models import SearchTraceCollectionArtifact
from src.az.config.base import Sha256
from src.az.evaluation.models import EvaluationPairResult
from src.az.evaluation.statistics import (
    LearningCurvePoint,
    learning_curve_statistics,
    summarize_match,
)
from src.az.replay.envelope import ReplayEnvelope, SearchStopReason, SearchStrategy
from src.az.reporting.models import (
    AvailableDistributionEvidence,
    AvailableMetricEvidence,
    AvailablePrefixDisagreement,
    CategoryCount,
    CheckpointTimingEvidence,
    EvaluationProtocolIdentity,
    MetricEvidence,
    PrefixDisagreementAtCheckpoint,
    ReportRun,
    ResearchReport,
    ResearchReportArtifact,
    RunDiagnostics,
    RunIdentity,
    UnavailableMetricEvidence,
    UnavailablePrefixDisagreement,
)


@dataclass(frozen=True)
class EvaluationCheckpointEvidence:
    elapsed_hours: float
    pairs: tuple[EvaluationPairResult, ...]
    bootstrap_samples: int
    confidence_level: float
    bootstrap_seed: int


@dataclass(frozen=True)
class RunReportEvidence:
    identity: RunIdentity
    committed_replay_envelopes: tuple[ReplayEnvelope, ...]
    evaluation_checkpoints: tuple[EvaluationCheckpointEvidence, ...]
    checkpoint_timing: tuple[CheckpointTimingEvidence, ...]
    optimizer_steps: int | None
    replay_reuse: float | None
    gpu_utilization_percent: float | None
    source_artifact_sha256s: tuple[Sha256, ...]
    search_trace_artifacts: tuple[SearchTraceCollectionArtifact, ...] = ()

    def __post_init__(self) -> None:
        elapsed_hours = tuple(checkpoint.elapsed_hours for checkpoint in self.evaluation_checkpoints)
        if tuple(sorted(set(elapsed_hours))) != elapsed_hours:
            raise ValueError('Evaluation checkpoints must have unique strictly increasing elapsed hours.')


def _available(value: float, unit: str, source: str) -> AvailableMetricEvidence:
    return AvailableMetricEvidence(kind='available', value=value, unit=unit, source=source)


def _optional(value: float | None, unit: str, source: str, reason: str) -> MetricEvidence:
    return (
        UnavailableMetricEvidence(kind='unavailable', reason=reason)
        if value is None
        else _available(value, unit, source)
    )


def _mean_optional(values: tuple[float | None, ...], unit: str, source: str, reason: str) -> MetricEvidence:
    available = tuple(value for value in values if value is not None)
    if not available:
        return UnavailableMetricEvidence(kind='unavailable', reason=reason)
    return _available(sum(available) / len(available), unit, source)


def _quantile(values: tuple[int, ...], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _diagnostics(evidence: RunReportEvidence) -> RunDiagnostics:
    envelopes = evidence.committed_replay_envelopes
    committed_sample_ids = {envelope.sample_id for envelope in envelopes}
    if any(
        artifact.payload.replay_sample_id not in committed_sample_ids for artifact in evidence.search_trace_artifacts
    ):
        raise ValueError('Report trace evidence must reference a committed replay sample.')
    evaluation_games = tuple(
        game for checkpoint in evidence.evaluation_checkpoints for pair in checkpoint.pairs for game in pair.games
    )
    budget_categories = tuple(
        CategoryCount(
            category=budget.value,
            count=sum(envelope.budget_class is budget for envelope in envelopes),
        )
        for budget in sorted({envelope.budget_class for envelope in envelopes}, key=lambda item: item.value)
    )
    stop_categories = tuple(
        CategoryCount(
            category=reason.value,
            count=sum(envelope.stop_reason is reason for envelope in envelopes),
        )
        for reason in sorted({envelope.stop_reason for envelope in envelopes}, key=lambda item: item.value)
    )
    adaptive_positions = tuple(
        envelope for envelope in envelopes if envelope.search_strategy is SearchStrategy.ADAPTIVE
    )
    early_stops = sum(envelope.stop_reason is SearchStopReason.ADAPTIVE_CONFIDENCE for envelope in adaptive_positions)
    trace_checkpoints = tuple(
        sorted(
            {
                prefix.simulations
                for artifact in evidence.search_trace_artifacts
                for prefix in artifact.payload.observation.prefixes
            }
        )
    )
    prefix_disagreement = (
        AvailablePrefixDisagreement(
            kind='available',
            checkpoints=tuple(
                _prefix_disagreement_at_checkpoint(evidence.search_trace_artifacts, checkpoint)
                for checkpoint in trace_checkpoints
            ),
            source='committed sampled prefix traces',
        )
        if trace_checkpoints
        else UnavailablePrefixDisagreement(
            kind='unavailable',
            reason='No committed sampled prefix traces were supplied.',
        )
    )
    simulations = tuple(envelope.actual_simulations for envelope in envelopes)
    no_positions = UnavailableMetricEvidence(
        kind='unavailable',
        reason='No committed searched positions were supplied.',
    )
    return RunDiagnostics(
        committed_games=_available(len({envelope.game_id for envelope in envelopes}), 'games', 'committed replay'),
        committed_positions=_available(len(envelopes), 'positions', 'committed replay'),
        actual_simulations=_available(
            sum(envelope.actual_simulations for envelope in envelopes),
            'simulations',
            'committed replay',
        ),
        mean_actual_simulations_per_move=(
            _available(sum(simulations) / len(simulations), 'simulations per move', 'committed replay')
            if simulations
            else no_positions
        ),
        p50_actual_simulations_per_move=(
            _available(_quantile(simulations, 0.5), 'simulations per move', 'committed replay')
            if simulations
            else no_positions
        ),
        p95_actual_simulations_per_move=(
            _available(_quantile(simulations, 0.95), 'simulations per move', 'committed replay')
            if simulations
            else no_positions
        ),
        budget_class_distribution=AvailableDistributionEvidence(
            kind='available',
            values=budget_categories,
            source='committed replay',
        ),
        policy_eligible_positions=_available(
            sum(envelope.policy_target_eligible for envelope in envelopes),
            'positions',
            'committed replay',
        ),
        policy_eligible_fraction=(
            _available(
                sum(envelope.policy_target_eligible for envelope in envelopes) / len(envelopes),
                'fraction',
                'committed replay',
            )
            if envelopes
            else no_positions
        ),
        policy_weight_sum=_available(
            sum(envelope.policy_target_weight for envelope in envelopes),
            'weight',
            'committed replay',
        ),
        gpu_utilization=_optional(
            evidence.gpu_utilization_percent,
            'percent',
            'resource telemetry',
            'GPU utilization was not recorded.',
        ),
        optimizer_steps=_optional(
            None if evidence.optimizer_steps is None else float(evidence.optimizer_steps),
            'steps',
            'trainer checkpoint',
            'Optimizer-step evidence was not supplied.',
        ),
        replay_reuse=_optional(
            evidence.replay_reuse,
            'uses per position',
            'trainer checkpoint',
            'Replay-reuse evidence was not supplied.',
        ),
        stop_reason_distribution=AvailableDistributionEvidence(
            kind='available',
            values=stop_categories,
            source='committed replay',
        ),
        adaptive_early_stop_frequency=(
            _available(early_stops / len(adaptive_positions), 'fraction', 'committed replay')
            if adaptive_positions
            else UnavailableMetricEvidence(
                kind='unavailable',
                reason='The run has no adaptive-search positions.',
            )
        ),
        prefix_full_disagreement=prefix_disagreement,
        evaluation_games=_available(len(evaluation_games), 'games', 'evaluation results'),
        evaluation_wall_seconds=_available(
            sum(game.evaluation_wall_seconds for game in evaluation_games),
            'seconds',
            'evaluation results',
        ),
        evaluation_actual_simulations=_available(
            sum(game.candidate_actual_simulations + game.opponent_actual_simulations for game in evaluation_games),
            'simulations',
            'evaluation results',
        ),
        checkpoint_timing=evidence.checkpoint_timing,
    )


def _prefix_disagreement_at_checkpoint(
    artifacts: tuple[SearchTraceCollectionArtifact, ...],
    simulations: int,
) -> PrefixDisagreementAtCheckpoint:
    observations = tuple(
        (prefix, artifact.payload.observation.full)
        for artifact in artifacts
        for prefix in artifact.payload.observation.prefixes
        if prefix.simulations == simulations
    )
    policy = tuple(
        0.5
        * sum(
            abs(prefix_probability - full_probability)
            for prefix_probability, full_probability in zip(
                prefix.root_policy,
                full.root_policy,
                strict=True,
            )
        )
        for prefix, full in observations
    )
    values = tuple(abs(prefix.root_value - full.root_value) for prefix, full in observations)
    return PrefixDisagreementAtCheckpoint(
        simulations=simulations,
        observation_count=len(observations),
        mean_policy_total_variation=sum(policy) / len(policy),
        mean_value_absolute_error=sum(values) / len(values),
    )


def build_report(
    *,
    report_id: UUID,
    title: str,
    matrix_id: UUID,
    common_controls_sha256: str,
    runs: tuple[RunReportEvidence, ...],
) -> ResearchReportArtifact:
    if not runs:
        raise ValueError('A research report requires run evidence.')
    report_runs = tuple(_report_run(evidence) for evidence in runs)
    source_hashes = tuple(
        sorted(
            {
                digest
                for evidence in runs
                for digest in (
                    *evidence.source_artifact_sha256s,
                    *(artifact.payload_sha256 for artifact in evidence.search_trace_artifacts),
                )
            }
        )
    )
    payload = ResearchReport(
        report_id=report_id,
        title=title,
        matrix_id=matrix_id,
        common_controls_sha256=common_controls_sha256,
        runs=report_runs,
        independence_note=(
            'Paired games share conditions and are resampled as pairs; games within a pair are not treated as independent.'
        ),
        source_artifact_sha256s=source_hashes,
    )
    return ResearchReportArtifact.create(payload)


def _report_run(evidence: RunReportEvidence) -> ReportRun:
    if not evidence.evaluation_checkpoints:
        raise ValueError('Report run evidence requires evaluation checkpoints.')
    matches = tuple(
        summarize_match(
            checkpoint.pairs,
            checkpoint.bootstrap_samples,
            checkpoint.confidence_level,
            checkpoint.bootstrap_seed,
        )
        for checkpoint in evidence.evaluation_checkpoints
    )
    curve = learning_curve_statistics(
        tuple(
            LearningCurvePoint(
                elapsed_hours=checkpoint.elapsed_hours,
                score=match.mean_score,
                elo=match.elo,
            )
            for checkpoint, match in zip(evidence.evaluation_checkpoints, matches, strict=True)
        )
    )
    games = tuple(
        game for checkpoint in evidence.evaluation_checkpoints for pair in checkpoint.pairs for game in pair.games
    )
    protocols = {
        (
            game.opponent,
            game.common_search_sha256,
            game.board_size,
            game.komi_half_points,
            game.scoring_rule,
            game.ko_rule,
            game.suicide_rule,
        )
        for game in games
    }
    if len(protocols) != 1:
        raise ValueError('Report evaluation evidence must use one homogeneous opponent and Go protocol.')
    opponent, common_search_sha256, board_size, komi, scoring_rule, ko_rule, suicide_rule = next(iter(protocols))
    return ReportRun(
        identity=evidence.identity,
        evaluation_protocol=EvaluationProtocolIdentity(
            opponent=opponent,
            common_search_sha256=common_search_sha256,
            board_size=board_size,
            komi_half_points=komi,
            scoring_rule=scoring_rule,
            ko_rule=ko_rule,
            suicide_rule=suicide_rule,
        ),
        diagnostics=_diagnostics(evidence),
        final_match=matches[-1],
        learning_curve=curve,
    )
