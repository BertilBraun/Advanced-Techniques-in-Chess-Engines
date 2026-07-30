from __future__ import annotations

import csv
import io

from src.az.config.serialization import canonical_json
from src.az.reporting.models import (
    AvailableDistributionEvidence,
    AvailableMetricEvidence,
    AvailablePrefixDisagreement,
    ResearchReport,
    UnavailableDistributionEvidence,
    UnavailableMetricEvidence,
    UnavailablePrefixDisagreement,
)


def render_machine_json(report: ResearchReport) -> str:
    return canonical_json(report) + '\n'


def _metric(evidence: AvailableMetricEvidence | UnavailableMetricEvidence) -> str:
    match evidence:
        case AvailableMetricEvidence(value=value, unit=unit):
            return f'{value:.6g} {unit}'
        case UnavailableMetricEvidence(reason=reason):
            return f'unavailable: {reason}'


def _distribution(evidence: AvailableDistributionEvidence | UnavailableDistributionEvidence) -> str:
    match evidence:
        case AvailableDistributionEvidence(values=values):
            return ', '.join(f'{value.category}={value.count}' for value in values) or 'empty'
        case UnavailableDistributionEvidence(reason=reason):
            return f'unavailable: {reason}'


def render_markdown(report: ResearchReport) -> str:
    lines = [
        f'# {report.title}',
        '',
        report.independence_note,
        '',
        '| Arm | Seed | Score | Elo | Score AUC | Elo AUC | Games | Simulations | GPU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for run in sorted(report.runs, key=lambda item: (item.identity.arm_id.hex, item.identity.seed)):
        lines.append(
            '| '
            + ' | '.join(
                (
                    run.identity.arm_id.hex,
                    str(run.identity.seed),
                    f'{run.final_match.mean_score:.6f}',
                    f'{run.final_match.elo:.3f}',
                    f'{run.learning_curve.score_auc_score_hours:.6f}',
                    f'{run.learning_curve.elo_auc_elo_hours:.3f}',
                    _metric(run.diagnostics.committed_games),
                    _metric(run.diagnostics.actual_simulations),
                    _metric(run.diagnostics.gpu_utilization),
                )
            )
            + ' |'
        )
        diagnostics = run.diagnostics
        lines.extend(
            (
                '',
                f'## Diagnostics: {run.identity.arm_id.hex} / seed {run.identity.seed}',
                '',
                f'- Evaluation opponent: `{canonical_json(run.evaluation_protocol.opponent)}`',
                f'- Common search SHA-256: `{run.evaluation_protocol.common_search_sha256}`',
                f'- Go rules: {run.evaluation_protocol.board_size}x{run.evaluation_protocol.board_size}, '
                f'komi-half-points={run.evaluation_protocol.komi_half_points}, '
                f'{run.evaluation_protocol.scoring_rule}, {run.evaluation_protocol.ko_rule}, '
                f'{run.evaluation_protocol.suicide_rule}',
                f'- Committed positions: {_metric(diagnostics.committed_positions)}',
                f'- Policy eligible positions: {_metric(diagnostics.policy_eligible_positions)}',
                f'- Policy eligible fraction: {_metric(diagnostics.policy_eligible_fraction)}',
                f'- Policy weight sum: {_metric(diagnostics.policy_weight_sum)}',
                f'- Mean simulations/move: {_metric(diagnostics.mean_actual_simulations_per_move)}',
                f'- P50 simulations/move: {_metric(diagnostics.p50_actual_simulations_per_move)}',
                f'- P95 simulations/move: {_metric(diagnostics.p95_actual_simulations_per_move)}',
                f'- Budget classes: {_distribution(diagnostics.budget_class_distribution)}',
                f'- Stop reasons: {_distribution(diagnostics.stop_reason_distribution)}',
                f'- Optimizer steps: {_metric(diagnostics.optimizer_steps)}',
                f'- Replay reuse: {_metric(diagnostics.replay_reuse)}',
                f'- Adaptive early-stop frequency: {_metric(diagnostics.adaptive_early_stop_frequency)}',
                '- Prefix/full disagreement: ' + _prefix_disagreement(diagnostics.prefix_full_disagreement),
                f'- Evaluation games: {_metric(diagnostics.evaluation_games)}',
                f'- Evaluation wall time: {_metric(diagnostics.evaluation_wall_seconds)}',
                f'- Evaluation simulations: {_metric(diagnostics.evaluation_actual_simulations)}',
                '- Checkpoint timing: '
                + (
                    ', '.join(
                        f'{timing.requested_elapsed_seconds}s requested/'
                        f'{timing.published_elapsed_seconds:.6g}s published'
                        for timing in diagnostics.checkpoint_timing
                    )
                    or 'unavailable: no checkpoint timing evidence supplied.'
                ),
            )
        )
    return '\n'.join(lines) + '\n'


def render_csv(report: ResearchReport) -> str:
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(
        (
            'arm_id',
            'seed',
            'score',
            'elo',
            'score_auc_score_hours',
            'elo_auc_elo_hours',
            'wins',
            'draws',
            'losses',
            'opponent',
            'common_search_sha256',
            'go_rules',
        )
    )
    for run in sorted(report.runs, key=lambda item: (item.identity.arm_id.hex, item.identity.seed)):
        writer.writerow(
            (
                run.identity.arm_id.hex,
                run.identity.seed,
                f'{run.final_match.mean_score:.12g}',
                f'{run.final_match.elo:.12g}',
                f'{run.learning_curve.score_auc_score_hours:.12g}',
                f'{run.learning_curve.elo_auc_elo_hours:.12g}',
                run.final_match.wins,
                run.final_match.draws,
                run.final_match.losses,
                canonical_json(run.evaluation_protocol.opponent),
                run.evaluation_protocol.common_search_sha256,
                (
                    f'{run.evaluation_protocol.board_size}x{run.evaluation_protocol.board_size};'
                    f'komi_half_points={run.evaluation_protocol.komi_half_points};'
                    f'{run.evaluation_protocol.scoring_rule};'
                    f'{run.evaluation_protocol.ko_rule};'
                    f'{run.evaluation_protocol.suicide_rule}'
                ),
            )
        )
    return stream.getvalue()


def _prefix_disagreement(
    evidence: AvailablePrefixDisagreement | UnavailablePrefixDisagreement,
) -> str:
    match evidence:
        case AvailablePrefixDisagreement(checkpoints=checkpoints):
            return ', '.join(
                f'{checkpoint.simulations} sims: TV={checkpoint.mean_policy_total_variation:.6g}, '
                f'value={checkpoint.mean_value_absolute_error:.6g}, n={checkpoint.observation_count}'
                for checkpoint in checkpoints
            )
        case UnavailablePrefixDisagreement(reason=reason):
            return f'unavailable: {reason}'
