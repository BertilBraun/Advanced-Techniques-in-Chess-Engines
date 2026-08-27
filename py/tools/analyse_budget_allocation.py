from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision
from tools.measure_policy_target_fidelity import PerPositionReport

_LAGRANGE_MULTIPLIERS = tuple(10.0 ** (-6.0 + 0.05 * step) for step in range(160))


@dataclass(frozen=True)
class Arguments:
    per_position: Path
    output: Path
    signal_visits: int
    target_visits: int


class FrontierPoint(FrozenModel):
    mean_visits: float = Field(gt=0.0)
    mean_kullback_leibler: float = Field(ge=0.0)


class AllocationComparison(FrozenModel):
    mean_visits: float = Field(gt=0.0)
    flat_kullback_leibler: float = Field(ge=0.0)
    oracle_kullback_leibler: float = Field(ge=0.0)
    # How much any perfectly informed budget allocator could gain over a flat budget of the same cost.
    oracle_gain: float
    oracle_gain_as_visit_saving: float | None


class SignalQuality(FrozenModel):
    signal_visits: int = Field(gt=0)
    target_visits: int = Field(gt=0)
    positions: int = Field(gt=0)
    feature: str = Field(min_length=1)
    spearman_with_benefit: float = Field(ge=-1.0, le=1.0)
    top_decile_mean_benefit: float
    bottom_decile_mean_benefit: float
    population_mean_benefit: float


class AllocationReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    per_position_path: Path
    per_position_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    positions: int = Field(gt=0)
    reference_visits: int = Field(gt=0)
    budgets: tuple[int, ...] = Field(min_length=2)
    flat_frontier: tuple[FrontierPoint, ...] = Field(min_length=2)
    oracle_frontier: tuple[FrontierPoint, ...] = Field(min_length=2)
    comparisons: tuple[AllocationComparison, ...] = Field(min_length=1)
    signals: tuple[SignalQuality, ...]


def _spearman(left: list[float], right: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    left_ranks, right_ranks = ranks(left), ranks(right)
    count = len(left)
    left_mean = sum(left_ranks) / count
    right_mean = sum(right_ranks) / count
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True))
    left_variance = sum((a - left_mean) ** 2 for a in left_ranks)
    right_variance = sum((b - right_mean) ** 2 for b in right_ranks)
    if left_variance == 0.0 or right_variance == 0.0:
        return 0.0
    return covariance / (left_variance * right_variance) ** 0.5


def _interpolate(points: tuple[FrontierPoint, ...], at: float) -> float | None:
    if at < points[0].mean_visits or at > points[-1].mean_visits:
        return None
    for left, right in zip(points, points[1:], strict=False):
        if left.mean_visits <= at <= right.mean_visits:
            if right.mean_visits == left.mean_visits:
                return left.mean_kullback_leibler
            fraction = (at - left.mean_visits) / (right.mean_visits - left.mean_visits)
            return left.mean_kullback_leibler + fraction * (right.mean_kullback_leibler - left.mean_kullback_leibler)
    return None


def _inverse(points: tuple[FrontierPoint, ...], divergence: float) -> float | None:
    ordered = sorted(points, key=lambda point: point.mean_kullback_leibler)
    if divergence < ordered[0].mean_kullback_leibler or divergence > ordered[-1].mean_kullback_leibler:
        return None
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.mean_kullback_leibler <= divergence <= right.mean_kullback_leibler:
            span = right.mean_kullback_leibler - left.mean_kullback_leibler
            if span == 0.0:
                return left.mean_visits
            fraction = (divergence - left.mean_kullback_leibler) / span
            return left.mean_visits + fraction * (right.mean_visits - left.mean_visits)
    return None


def analyse(arguments: Arguments) -> AllocationReport:
    report = PerPositionReport.model_validate_json(arguments.per_position.read_text(encoding='utf-8'))
    records = report.records
    budgets = tuple(budget.visits for budget in records[0].budgets)
    divergences = [[budget.kullback_leibler for budget in record.budgets] for record in records]
    count = len(records)

    flat = tuple(
        FrontierPoint(
            mean_visits=float(visits),
            mean_kullback_leibler=sum(row[index] for row in divergences) / count,
        )
        for index, visits in enumerate(budgets)
    )

    # A Lagrangian sweep traces the lower convex envelope of every possible per-position allocation,
    # which is the best any budget predictor could achieve however it is built.
    oracle_points: dict[float, FrontierPoint] = {}
    for multiplier in _LAGRANGE_MULTIPLIERS:
        total_visits = 0.0
        total_divergence = 0.0
        for row in divergences:
            best = min(range(len(budgets)), key=lambda index: row[index] + multiplier * budgets[index])
            total_visits += budgets[best]
            total_divergence += row[best]
        mean_visits = total_visits / count
        point = FrontierPoint(mean_visits=mean_visits, mean_kullback_leibler=total_divergence / count)
        existing = oracle_points.get(round(mean_visits, 3))
        if existing is None or point.mean_kullback_leibler < existing.mean_kullback_leibler:
            oracle_points[round(mean_visits, 3)] = point
    oracle = tuple(sorted(oracle_points.values(), key=lambda point: point.mean_visits))

    comparisons: list[AllocationComparison] = []
    for visits in budgets:
        if visits < oracle[0].mean_visits or visits > oracle[-1].mean_visits:
            continue
        flat_value = _interpolate(flat, float(visits))
        oracle_value = _interpolate(oracle, float(visits))
        if flat_value is None or oracle_value is None:
            continue
        equivalent = _inverse(flat, oracle_value)
        comparisons.append(
            AllocationComparison(
                mean_visits=float(visits),
                flat_kullback_leibler=flat_value,
                oracle_kullback_leibler=oracle_value,
                oracle_gain=flat_value - oracle_value,
                oracle_gain_as_visit_saving=None if equivalent is None else equivalent - visits,
            )
        )

    signal_index = budgets.index(arguments.signal_visits)
    # Benefit is what this position gains by searching on to the target budget, not to the reference,
    # whose divergence is zero everywhere by construction and would make the question circular.
    target_index = budgets.index(arguments.target_visits)
    benefit = [row[signal_index] - row[target_index] for row in divergences]
    signals: list[SignalQuality] = []
    for feature, values in (
        ('top_visit_share', [record.budgets[signal_index].top_visit_share for record in records]),
        ('top_two_margin', [record.budgets[signal_index].top_two_margin for record in records]),
        ('kullback_leibler_at_signal', [row[signal_index] for row in divergences]),
    ):
        ordered = sorted(range(count), key=lambda index: values[index])
        decile = max(1, count // 10)
        signals.append(
            SignalQuality(
                signal_visits=arguments.signal_visits,
                target_visits=arguments.target_visits,
                positions=count,
                feature=feature,
                spearman_with_benefit=_spearman(values, benefit),
                top_decile_mean_benefit=sum(benefit[index] for index in ordered[-decile:]) / decile,
                bottom_decile_mean_benefit=sum(benefit[index] for index in ordered[:decile]) / decile,
                population_mean_benefit=sum(benefit) / count,
            )
        )

    return AllocationReport(
        source_revision=read_source_revision().commit,
        per_position_path=arguments.per_position.resolve(),
        per_position_sha256=file_sha256(arguments.per_position),
        positions=count,
        reference_visits=report.reference_visits,
        budgets=budgets,
        flat_frontier=flat,
        oracle_frontier=oracle,
        comparisons=tuple(comparisons),
        signals=tuple(signals),
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Bound what any per-position search budget allocator could gain.')
    parser.add_argument('--per-position', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--signal-visits', type=int, default=200)
    parser.add_argument('--target-visits', type=int, default=600)
    parsed = parser.parse_args()
    if parsed.signal_visits <= 0 or parsed.target_visits <= parsed.signal_visits:
        parser.error('--signal-visits must be positive and below --target-visits')
    return Arguments(
        per_position=parsed.per_position,
        output=parsed.output,
        signal_visits=parsed.signal_visits,
        target_visits=parsed.target_visits,
    )


def main() -> None:
    arguments = parse_arguments()
    report = analyse(arguments)
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    print(arguments.output)


if __name__ == '__main__':
    main()
