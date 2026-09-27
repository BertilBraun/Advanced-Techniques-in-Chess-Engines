"""Reproduce the plateau arithmetic and clean-cutoff sensitivity from ladder export data."""

from __future__ import annotations

import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from statistics import fmean

from tools.render_ladder_progression import LadderPoint, load_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/ladder-elo-export.json'
OUTPUT_PATH = REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/plateau-comparison.csv'
SECONDS_PER_DAY = 24 * 60 * 60
ESTIMATOR_TRANSFER_ELO = 2.7


@dataclass(frozen=True)
class PlateauComparison:
    window: str
    previous_count: int
    previous_single_rung_mean: float
    transfer_to_three_rung: float
    previous_three_rung_estimate: float
    final_count: int
    final_three_rung_mean: float
    matched_difference: float


def select_points(points: tuple[LadderPoint, ...], start_seconds: int, end_seconds: int) -> tuple[LadderPoint, ...]:
    selected = tuple(point for point in points if start_seconds <= point.seconds <= end_seconds)
    if not selected:
        raise ValueError('Plateau window contains no ladder observations.')
    return selected


def compare_windows(
    label: str,
    previous_points: tuple[LadderPoint, ...],
    final_points: tuple[LadderPoint, ...],
) -> PlateauComparison:
    previous_mean = fmean(point.elo for point in previous_points)
    final_mean = fmean(point.elo for point in final_points)
    matched_previous = previous_mean + ESTIMATOR_TRANSFER_ELO
    return PlateauComparison(
        window=label,
        previous_count=len(previous_points),
        previous_single_rung_mean=round(previous_mean, 4),
        transfer_to_three_rung=ESTIMATOR_TRANSFER_ELO,
        previous_three_rung_estimate=round(matched_previous, 4),
        final_count=len(final_points),
        final_three_rung_mean=round(final_mean, 4),
        matched_difference=round(final_mean - matched_previous, 4),
    )


def main() -> None:
    export = load_source(SOURCE_PATH)
    previous = export.series['v34::evaluation/ladder_elo'].points
    final = export.series['final::evaluation/ladder_elo'].points
    previous_plateau = select_points(previous, round(2.7 * SECONDS_PER_DAY), round(3.9375 * SECONDS_PER_DAY))
    final_plateau = select_points(final, 2 * SECONDS_PER_DAY, final[-1].seconds)
    if len(previous_plateau) != 59 or len(final_plateau) != 86:
        raise ValueError('Historical plateau observation counts changed.')
    comparisons = (
        compare_windows('original_retrospective', previous_plateau, final_plateau),
        compare_windows(
            'publication_cutoff_sensitivity',
            select_points(previous, round(2.7 * SECONDS_PER_DAY), 3 * SECONDS_PER_DAY),
            select_points(final, 2 * SECONDS_PER_DAY, round(2.5 * SECONDS_PER_DAY)),
        ),
    )
    with OUTPUT_PATH.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow([field.name for field in fields(PlateauComparison)])
        writer.writerows(astuple(comparison) for comparison in comparisons)
    for comparison in comparisons:
        print(f'{comparison.window}: {comparison.matched_difference:.1f} Elo')


if __name__ == '__main__':
    main()
