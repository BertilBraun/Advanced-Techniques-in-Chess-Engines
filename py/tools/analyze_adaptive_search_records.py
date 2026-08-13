from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import json
from math import ceil, floor, log2, sqrt
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field


class Visit(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    action_id: int
    visit_count: int = Field(gt=0)


class Wdl(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    win: float
    draw: float
    loss: float


class Identity(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    worker_id: int
    process_instance_id: str
    game_number: int


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    ply: int
    model_generation: int
    policy_target_visits: tuple[Visit, ...]
    root_value: float
    highest_visited_child_action_id: int | None = None
    highest_visited_child_visit_count: int | None = None
    highest_visited_child_q: float | None = None
    selected_action_id: int | None
    full_search: bool
    sample_weight: float
    search_budget: int


class CompletedGame(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int
    identity: Identity
    created_at_seconds: float
    generation_seconds: float
    action_ids: tuple[int, ...]
    observations: tuple[Observation, ...]
    final_wdl: Wdl
    termination_reason: str
    is_resignation_continuation: bool = False
    resignation_threshold: float | None = None


@dataclass(frozen=True)
class SampleInput:
    name: str
    path: Path


@dataclass(frozen=True)
class PositionRow:
    sample: str
    game_key: str
    termination_reason: str
    outcome: str
    game_plies: int
    ply: int
    generation: int
    full_search: bool
    budget: int
    visit_total: int
    top_action_id: int
    top_visits: int
    second_visits: int
    top_mass: float
    margin_mass: float
    entropy_bits: float
    selected_matches_target_top: bool | None
    selected_matches_raw_top: bool | None
    target_top_matches_raw_top: bool | None
    root_value: float
    highest_child_q: float | None
    heuristic_trigger_50: int | None
    heuristic_trigger_75: int | None


@dataclass(frozen=True)
class NumericSummary:
    count: int
    mean: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    p99: float


def parse_sample(value: str) -> SampleInput:
    name, separator, raw_path = value.partition('=')
    if not separator or not name or not raw_path:
        raise ValueError('Samples must use NAME=PATH syntax.')
    path = Path(raw_path)
    if not path.is_dir():
        raise ValueError(f'Sample path is not a directory: {path}')
    return SampleInput(name=name, path=path)


def load_games(sample: SampleInput) -> tuple[CompletedGame, ...]:
    paths = tuple(sorted(sample.path.glob('*.json')))
    if not paths:
        raise ValueError(f'Sample has no JSON games: {sample.path}')
    return tuple(CompletedGame.model_validate_json(path.read_text(encoding='utf-8')) for path in paths)


def outcome_label(wdl: Wdl) -> str:
    values = (('win', wdl.win), ('draw', wdl.draw), ('loss', wdl.loss))
    return max(values, key=lambda item: item[1])[0]


def ordered_visits(observation: Observation) -> tuple[Visit, ...]:
    return tuple(sorted(observation.policy_target_visits, key=lambda visit: (-visit.visit_count, visit.action_id)))


def entropy_bits(visits: Sequence[Visit]) -> float:
    total = sum(visit.visit_count for visit in visits)
    return -sum(
        (visit.visit_count / total) * log2(visit.visit_count / total) for visit in visits if visit.visit_count > 0
    )


def proportional_uncatchable_trigger(observation: Observation, minimum_fraction: float) -> int | None:
    if observation.full_search:
        return None
    visits = ordered_visits(observation)
    total = sum(visit.visit_count for visit in visits)
    second = visits[1].visit_count if len(visits) > 1 else 0
    margin_share = (visits[0].visit_count - second) / total
    proportional_trigger = floor(observation.search_budget / (1.0 + margin_share)) + 1
    minimum_trigger = ceil(observation.search_budget * minimum_fraction)
    return min(observation.search_budget, max(proportional_trigger, minimum_trigger))


def position_rows(sample: SampleInput, games: Iterable[CompletedGame]) -> tuple[PositionRow, ...]:
    rows: list[PositionRow] = []
    for game in games:
        game_key = f'{game.identity.worker_id}:{game.identity.process_instance_id}:{game.identity.game_number}'
        for observation in game.observations:
            visits = ordered_visits(observation)
            total = sum(visit.visit_count for visit in visits)
            second = visits[1].visit_count if len(visits) > 1 else 0
            selected = observation.selected_action_id
            raw_top_action_id = observation.highest_visited_child_action_id
            if raw_top_action_id is None and not observation.full_search:
                raw_top_action_id = visits[0].action_id
            rows.append(
                PositionRow(
                    sample=sample.name,
                    game_key=game_key,
                    termination_reason=game.termination_reason,
                    outcome=outcome_label(game.final_wdl),
                    game_plies=len(game.action_ids),
                    ply=observation.ply,
                    generation=observation.model_generation,
                    full_search=observation.full_search,
                    budget=observation.search_budget,
                    visit_total=total,
                    top_action_id=visits[0].action_id,
                    top_visits=visits[0].visit_count,
                    second_visits=second,
                    top_mass=visits[0].visit_count / total,
                    margin_mass=(visits[0].visit_count - second) / total,
                    entropy_bits=entropy_bits(visits),
                    selected_matches_target_top=None if selected is None else selected == visits[0].action_id,
                    selected_matches_raw_top=(
                        None if selected is None or raw_top_action_id is None else selected == raw_top_action_id
                    ),
                    target_top_matches_raw_top=(
                        None if raw_top_action_id is None else visits[0].action_id == raw_top_action_id
                    ),
                    root_value=observation.root_value,
                    highest_child_q=observation.highest_visited_child_q,
                    heuristic_trigger_50=proportional_uncatchable_trigger(observation, 0.50),
                    heuristic_trigger_75=proportional_uncatchable_trigger(observation, 0.75),
                )
            )
    return tuple(rows)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError('Cannot calculate a quantile of an empty sequence.')
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> NumericSummary:
    if not values:
        return NumericSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return NumericSummary(
        count=len(values),
        mean=fmean(values),
        median=median(values),
        p10=quantile(values, 0.10),
        p25=quantile(values, 0.25),
        p75=quantile(values, 0.75),
        p90=quantile(values, 0.90),
        p95=quantile(values, 0.95),
        p99=quantile(values, 0.99),
    )


def wilson_interval(successes: int, count: int, z_score: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return (0.0, 0.0)
    rate = successes / count
    denominator = 1.0 + z_score**2 / count
    center = (rate + z_score**2 / (2.0 * count)) / denominator
    radius = z_score * sqrt(rate * (1.0 - rate) / count + z_score**2 / (4.0 * count**2)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def generation_band(generation: int) -> str:
    if generation < 10:
        return '0-9'
    if generation < 30:
        return '10-29'
    if generation < 50:
        return '30-49'
    if generation < 70:
        return '50-69'
    if generation < 90:
        return '70-89'
    return '90+'


def ply_band(ply: int) -> str:
    if ply < 40:
        return '0-39'
    if ply < 80:
        return '40-79'
    if ply < 120:
        return '80-119'
    return '120+'


def value_band(root_value: float) -> str:
    magnitude = abs(root_value)
    if magnitude <= 0.30:
        return 'close_abs_le_0.30'
    if magnitude >= 0.70:
        return 'decisive_abs_ge_0.70'
    return 'intermediate'


def rate_record(successes: int, count: int) -> dict[str, float | int | None]:
    lower, upper = wilson_interval(successes, count)
    return {
        'successes': successes,
        'count': count,
        'rate': None if count == 0 else successes / count,
        'ci95_lower': lower,
        'ci95_upper': upper,
    }


def summarize_group(rows: Sequence[PositionRow]) -> dict[str, object]:
    played = tuple(row for row in rows if row.selected_matches_raw_top is not None)
    raw_disagreements = sum(not row.selected_matches_raw_top for row in played)
    raw_target_known = tuple(row for row in rows if row.target_top_matches_raw_top is not None)
    target_raw_disagreements = sum(not row.target_top_matches_raw_top for row in raw_target_known)
    value_known = tuple(row for row in rows if row.highest_child_q is not None)
    value_sign_agreements = sum(
        row.root_value * row.highest_child_q >= 0.0 for row in value_known if row.highest_child_q is not None
    )
    jointly_decisive_values = sum(
        abs(row.root_value) >= 0.70 and abs(row.highest_child_q) >= 0.70
        for row in value_known
        if row.highest_child_q is not None
    )
    result: dict[str, object] = {
        'positions': len(rows),
        'budgets': dict(sorted(Counter(row.budget for row in rows).items())),
        'top_mass': asdict(numeric_summary([row.top_mass for row in rows])),
        'margin_mass': asdict(numeric_summary([row.margin_mass for row in rows])),
        'entropy_bits': asdict(numeric_summary([row.entropy_bits for row in rows])),
        'selected_vs_final_raw_top_disagreement': rate_record(raw_disagreements, len(played)),
        'target_top_vs_raw_top_disagreement': rate_record(target_raw_disagreements, len(raw_target_known)),
        'root_child_value_sign_agreement': rate_record(value_sign_agreements, len(value_known)),
        'root_child_both_absolute_at_least_0.70': rate_record(jointly_decisive_values, len(value_known)),
        'root_child_absolute_difference': asdict(
            numeric_summary(
                [abs(row.root_value - row.highest_child_q) for row in value_known if row.highest_child_q is not None]
            )
        ),
    }
    for threshold in (0.70, 0.80, 0.90):
        count = sum(row.top_mass >= threshold for row in rows)
        result[f'final_top_mass_at_least_{threshold:.2f}'] = rate_record(count, len(rows))
    for minimum in (50, 75):
        triggers = (
            tuple((row, row.heuristic_trigger_50) for row in rows)
            if minimum == 50
            else tuple((row, row.heuristic_trigger_75) for row in rows)
        )
        eligible = tuple((row, trigger) for row, trigger in triggers if trigger is not None)
        saved = [row.budget - trigger for row, trigger in eligible if trigger is not None]
        all_search_saved = [0 if trigger is None else row.budget - trigger for row, trigger in triggers]
        nominal_budget = sum(row.budget for row in rows)
        result[f'fast_proportional_uncatchable_minimum_{minimum}_percent'] = {
            'positions': len(eligible),
            'simulations_saved': asdict(numeric_summary(saved)),
            'fraction_saved': asdict(
                numeric_summary([amount / row.budget for amount, (row, _) in zip(saved, eligible, strict=True)])
            ),
            'all_searches_with_full_savings_set_to_zero': {
                'simulations_saved': asdict(numeric_summary(all_search_saved)),
                'total_saved': sum(all_search_saved),
                'fraction_of_nominal_visit_limits': (
                    0.0 if nominal_budget == 0 else sum(all_search_saved) / nominal_budget
                ),
            },
        }
    return result


def grouped(
    rows: Sequence[PositionRow],
    key: Callable[[PositionRow], str],
) -> dict[str, dict[str, object]]:
    groups: defaultdict[str, list[PositionRow]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: summarize_group(group) for name, group in sorted(groups.items())}


def report(samples: Sequence[SampleInput]) -> dict[str, object]:
    all_rows: list[PositionRow] = []
    sample_records: dict[str, object] = {}
    for sample in samples:
        games = load_games(sample)
        rows = position_rows(sample, games)
        all_rows.extend(rows)
        sample_records[sample.name] = {
            'path': str(sample.path.resolve()),
            'games': len(games),
            'positions': len(rows),
            'generation_minimum': min(row.generation for row in rows),
            'generation_maximum': max(row.generation for row in rows),
            'game_plies': asdict(numeric_summary([len(game.action_ids) for game in games])),
            'termination_reasons': dict(sorted(Counter(game.termination_reason for game in games).items())),
            'outcomes': dict(sorted(Counter(outcome_label(game.final_wdl) for game in games).items())),
            'search_kind': {
                'full': sum(row.full_search for row in rows),
                'fast': sum(not row.full_search for row in rows),
            },
            'summary': summarize_group(rows),
            'by_search_kind': {
                'full': summarize_group(tuple(row for row in rows if row.full_search)),
                'fast': summarize_group(tuple(row for row in rows if not row.full_search)),
            },
            'by_generation_band': grouped(
                tuple(replace(row, sample=generation_band(row.generation)) for row in rows),
                lambda row: row.sample,
            ),
            'by_generation': grouped(rows, lambda row: str(row.generation)),
            'by_ply_band': grouped(
                tuple(replace(row, sample=ply_band(row.ply)) for row in rows),
                lambda row: row.sample,
            ),
            'by_value_band': grouped(rows, lambda row: value_band(row.root_value)),
        }
    return {
        'method': {
            'confidence_intervals': 'Wilson score, two-sided 95%',
            'heuristic': (
                'Fast searches only: extrapolate the final raw visit shares backward as constant; trigger when '
                'the projected top-one lead exceeds all remaining budget, subject to a 50% or 75% minimum.'
            ),
            'limitation': (
                'No temporal snapshots are recorded. Full-search policy targets are forced-playout-pruned and are '
                'not raw visits. Heuristic savings are not replayed or causal measurements.'
            ),
        },
        'samples': sample_records,
        'combined': summarize_group(all_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='append', required=True, type=parse_sample)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    result = report(arguments.sample)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
