from __future__ import annotations

import hashlib
import random
from enum import Enum
from math import log10
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PlayerColor(str, Enum):
    WHITE = 'white'
    BLACK = 'black'


class GameOutcome(str, Enum):
    WIN = 'win'
    DRAW = 'draw'
    LOSS = 'loss'


class Opening(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    opening_id: str = Field(min_length=1)
    fen: str = Field(min_length=1)


class ScheduledGame(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schedule_index: int
    opening_id: str
    fen: str
    candidate_color: PlayerColor


class GameRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schedule_index: int
    opening_id: str
    starting_fen: str
    candidate_color: PlayerColor
    outcome: GameOutcome
    moves_uci: tuple[str, ...]


class MatchSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    game_count: int
    opening_pair_count: int
    wins: int
    draws: int
    losses: int
    score: float
    score_confidence_low: float
    score_confidence_high: float
    descriptive_logistic_elo_difference: float | None
    descriptive_logistic_elo_confidence_low: float | None
    descriptive_logistic_elo_confidence_high: float | None


class EngineSetting(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    value: str


class EngineCondition(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    identifier: str
    artifact_sha256: str
    search_limit_name: str
    search_limit_value: float
    threads: int
    settings: tuple[EngineSetting, ...]


class MatchConditions(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source_revision: str
    evaluation_source_revision: str
    opening_suite_path: str
    opening_suite_sha256: str
    candidate: EngineCondition
    opponent: EngineCondition
    maximum_game_plies: int | None
    bootstrap_seed: int
    bootstrap_samples: int


class MatchReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    conditions: MatchConditions
    summary: MatchSummary
    games: tuple[GameRecord, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_opening_suite(path: Path) -> tuple[Opening, ...]:
    openings: list[Opening] = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line or line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) != 2:
            raise ValueError(f'Invalid opening-suite row at line {line_number}: {line!r}')
        openings.append(Opening(opening_id=fields[0], fen=fields[1]))
    if not openings:
        raise ValueError(f'Opening suite contains no openings: {path}')
    if len({opening.opening_id for opening in openings}) != len(openings):
        raise ValueError(f'Opening IDs must be unique: {path}')
    return tuple(openings)


def build_paired_schedule(openings: tuple[Opening, ...]) -> tuple[ScheduledGame, ...]:
    scheduled_games: list[ScheduledGame] = []
    for opening in openings:
        for color in (PlayerColor.WHITE, PlayerColor.BLACK):
            scheduled_games.append(
                ScheduledGame(
                    schedule_index=len(scheduled_games),
                    opening_id=opening.opening_id,
                    fen=opening.fen,
                    candidate_color=color,
                )
            )
    return tuple(scheduled_games)


def _outcome_score(outcome: GameOutcome) -> float:
    match outcome:
        case GameOutcome.WIN:
            return 1.0
        case GameOutcome.DRAW:
            return 0.5
        case GameOutcome.LOSS:
            return 0.0


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError(f'Probability must be in [0, 1], received {probability}.')
    position = probability * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def _logistic_elo_difference(score: float) -> float | None:
    if not 0 < score < 1:
        return None
    return -400 * log10(1 / score - 1)


def _pair_scores(records: tuple[GameRecord, ...]) -> tuple[float, ...]:
    records_by_opening: dict[str, list[GameRecord]] = {}
    for record in records:
        records_by_opening.setdefault(record.opening_id, []).append(record)

    pair_scores: list[float] = []
    for opening_id, opening_records in records_by_opening.items():
        if len(opening_records) != 2:
            raise ValueError(f'Opening {opening_id!r} must have exactly two games.')
        colors = {record.candidate_color for record in opening_records}
        if colors != {PlayerColor.WHITE, PlayerColor.BLACK}:
            raise ValueError(f'Opening {opening_id!r} must swap candidate colors.')
        pair_scores.append(sum(_outcome_score(record.outcome) for record in opening_records) / 2)
    return tuple(pair_scores)


def summarize_match(
    records: tuple[GameRecord, ...],
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence_level: float = 0.95,
) -> MatchSummary:
    if not records:
        raise ValueError('Cannot summarize an empty match.')
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive.')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must be in (0, 1).')

    pair_scores = _pair_scores(records)
    random_number_generator = random.Random(bootstrap_seed)
    bootstrap_scores = sorted(
        sum(random_number_generator.choice(pair_scores) for _ in pair_scores) / len(pair_scores)
        for _ in range(bootstrap_samples)
    )
    tail_probability = (1 - confidence_level) / 2
    confidence_low = _quantile(bootstrap_scores, tail_probability)
    confidence_high = _quantile(bootstrap_scores, 1 - tail_probability)

    wins = sum(record.outcome == GameOutcome.WIN for record in records)
    draws = sum(record.outcome == GameOutcome.DRAW for record in records)
    losses = sum(record.outcome == GameOutcome.LOSS for record in records)
    score = sum(_outcome_score(record.outcome) for record in records) / len(records)

    return MatchSummary(
        game_count=len(records),
        opening_pair_count=len(pair_scores),
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        score_confidence_low=confidence_low,
        score_confidence_high=confidence_high,
        descriptive_logistic_elo_difference=_logistic_elo_difference(score),
        descriptive_logistic_elo_confidence_low=_logistic_elo_difference(confidence_low),
        descriptive_logistic_elo_confidence_high=_logistic_elo_difference(confidence_high),
    )


def write_match_report(path: Path, report: MatchReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    temporary_path.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')
    temporary_path.replace(path)
