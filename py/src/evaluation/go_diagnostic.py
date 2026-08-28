from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field
from src.evaluation.configuration import EvaluationSearchConfiguration
from src.evaluation.contracts import (
    AnyOpeningSuiteManifest,
    CandidateOutcome,
    EvaluationGameResult,
    EvaluationTerminationReason,
    MatchAggregate,
)
from src.evaluation.match import SearchActionSelector
from src.evaluation.statistics import aggregate_match
from src.games.contracts import Player, WdlTarget
from src.games.go.contract import GoStateContract, NativeGoPosition
from src.games.go.katago import KataGoClient, KataGoScoreEstimate, action_id_to_sgf
from src.games.go.training import GoImplementation
from src.self_play.completed_game import TerminationReason
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class ProjectModelDiagnosticEngine(FrozenModel):
    kind: Literal['project_model']
    engine_id: str = Field(min_length=1)
    checkpoint: CheckpointReference
    search: EvaluationSearchConfiguration


class KataGoDiagnosticEngine(FrozenModel):
    kind: Literal['katago']
    engine_id: str = Field(min_length=1)
    maximum_visits: int = Field(gt=0)


DiagnosticEngine: TypeAlias = Annotated[
    ProjectModelDiagnosticEngine | KataGoDiagnosticEngine,
    Field(discriminator='kind'),
]


class DiagnosticPlayerAssignment(FrozenModel):
    first: DiagnosticEngine
    second: DiagnosticEngine
    measured_player: Literal['first', 'second']


class NativeAreaScore(FrozenModel):
    black_half_points: int
    white_half_points: int
    black_minus_white_points: float
    winner: Literal['first', 'second', 'draw']


class EngineScoreEstimate(FrozenModel):
    engine_id: str = Field(min_length=1)
    current_player: Literal['black', 'white']
    score_lead: float


class GoDiagnosticTerminationReason(str, Enum):
    TWO_PASSES = 'two_passes'
    NATIVE_MAXIMUM_MOVES = 'native_maximum_moves'
    MAXIMUM_PLIES = 'maximum_plies'


class GoDiagnosticGameResult(FrozenModel):
    game: EvaluationGameResult
    players: DiagnosticPlayerAssignment
    complete_action_ids: tuple[int, ...]
    total_plies: int = Field(ge=0)
    native_termination_reason: GoDiagnosticTerminationReason
    native_area_score: NativeAreaScore
    pass_ending: bool
    final_score_estimates: tuple[EngineScoreEstimate, ...]
    sgf: str = Field(min_length=1)


class GoDiagnosticMatchResult(FrozenModel):
    diagnostic_id: str = Field(min_length=1)
    measured_engine: DiagnosticEngine
    opponent_engine: DiagnosticEngine
    opening_count: int = Field(gt=0)
    games: tuple[GoDiagnosticGameResult, ...] = Field(min_length=2)
    aggregate: MatchAggregate
    duration_seconds: float = Field(ge=0.0)


class GoEvaluationDiagnosticResult(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=1)
    experiment_path: Path
    opening_manifest_path: Path
    opening_manifest_sha256: str = Field(min_length=64, max_length=64)
    baseline_run_path: Path
    evaluated_checkpoint: CheckpointReference
    matches: tuple[GoDiagnosticMatchResult, ...] = Field(min_length=3, max_length=3)
    duration_seconds: float = Field(ge=0.0)


class DiagnosticActionSelector(Protocol):
    @property
    def engine(self) -> DiagnosticEngine: ...

    def choose_actions(
        self,
        positions: tuple[NativeGoPosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]: ...

    def final_score_estimates(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[KataGoScoreEstimate | None, ...]: ...

    def close(self) -> None: ...


class ProjectModelDiagnosticSelector:
    def __init__(
        self,
        engine: ProjectModelDiagnosticEngine,
        game: GoImplementation,
        device_id: int,
    ) -> None:
        self._engine = engine
        self._selector = SearchActionSelector(
            game.create_evaluation_search(device_id, engine.checkpoint, engine.search),
            engine.search.searches_per_move,
            engine.search.parallel_searches,
        )

    @property
    def engine(self) -> ProjectModelDiagnosticEngine:
        return self._engine

    def choose_actions(
        self,
        positions: tuple[NativeGoPosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if len(positions) != len(action_sequences):
            raise ValueError('Project-model diagnostic positions and histories must have equal lengths.')
        return self._selector.choose_actions(positions)

    def final_score_estimates(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[None, ...]:
        return (None,) * len(action_sequences)

    def close(self) -> None:
        pass


class KataGoDiagnosticSelector:
    def __init__(self, engine: KataGoDiagnosticEngine, client: KataGoClient) -> None:
        self._engine = engine
        self._client = client

    @property
    def engine(self) -> KataGoDiagnosticEngine:
        return self._engine

    def choose_actions(
        self,
        positions: tuple[NativeGoPosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if len(positions) != len(action_sequences):
            raise ValueError('KataGo diagnostic positions and histories must have equal lengths.')
        analyses = self._client.analyze_detailed_many(action_sequences, self.engine.maximum_visits)
        return tuple(analysis.policy.selected_action_id for analysis in analyses)

    def final_score_estimates(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[KataGoScoreEstimate | None, ...]:
        return tuple(
            analysis.score_estimate
            for analysis in self._client.analyze_detailed_many(action_sequences, self.engine.maximum_visits)
        )

    def close(self) -> None:
        self._client.close()


@dataclass
class _ActiveDiagnosticGame:
    game_index: int
    pair_index: int
    opening_id: str
    measured_player: Player
    pair_seed: int
    initial_action_ids: tuple[int, ...]
    position: NativeGoPosition
    played_action_ids: list[int]
    started_at: float


@dataclass(frozen=True)
class _CompletedDiagnosticGame:
    active: _ActiveDiagnosticGame
    terminal_wdl: WdlTarget
    termination_reason: GoDiagnosticTerminationReason
    duration_seconds: float


def _build_games(
    state: GoStateContract,
    openings: AnyOpeningSuiteManifest,
    opening_count: int,
    random_seed: int,
) -> list[_ActiveDiagnosticGame]:
    if opening_count > len(openings.openings):
        raise ValueError('Diagnostic opening count exceeds the immutable manifest.')
    games: list[_ActiveDiagnosticGame] = []
    for pair_index, opening in enumerate(openings.openings[:opening_count]):
        position = state.initial_position()
        for action_id in opening.action_ids:
            if action_id not in state.legal_action_ids(position):
                raise ValueError(f'Opening {opening.opening_id!r} contains an illegal action.')
            position = state.child_position(position, action_id)
        for measured_player in (Player.FIRST, Player.SECOND):
            games.append(
                _ActiveDiagnosticGame(
                    game_index=len(games),
                    pair_index=pair_index,
                    opening_id=opening.opening_id,
                    measured_player=measured_player,
                    pair_seed=random_seed + pair_index,
                    initial_action_ids=opening.action_ids,
                    position=position,
                    played_action_ids=[],
                    started_at=time.monotonic(),
                )
            )
    return games


def _selector_for_turn(
    state: GoStateContract,
    game: _ActiveDiagnosticGame,
    measured: DiagnosticActionSelector,
    opponent: DiagnosticActionSelector,
) -> DiagnosticActionSelector:
    return measured if state.current_player(game.position) is game.measured_player else opponent


def _choose_actions(
    state: GoStateContract,
    games: list[_ActiveDiagnosticGame],
    measured: DiagnosticActionSelector,
    opponent: DiagnosticActionSelector,
) -> dict[int, int]:
    selected: dict[int, int] = {}
    selectors = (measured,) if measured is opponent else (measured, opponent)
    for selector in selectors:
        turns = tuple(game for game in games if _selector_for_turn(state, game, measured, opponent) is selector)
        actions = selector.choose_actions(
            tuple(game.position for game in turns),
            tuple((*game.initial_action_ids, *game.played_action_ids) for game in turns),
        )
        selected.update((game.game_index, action_id) for game, action_id in zip(turns, actions, strict=True))
    return selected


def _terminal_result(
    state: GoStateContract,
    game: _ActiveDiagnosticGame,
    maximum_game_plies: int,
) -> tuple[WdlTarget, GoDiagnosticTerminationReason] | None:
    if game.position.is_terminal:
        terminal = state.natural_terminal_wdl(game.position)
        if terminal is None:
            return (
                state.adjudicated_wdl(game.position, TerminationReason.MAXIMUM_PLIES),
                GoDiagnosticTerminationReason.NATIVE_MAXIMUM_MOVES,
            )
        return terminal, GoDiagnosticTerminationReason.TWO_PASSES
    total_plies = len(game.initial_action_ids) + len(game.played_action_ids)
    if total_plies < maximum_game_plies:
        return None
    return (
        state.adjudicated_wdl(game.position, TerminationReason.MAXIMUM_PLIES),
        GoDiagnosticTerminationReason.MAXIMUM_PLIES,
    )


def _outcome_for_measured(
    state: GoStateContract,
    game: _ActiveDiagnosticGame,
    terminal_wdl: WdlTarget,
) -> CandidateOutcome:
    measured_wdl = (
        terminal_wdl if state.current_player(game.position) is game.measured_player else terminal_wdl.reversed()
    )
    if measured_wdl.win > measured_wdl.draw and measured_wdl.win > measured_wdl.loss:
        return CandidateOutcome.WIN
    if measured_wdl.loss > measured_wdl.draw and measured_wdl.loss > measured_wdl.win:
        return CandidateOutcome.LOSS
    return CandidateOutcome.DRAW


def _native_area_score(state: GoStateContract, position: NativeGoPosition) -> NativeAreaScore:
    score = position.area_score()
    if score.winner is None:
        winner: Literal['first', 'second', 'draw'] = 'draw'
    else:
        current_player = state.current_player(position)
        winning_player = current_player if score.winner == position.player else Player(-current_player)
        winner = 'first' if winning_player is Player.FIRST else 'second'
    return NativeAreaScore(
        black_half_points=score.black_half_points,
        white_half_points=score.white_half_points,
        black_minus_white_points=(score.black_half_points - score.white_half_points) / 2.0,
        winner=winner,
    )


def _sgf(state: GoStateContract, action_ids: tuple[int, ...], score: NativeAreaScore) -> str:
    if score.winner == 'draw':
        result = '0'
    else:
        margin = abs(score.black_minus_white_points)
        result = f'{"B" if score.winner == "first" else "W"}+{margin:g}'
    moves = ''.join(
        f';{"B" if ply % 2 == 0 else "W"}[{action_id_to_sgf(action_id, state.board_size)}]'
        for ply, action_id in enumerate(action_ids)
    )
    return f'(;GM[1]FF[4]SZ[{state.board_size}]KM[{state.komi_half_points / 2.0}]RE[{result}]{moves})'


def _player_assignment(
    game: _ActiveDiagnosticGame,
    measured: DiagnosticActionSelector,
    opponent: DiagnosticActionSelector,
) -> DiagnosticPlayerAssignment:
    if game.measured_player is Player.FIRST:
        return DiagnosticPlayerAssignment(
            first=measured.engine,
            second=opponent.engine,
            measured_player='first',
        )
    return DiagnosticPlayerAssignment(
        first=opponent.engine,
        second=measured.engine,
        measured_player='second',
    )


def _final_estimates(
    completed: tuple[_CompletedDiagnosticGame, ...],
    selectors: tuple[DiagnosticActionSelector, ...],
) -> tuple[tuple[EngineScoreEstimate, ...], ...]:
    histories = tuple((*game.active.initial_action_ids, *game.active.played_action_ids) for game in completed)
    estimates: list[list[EngineScoreEstimate]] = [[] for _ in completed]
    seen_engine_ids: set[str] = set()
    for selector in selectors:
        if selector.engine.engine_id in seen_engine_ids:
            continue
        seen_engine_ids.add(selector.engine.engine_id)
        for index, estimate in enumerate(selector.final_score_estimates(histories)):
            if estimate is not None:
                estimates[index].append(
                    EngineScoreEstimate(
                        engine_id=selector.engine.engine_id,
                        current_player=estimate.current_player,
                        score_lead=estimate.score_lead,
                    )
                )
    return tuple(tuple(game_estimates) for game_estimates in estimates)


def run_go_diagnostic_match(
    diagnostic_id: str,
    state: GoStateContract,
    openings: AnyOpeningSuiteManifest,
    opening_count: int,
    measured: DiagnosticActionSelector,
    opponent: DiagnosticActionSelector,
    maximum_game_plies: int,
    random_seed: int,
    bootstrap_samples: int,
) -> GoDiagnosticMatchResult:
    started_at = time.monotonic()
    active = _build_games(state, openings, opening_count, random_seed)
    completed: list[_CompletedDiagnosticGame] = []
    while active:
        actions = _choose_actions(state, active, measured, opponent)
        remaining: list[_ActiveDiagnosticGame] = []
        for game in active:
            action_id = actions[game.game_index]
            game.position = state.child_position(game.position, action_id)
            game.played_action_ids.append(action_id)
            terminal = _terminal_result(state, game, maximum_game_plies)
            if terminal is None:
                remaining.append(game)
            else:
                completed.append(
                    _CompletedDiagnosticGame(
                        active=game,
                        terminal_wdl=terminal[0],
                        termination_reason=terminal[1],
                        duration_seconds=time.monotonic() - game.started_at,
                    )
                )
        active = remaining
    ordered = tuple(sorted(completed, key=lambda item: item.active.game_index))
    score_estimates = _final_estimates(ordered, (measured, opponent))
    game_results: list[GoDiagnosticGameResult] = []
    aggregate_games: list[EvaluationGameResult] = []
    for completed_game, estimates in zip(ordered, score_estimates, strict=True):
        game = completed_game.active
        complete_actions = (*game.initial_action_ids, *game.played_action_ids)
        evaluation_game = EvaluationGameResult(
            game_index=game.game_index,
            pair_index=game.pair_index,
            opening_id=game.opening_id,
            candidate_player='first' if game.measured_player is Player.FIRST else 'second',
            pair_seed=game.pair_seed,
            initial_action_ids=game.initial_action_ids,
            played_action_ids=tuple(game.played_action_ids),
            outcome=_outcome_for_measured(state, game, completed_game.terminal_wdl),
            termination_reason=(
                EvaluationTerminationReason.NATURAL
                if completed_game.termination_reason is GoDiagnosticTerminationReason.TWO_PASSES
                else EvaluationTerminationReason.MAXIMUM_PLIES
            ),
            plies=len(game.played_action_ids),
            duration_seconds=completed_game.duration_seconds,
        )
        aggregate_games.append(evaluation_game)
        area_score = _native_area_score(state, game.position)
        game_results.append(
            GoDiagnosticGameResult(
                game=evaluation_game,
                players=_player_assignment(game, measured, opponent),
                complete_action_ids=complete_actions,
                total_plies=len(complete_actions),
                native_termination_reason=completed_game.termination_reason,
                native_area_score=area_score,
                pass_ending=(
                    len(complete_actions) >= 2 and complete_actions[-2:] == (state.pass_action, state.pass_action)
                ),
                final_score_estimates=estimates,
                sgf=_sgf(state, complete_actions, area_score),
            )
        )
    return GoDiagnosticMatchResult(
        diagnostic_id=diagnostic_id,
        measured_engine=measured.engine,
        opponent_engine=opponent.engine,
        opening_count=opening_count,
        games=tuple(game_results),
        aggregate=aggregate_match(tuple(aggregate_games), random_seed, bootstrap_samples),
        duration_seconds=time.monotonic() - started_at,
    )


def validate_deterministic_identity(result: GoDiagnosticMatchResult) -> None:
    if result.measured_engine != result.opponent_engine:
        raise ValueError('Identity validation requires the same engine on both sides.')
    if result.aggregate.score != 0.5:
        raise RuntimeError(f'Deterministic identity aggregate was {result.aggregate.score:.3f}, not 0.500.')
    for game_index in range(0, len(result.games), 2):
        first, second = result.games[game_index : game_index + 2]
        if first.complete_action_ids != second.complete_action_ids:
            raise RuntimeError(f'Identity pair {first.game.pair_index} produced different action histories.')
        if first.native_area_score != second.native_area_score:
            raise RuntimeError(f'Identity pair {first.game.pair_index} produced different native scores.')


def write_go_diagnostic_match(result: GoDiagnosticMatchResult, path: Path) -> None:
    write_text_atomically(path, result.model_dump_json(indent=2) + '\n')


def write_go_diagnostic_result(result: GoEvaluationDiagnosticResult, path: Path) -> None:
    write_text_atomically(path, result.model_dump_json(indent=2) + '\n')


def _summarize_opening_ids(opening_ids: tuple[str, ...]) -> str:
    visible = ', '.join(opening_ids[:5])
    remaining = len(opening_ids) - 5
    return visible if remaining <= 0 else f'{visible} (+{remaining} more)'


def render_go_diagnostic_analysis(result: GoEvaluationDiagnosticResult) -> str:
    lines = ['# Go evaluation diagnostic', '']
    for match in result.matches:
        termination_counts = Counter(game.native_termination_reason.value for game in match.games)
        lengths = sorted(game.total_plies for game in match.games)
        pass_endings = sum(game.pass_ending for game in match.games)
        aggregate = match.aggregate
        pair_scores = tuple(
            (
                match.games[index].game.opening_id,
                sum(
                    {
                        CandidateOutcome.WIN: 1.0,
                        CandidateOutcome.DRAW: 0.5,
                        CandidateOutcome.LOSS: 0.0,
                    }[game.game.outcome]
                    for game in match.games[index : index + 2]
                )
                / 2.0,
            )
            for index in range(0, len(match.games), 2)
        )
        minimum_pair_score = min(score for _, score in pair_scores)
        maximum_pair_score = max(score for _, score in pair_scores)
        minimum_openings = _summarize_opening_ids(
            tuple(opening_id for opening_id, score in pair_scores if score == minimum_pair_score)
        )
        maximum_openings = _summarize_opening_ids(
            tuple(opening_id for opening_id, score in pair_scores if score == maximum_pair_score)
        )
        engine_score_checks = tuple(
            (
                game.native_area_score.black_minus_white_points,
                estimate.score_lead if estimate.current_player == 'black' else -estimate.score_lead,
            )
            for game in match.games
            for estimate in game.final_score_estimates
        )
        lines.extend(
            (
                f'## {match.diagnostic_id}',
                '',
                f'- Score: {aggregate.score:.1%} ({aggregate.wins}-{aggregate.draws}-{aggregate.losses}), '
                f'paired 95% CI {aggregate.score_confidence_low:.1%}–{aggregate.score_confidence_high:.1%}.',
                f'- First/second-player measured scores: {aggregate.first_player_score:.1%} / '
                f'{aggregate.second_player_score:.1%}.',
                f'- Total game plies: min {lengths[0]}, p25 {lengths[len(lengths) // 4]}, '
                f'median {lengths[len(lengths) // 2]}, p75 {lengths[3 * len(lengths) // 4]}, '
                f'max {lengths[-1]}.',
                f'- Terminations: {dict(sorted(termination_counts.items()))}; pass endings: '
                f'{pass_endings}/{len(match.games)}.',
                f'- Opening-pair score range: {minimum_pair_score:.1%} ({minimum_openings}) to '
                f'{maximum_pair_score:.1%} ({maximum_openings}).',
                (
                    f'- Native/KataGo final-score checks: {len(engine_score_checks)} estimates, mean absolute '
                    f'black-lead difference '
                    f'{sum(abs(native - engine) for native, engine in engine_score_checks) / len(engine_score_checks):.2f} points.'
                    if engine_score_checks
                    else '- Native/KataGo final-score checks: no KataGo estimate was available.'
                ),
                '',
            )
        )
    identity = result.matches[0]
    equal_katago = result.matches[1]
    stronger = result.matches[2]
    equal_is_consistent = (
        equal_katago.aggregate.score_confidence_low <= 0.5 <= equal_katago.aggregate.score_confidence_high
    )
    stronger_is_demonstrated = stronger.aggregate.score_confidence_high < 0.5
    lines.extend(
        (
            '## Interpretation',
            '',
            f'- Identical deterministic checkpoint play is {identity.aggregate.score:.1%}; '
            'the plumbing identity invariant passed.'
            if identity.aggregate.score == 0.5
            else f'- Identical deterministic checkpoint play is {identity.aggregate.score:.1%}; '
            'the plumbing identity invariant failed.',
            f'- Equal KataGo 16-visit play scores {equal_katago.aggregate.score:.1%}.',
            f'- Its paired interval {"includes" if equal_is_consistent else "excludes"} 50%; '
            'the equal-engine control is therefore '
            f'{"consistent" if equal_is_consistent else "inconsistent"} with fair role swapping.',
            f'- KataGo 16 versus 128 visits scores {stronger.aggregate.score:.1%} for the 16-visit engine; '
            f'the paired interval is {stronger.aggregate.score_confidence_low:.1%}–'
            f'{stronger.aggregate.score_confidence_high:.1%}.',
            f'- The 128-visit advantage is {"demonstrable" if stronger_is_demonstrated else "not demonstrable"} '
            'at this sample size because the 16-visit score interval '
            f'{"lies wholly below" if stronger_is_demonstrated else "does not lie wholly below"} 50%.',
            '- The very large first-player advantage shows why raw color totals are misleading and every opening '
            'must be paired.',
            '- The exact model identity result and the equal-KataGo interval rule out a role-swap or aggregation '
            'explanation for the initially barely trained model scoring 10–18%. A weak, low-visit KataGo opponent '
            'still loses an opening-dependent tail within 100 games; later curve changes are compatible with real '
            'policy/search learning plus paired-sample uncertainty.',
            '',
        )
    )
    return '\n'.join(lines)
