from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
import time
from typing import Generic, Literal, Protocol, TypeVar

from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    KataGoEvaluationDefinition,
    PolicyRandomOpponentEvaluationDefinition,
    PreviousCheckpointEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
    ReferenceCheckpointEvaluationDefinition,
    StockfishEvaluationDefinition,
)
from src.evaluation.contracts import (
    CandidateOutcome,
    EvaluationGameResult,
    EvaluationTerminationReason,
    MatchEvaluationJob,
    MatchEvaluationResult,
    AnyOpeningSuiteManifest,
)
from src.evaluation.inference import PolicyActionSelector
from src.evaluation.statistics import aggregate_match
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.implementation import GameImplementation
from src.self_play.completed_game import TerminationReason
from src.self_play.native_search import NativeSelfPlaySearch


PositionT = TypeVar('PositionT')
NativeSearchT = TypeVar('NativeSearchT', bound=NativeSelfPlaySearch)


class ExternalMatchEngine(Protocol, Generic[PositionT]):
    def choose_actions(
        self,
        positions: tuple[PositionT, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]: ...

    def close(self) -> None: ...


class MatchActionSelector(Protocol, Generic[PositionT]):
    def choose_actions(self, positions: tuple[PositionT, ...]) -> tuple[int, ...]: ...


class SearchActionSelector(Generic[PositionT]):
    def __init__(self, search: NativeSelfPlaySearch) -> None:
        self.search = search

    def choose_actions(self, positions: tuple[PositionT, ...]) -> tuple[int, ...]:
        if not positions:
            return ()
        roots = tuple(self.search.new_root(position) for position in positions)
        batch = self.search.search([self.search.request(root, True) for root in roots])
        selected = []
        for result in batch.results:
            if not result.visits:
                raise RuntimeError('Evaluation search returned no visits for a nonterminal position.')
            selected.append(min(result.visits, key=lambda visit: (-visit.visit_count, visit.action_id)).action_id)
        return tuple(selected)


@dataclass
class _ActiveMatch(Generic[PositionT]):
    game_index: int
    pair_index: int
    opening_id: str
    candidate_player: Player
    pair_seed: int
    initial_action_ids: tuple[int, ...]
    position: PositionT
    played_action_ids: list[int]
    started_at: float
    random_generator: random.Random


@dataclass(frozen=True)
class _MatchSelectors(Generic[PositionT]):
    candidate: MatchActionSelector[PositionT]
    opponent: MatchActionSelector[PositionT] | None


def _definition_search(job: MatchEvaluationJob) -> EvaluationSearchConfiguration:
    match job.definition:
        case RandomOpponentEvaluationDefinition(search=search):
            return search
        case PreviousCheckpointEvaluationDefinition(search=search):
            return search
        case ReferenceCheckpointEvaluationDefinition(search=search):
            return search
        case StockfishEvaluationDefinition(search=search):
            return search
        case KataGoEvaluationDefinition(search=search):
            return search
        case _:
            raise ValueError('Match job must contain a match evaluation definition.')


def _maximum_game_plies(job: MatchEvaluationJob) -> int:
    return job.definition.maximum_game_plies


def _build_matches(
    state: GameStateContract[PositionT],
    openings: AnyOpeningSuiteManifest,
    opening_pair_count: int,
    random_seed: int,
) -> list[_ActiveMatch[PositionT]]:
    matches: list[_ActiveMatch[PositionT]] = []
    if opening_pair_count > len(openings.openings):
        raise ValueError('Opening suite does not contain enough pairs for the evaluation definition.')
    for pair_index, opening in enumerate(openings.openings[:opening_pair_count]):
        for candidate_player in (Player.FIRST, Player.SECOND):
            position = state.initial_position()
            for action_id in opening.action_ids:
                if action_id not in state.legal_action_ids(position):
                    raise ValueError(f'Opening {opening.opening_id!r} contains an illegal action.')
                position = state.child_position(position, action_id)
            game_index = len(matches)
            pair_seed = random_seed + pair_index
            matches.append(
                _ActiveMatch(
                    game_index=game_index,
                    pair_index=pair_index,
                    opening_id=opening.opening_id,
                    candidate_player=candidate_player,
                    pair_seed=pair_seed,
                    initial_action_ids=opening.action_ids,
                    position=position,
                    played_action_ids=[],
                    started_at=time.monotonic(),
                    random_generator=random.Random(pair_seed + int(candidate_player)),
                )
            )
    return matches


def _outcome_for_candidate(
    state: GameStateContract[PositionT],
    match: _ActiveMatch[PositionT],
    wdl: WdlTarget,
) -> CandidateOutcome:
    candidate_wdl = wdl if state.current_player(match.position) is match.candidate_player else wdl.reversed()
    if candidate_wdl.win > candidate_wdl.draw and candidate_wdl.win > candidate_wdl.loss:
        return CandidateOutcome.WIN
    if candidate_wdl.loss > candidate_wdl.draw and candidate_wdl.loss > candidate_wdl.win:
        return CandidateOutcome.LOSS
    return CandidateOutcome.DRAW


def _create_match_selectors(
    job: MatchEvaluationJob,
    game: GameImplementation[PositionT, NativeSearchT],
    device_type: Literal['cpu', 'cuda'],
) -> _MatchSelectors[PositionT]:
    if isinstance(job.definition, PolicyRandomOpponentEvaluationDefinition):
        return _MatchSelectors(
            candidate=PolicyActionSelector(
                game.state,
                job.candidate.inference_model_path,
                job.device_id,
                device_type,
            ),
            opponent=None,
        )
    search_configuration = _definition_search(job)
    candidate = SearchActionSelector(game.create_evaluation_search(job.device_id, job.candidate, search_configuration))
    opponent = (
        SearchActionSelector(
            game.create_evaluation_search(job.device_id, job.opponent.checkpoint, search_configuration)
        )
        if job.opponent.kind == 'checkpoint'
        else None
    )
    return _MatchSelectors(candidate=candidate, opponent=opponent)


def _partition_turns(
    state: GameStateContract[PositionT],
    active_matches: list[_ActiveMatch[PositionT]],
) -> tuple[tuple[_ActiveMatch[PositionT], ...], tuple[_ActiveMatch[PositionT], ...]]:
    candidate_turns = tuple(
        active_match
        for active_match in active_matches
        if state.current_player(active_match.position) is active_match.candidate_player
    )
    opponent_turns = tuple(
        active_match
        for active_match in active_matches
        if state.current_player(active_match.position) is not active_match.candidate_player
    )
    return candidate_turns, opponent_turns


def _choose_opponent_actions(
    job: MatchEvaluationJob,
    state: GameStateContract[PositionT],
    opponent_turns: tuple[_ActiveMatch[PositionT], ...],
    opponent_selector: MatchActionSelector[PositionT] | None,
    external_engine: ExternalMatchEngine[PositionT] | None,
) -> tuple[int, ...]:
    match job.opponent.kind:
        case 'checkpoint':
            assert opponent_selector is not None
            return opponent_selector.choose_actions(tuple(active_match.position for active_match in opponent_turns))
        case 'random':
            return tuple(
                active_match.random_generator.choice(state.legal_action_ids(active_match.position))
                for active_match in opponent_turns
            )
        case 'stockfish' | 'katago':
            if external_engine is None:
                raise ValueError('External-engine match requires one job-local engine.')
            return external_engine.choose_actions(
                tuple(active_match.position for active_match in opponent_turns),
                tuple(
                    (*active_match.initial_action_ids, *active_match.played_action_ids)
                    for active_match in opponent_turns
                ),
            )


def _choose_turn_actions(
    job: MatchEvaluationJob,
    state: GameStateContract[PositionT],
    active_matches: list[_ActiveMatch[PositionT]],
    selectors: _MatchSelectors[PositionT],
    external_engine: ExternalMatchEngine[PositionT] | None,
) -> dict[int, int]:
    candidate_turns, opponent_turns = _partition_turns(state, active_matches)
    candidate_actions = selectors.candidate.choose_actions(
        tuple(active_match.position for active_match in candidate_turns)
    )
    opponent_actions = _choose_opponent_actions(
        job,
        state,
        opponent_turns,
        selectors.opponent,
        external_engine,
    )
    return {
        active_match.game_index: action_id
        for turns, actions in ((candidate_turns, candidate_actions), (opponent_turns, opponent_actions))
        for active_match, action_id in zip(turns, actions, strict=True)
    }


def _terminal_result(
    state: GameStateContract[PositionT],
    active_match: _ActiveMatch[PositionT],
    maximum_game_plies: int,
) -> tuple[WdlTarget, EvaluationTerminationReason] | None:
    terminal = state.natural_terminal_wdl(active_match.position)
    if terminal is not None:
        return terminal, EvaluationTerminationReason.NATURAL
    total_plies = len(active_match.initial_action_ids) + len(active_match.played_action_ids)
    if total_plies < maximum_game_plies:
        return None
    adjudicated = (
        WdlTarget(win=0.0, draw=1.0, loss=0.0)
        if state.name == 'chess'
        else state.adjudicated_wdl(active_match.position, TerminationReason.MAXIMUM_PLIES)
    )
    return adjudicated, EvaluationTerminationReason.MAXIMUM_PLIES


def _complete_match(
    state: GameStateContract[PositionT],
    active_match: _ActiveMatch[PositionT],
    terminal: WdlTarget,
    termination_reason: EvaluationTerminationReason,
) -> EvaluationGameResult:
    return EvaluationGameResult(
        game_index=active_match.game_index,
        pair_index=active_match.pair_index,
        opening_id=active_match.opening_id,
        candidate_player=('first' if active_match.candidate_player is Player.FIRST else 'second'),
        pair_seed=active_match.pair_seed,
        initial_action_ids=active_match.initial_action_ids,
        played_action_ids=tuple(active_match.played_action_ids),
        outcome=_outcome_for_candidate(state, active_match, terminal),
        termination_reason=termination_reason,
        plies=len(active_match.played_action_ids),
        duration_seconds=time.monotonic() - active_match.started_at,
    )


def _advance_matches(
    state: GameStateContract[PositionT],
    active_matches: list[_ActiveMatch[PositionT]],
    selected_actions: Mapping[int, int],
    maximum_game_plies: int,
) -> tuple[list[_ActiveMatch[PositionT]], list[EvaluationGameResult]]:
    remaining: list[_ActiveMatch[PositionT]] = []
    completed: list[EvaluationGameResult] = []
    for active_match in active_matches:
        action_id = selected_actions[active_match.game_index]
        active_match.position = state.child_position(active_match.position, action_id)
        active_match.played_action_ids.append(action_id)
        terminal_result = _terminal_result(state, active_match, maximum_game_plies)
        if terminal_result is None:
            remaining.append(active_match)
            continue
        terminal, termination_reason = terminal_result
        completed.append(_complete_match(state, active_match, terminal, termination_reason))
    return remaining, completed


def run_match(
    job: MatchEvaluationJob,
    game: GameImplementation[PositionT, NativeSearchT],
    openings: AnyOpeningSuiteManifest,
    bootstrap_samples: int,
    external_engine: ExternalMatchEngine[PositionT] | None,
    device_type: Literal['cpu', 'cuda'],
) -> MatchEvaluationResult:
    started_at = time.monotonic()
    selectors = _create_match_selectors(job, game, device_type)
    active = _build_matches(game.state, openings, job.definition.opening_pair_count, job.random_seed)
    completed: list[EvaluationGameResult] = []
    maximum_game_plies = _maximum_game_plies(job)
    while active:
        selected_actions = _choose_turn_actions(job, game.state, active, selectors, external_engine)
        active, newly_completed = _advance_matches(game.state, active, selected_actions, maximum_game_plies)
        completed.extend(newly_completed)
    ordered = tuple(sorted(completed, key=lambda result: result.game_index))
    return MatchEvaluationResult(
        kind='match',
        job=job,
        games=ordered,
        aggregate=aggregate_match(ordered, job.random_seed, bootstrap_samples),
        duration_seconds=time.monotonic() - started_at,
    )
