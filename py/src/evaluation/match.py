from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    KataGoEvaluationDefinition,
    PolicyRandomOpponentEvaluationDefinition,
    PreviousCheckpointEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
    ReferenceCheckpointEvaluationDefinition,
    StockfishAdaptiveNodesEvaluationDefinition,
    StockfishEvaluationDefinition,
    StockfishFixedNodesEvaluationDefinition,
)
from src.evaluation.contracts import (
    AnyOpeningSuiteManifest,
    CandidateOutcome,
    EvaluationGameResult,
    EvaluationTerminationReason,
    MatchEvaluationJob,
    MatchEvaluationResult,
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
    def __init__(
        self,
        search: NativeSelfPlaySearch,
        searches_per_move: int,
        parallel_searches: int,
    ) -> None:
        self.search = search
        self.searches_per_move = searches_per_move
        self.parallel_searches = parallel_searches

    def choose_actions(self, positions: tuple[PositionT, ...]) -> tuple[int, ...]:
        if not positions:
            return ()
        roots = tuple(self.search.new_root(position) for position in positions)
        batch = self.search.search(
            [
                self.search.request(
                    root,
                    assigned_additional_visits=self.searches_per_move,
                    parallel_searches=self.parallel_searches,
                    add_root_noise=False,
                )
                for root in roots
            ]
        )
        selected = []
        for result in batch.results:
            if not result.search_visits:
                raise RuntimeError('Evaluation search returned no visits for a nonterminal position.')
            selected.append(
                min(result.search_visits, key=lambda visit: (-visit.visit_count, visit.action_id)).action_id
            )
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
        case StockfishFixedNodesEvaluationDefinition(search=search):
            return search
        case StockfishAdaptiveNodesEvaluationDefinition(search=search):
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


def _create_candidate_selector(
    job: MatchEvaluationJob,
    game: GameImplementation[PositionT, NativeSearchT],
    device_type: Literal['cpu', 'cuda'],
    candidate_selector: MatchActionSelector[PositionT] | None,
) -> MatchActionSelector[PositionT]:
    if isinstance(job.definition, PolicyRandomOpponentEvaluationDefinition):
        if candidate_selector is not None:
            raise ValueError('Policy-only evaluation does not accept a search selector override.')
        return PolicyActionSelector(
            game.state,
            job.candidate.inference_model_path,
            job.device_id,
            device_type,
        )
    if candidate_selector is not None:
        return candidate_selector
    search_configuration = _definition_search(job)
    return SearchActionSelector(
        game.create_evaluation_search(job.device_id, job.candidate, search_configuration),
        search_configuration.searches_per_move,
        search_configuration.parallel_searches,
    )


def _create_opponent_selector(
    job: MatchEvaluationJob,
    game: GameImplementation[PositionT, NativeSearchT],
) -> MatchActionSelector[PositionT] | None:
    if isinstance(job.definition, PolicyRandomOpponentEvaluationDefinition) or job.opponent.kind != 'checkpoint':
        return None
    search_configuration = _definition_search(job)
    return SearchActionSelector(
        game.create_evaluation_search(job.device_id, job.opponent.checkpoint, search_configuration),
        search_configuration.searches_per_move,
        search_configuration.parallel_searches,
    )


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
        case 'stockfish' | 'stockfish_fixed_nodes' | 'katago':
            if external_engine is None:
                raise ValueError('External-engine match requires one job-local engine.')
            return external_engine.choose_actions(
                tuple(active_match.position for active_match in opponent_turns),
                tuple(
                    (*active_match.initial_action_ids, *active_match.played_action_ids)
                    for active_match in opponent_turns
                ),
            )


def _selected_actions(
    turns: tuple[tuple[_ActiveMatch[PositionT], ...], ...],
    actions: tuple[tuple[int, ...], ...],
) -> dict[int, int]:
    return {
        active_match.game_index: action_id
        for group_turns, group_actions in zip(turns, actions, strict=True)
        for active_match, action_id in zip(group_turns, group_actions, strict=True)
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


@dataclass(frozen=True)
class ConcurrentMatchGroup(Generic[PositionT]):
    job: MatchEvaluationJob
    openings: AnyOpeningSuiteManifest
    external_engine: ExternalMatchEngine[PositionT] | None


@dataclass
class _GroupState(Generic[PositionT]):
    group: ConcurrentMatchGroup[PositionT]
    opponent_selector: MatchActionSelector[PositionT] | None
    maximum_game_plies: int
    active: list[_ActiveMatch[PositionT]]
    completed: list[EvaluationGameResult]
    finished_at: float


def _candidate_search(job: MatchEvaluationJob) -> EvaluationSearchConfiguration | None:
    if isinstance(job.definition, PolicyRandomOpponentEvaluationDefinition):
        return None
    return _definition_search(job)


def _validate_shared_candidate(groups: tuple[ConcurrentMatchGroup[PositionT], ...]) -> None:
    first = groups[0].job
    for group in groups[1:]:
        if group.job.candidate != first.candidate or group.job.device_id != first.device_id:
            raise ValueError('Concurrent matches must share one candidate checkpoint and device.')
        if _candidate_search(group.job) != _candidate_search(first):
            raise ValueError('Concurrent matches must share one candidate search configuration.')


def run_concurrent_matches(
    groups: tuple[ConcurrentMatchGroup[PositionT], ...],
    game: GameImplementation[PositionT, NativeSearchT],
    bootstrap_samples: int,
    device_type: Literal['cpu', 'cuda'],
    candidate_selector: MatchActionSelector[PositionT] | None = None,
) -> tuple[MatchEvaluationResult, ...]:
    """One shared candidate selector plays every group, so a single inference batch spans all of them."""
    if not groups:
        raise ValueError('Concurrent match execution requires at least one group.')
    _validate_shared_candidate(groups)
    started_at = time.monotonic()
    candidate = _create_candidate_selector(groups[0].job, game, device_type, candidate_selector)
    states = [
        _GroupState(
            group=group,
            opponent_selector=_create_opponent_selector(group.job, game),
            maximum_game_plies=_maximum_game_plies(group.job),
            active=_build_matches(
                game.state,
                group.openings,
                group.job.definition.opening_pair_count,
                group.job.random_seed,
            ),
            completed=[],
            finished_at=started_at,
        )
        for group in groups
    ]
    while any(state.active for state in states):
        partitioned = tuple(_partition_turns(game.state, state.active) for state in states)
        candidate_turns = tuple(turns for turns, _ in partitioned)
        candidate_actions = candidate.choose_actions(
            tuple(active_match.position for turns in candidate_turns for active_match in turns)
        )
        offset = 0
        sliced_candidate_actions: list[tuple[int, ...]] = []
        for turns in candidate_turns:
            sliced_candidate_actions.append(candidate_actions[offset : offset + len(turns)])
            offset += len(turns)
        if offset != len(candidate_actions):
            raise RuntimeError('Candidate selector returned an action count that does not match its positions.')
        for state, (turns, opponent_turns), actions in zip(
            states,
            partitioned,
            sliced_candidate_actions,
            strict=True,
        ):
            opponent_actions = _choose_opponent_actions(
                state.group.job,
                game.state,
                opponent_turns,
                state.opponent_selector,
                state.group.external_engine,
            )
            selected_actions = _selected_actions((turns, opponent_turns), (actions, opponent_actions))
            state.active, newly_completed = _advance_matches(
                game.state,
                state.active,
                selected_actions,
                state.maximum_game_plies,
            )
            state.completed.extend(newly_completed)
            if newly_completed:
                state.finished_at = time.monotonic()
    results: list[MatchEvaluationResult] = []
    for state in states:
        ordered = tuple(sorted(state.completed, key=lambda result: result.game_index))
        results.append(
            MatchEvaluationResult(
                kind='match',
                job=state.group.job,
                games=ordered,
                aggregate=aggregate_match(ordered, state.group.job.random_seed, bootstrap_samples),
                duration_seconds=state.finished_at - started_at,
            )
        )
    return tuple(results)


def run_match(
    job: MatchEvaluationJob,
    game: GameImplementation[PositionT, NativeSearchT],
    openings: AnyOpeningSuiteManifest,
    bootstrap_samples: int,
    external_engine: ExternalMatchEngine[PositionT] | None,
    device_type: Literal['cpu', 'cuda'],
    candidate_selector: MatchActionSelector[PositionT] | None = None,
) -> MatchEvaluationResult:
    group = ConcurrentMatchGroup(job=job, openings=openings, external_engine=external_engine)
    return run_concurrent_matches((group,), game, bootstrap_samples, device_type, candidate_selector)[0]
