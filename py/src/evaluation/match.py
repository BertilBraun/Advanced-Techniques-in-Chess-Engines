from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Generic, Literal, Protocol, TypeVar

from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    FixedCheckpointEvaluationDefinition,
    KataGoEvaluationDefinition,
    PolicyRandomOpponentEvaluationDefinition,
    PreviousCheckpointEvaluationDefinition,
    RandomOpponentEvaluationDefinition,
    StockfishEvaluationDefinition,
)
from src.evaluation.contracts import (
    CandidateOutcome,
    EvaluationGameResult,
    EvaluationTerminationReason,
    MatchEvaluationJob,
    MatchEvaluationResult,
    OpeningSuiteManifest,
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


def _definition_search(job: MatchEvaluationJob) -> EvaluationSearchConfiguration:
    match job.definition:
        case RandomOpponentEvaluationDefinition(search=search):
            return search
        case PreviousCheckpointEvaluationDefinition(search=search):
            return search
        case FixedCheckpointEvaluationDefinition(search=search):
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
    openings: OpeningSuiteManifest,
    random_seed: int,
) -> list[_ActiveMatch[PositionT]]:
    matches: list[_ActiveMatch[PositionT]] = []
    for pair_index, opening in enumerate(openings.openings):
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


def run_match(
    job: MatchEvaluationJob,
    game: GameImplementation[PositionT, NativeSearchT],
    openings: OpeningSuiteManifest,
    bootstrap_samples: int,
    external_engine: ExternalMatchEngine[PositionT] | None,
    device_type: Literal['cpu', 'cuda'],
) -> MatchEvaluationResult:
    started_at = time.monotonic()
    if isinstance(job.definition, PolicyRandomOpponentEvaluationDefinition):
        candidate_selector: MatchActionSelector[PositionT] = PolicyActionSelector(
            game.state,
            job.candidate.inference_model_path,
            job.device_id,
            device_type,
        )
        opponent_selector = None
    else:
        search_configuration = _definition_search(job)
        candidate_selector = SearchActionSelector(
            game.create_evaluation_search(job.device_id, job.candidate, search_configuration)
        )
        opponent_selector = (
            SearchActionSelector(
                game.create_evaluation_search(job.device_id, job.opponent.checkpoint, search_configuration)
            )
            if job.opponent.kind == 'checkpoint'
            else None
        )
    active = _build_matches(game.state, openings, job.random_seed)
    completed: list[EvaluationGameResult] = []
    random_generators = {
        match.game_index: random.Random(match.pair_seed + int(match.candidate_player)) for match in active
    }
    maximum_game_plies = _maximum_game_plies(job)
    while active:
        candidate_turns = tuple(
            match for match in active if game.state.current_player(match.position) is match.candidate_player
        )
        opponent_turns = tuple(
            match for match in active if game.state.current_player(match.position) is not match.candidate_player
        )
        actions: dict[int, int] = {}
        for match, action_id in zip(
            candidate_turns,
            candidate_selector.choose_actions(tuple(match.position for match in candidate_turns)),
            strict=True,
        ):
            actions[match.game_index] = action_id
        match job.opponent.kind:
            case 'checkpoint':
                assert opponent_selector is not None
                opponent_actions = opponent_selector.choose_actions(tuple(match.position for match in opponent_turns))
            case 'random':
                opponent_actions = tuple(
                    random_generators[match.game_index].choice(game.state.legal_action_ids(match.position))
                    for match in opponent_turns
                )
            case 'stockfish' | 'katago':
                if external_engine is None:
                    raise ValueError('External-engine match requires one job-local engine.')
                opponent_actions = external_engine.choose_actions(
                    tuple(match.position for match in opponent_turns),
                    tuple((*match.initial_action_ids, *match.played_action_ids) for match in opponent_turns),
                )
        for match, action_id in zip(opponent_turns, opponent_actions, strict=True):
            actions[match.game_index] = action_id
        remaining: list[_ActiveMatch[PositionT]] = []
        for active_match in active:
            action_id = actions[active_match.game_index]
            active_match.position = game.state.child_position(active_match.position, action_id)
            active_match.played_action_ids.append(action_id)
            terminal = game.state.natural_terminal_wdl(active_match.position)
            termination_reason = EvaluationTerminationReason.NATURAL
            total_plies = len(active_match.initial_action_ids) + len(active_match.played_action_ids)
            if terminal is None and total_plies >= maximum_game_plies:
                termination_reason = EvaluationTerminationReason.MAXIMUM_PLIES
                terminal = (
                    WdlTarget(win=0.0, draw=1.0, loss=0.0)
                    if game.state.name == 'chess'
                    else game.state.adjudicated_wdl(active_match.position, TerminationReason.MAXIMUM_PLIES)
                )
            if terminal is None:
                remaining.append(active_match)
                continue
            completed.append(
                EvaluationGameResult(
                    game_index=active_match.game_index,
                    pair_index=active_match.pair_index,
                    opening_id=active_match.opening_id,
                    candidate_player=('first' if active_match.candidate_player is Player.FIRST else 'second'),
                    pair_seed=active_match.pair_seed,
                    initial_action_ids=active_match.initial_action_ids,
                    played_action_ids=tuple(active_match.played_action_ids),
                    outcome=_outcome_for_candidate(game.state, active_match, terminal),
                    termination_reason=termination_reason,
                    plies=len(active_match.played_action_ids),
                    duration_seconds=time.monotonic() - active_match.started_at,
                )
            )
        active = remaining
    ordered = tuple(sorted(completed, key=lambda result: result.game_index))
    return MatchEvaluationResult(
        kind='match',
        job=job,
        games=ordered,
        aggregate=aggregate_match(ordered, job.random_seed, bootstrap_samples),
        duration_seconds=time.monotonic() - started_at,
    )
