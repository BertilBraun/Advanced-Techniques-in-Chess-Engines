from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast
from uuid import UUID

import numpy as np
import pytest

pytest.importorskip('AlphaZeroCpp')
from decimal import Decimal

from AlphaZeroCpp import GameSearchVisit
from AlphaZeroCpp import SearchStopReason as NativeSearchStopReason
from src.games.contracts import TerminalOracle, WdlTarget
from src.games.implementation import GameImplementation
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.policy import SearchStopPolicy, flat_stop_policy
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
)
from src.self_play.parameters import (
    RandomOpeningStartParameters,
    ResolvedSelfPlayParameters,
    RestartStateStartParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.self_play.resignation import (
    CalibratedResignationConfiguration,
    PublishedResignationPolicy,
)
from src.self_play.restart_archive import RestartStateArchive, worker_restart_archive_path
from src.self_play.worker import SelfPlayWorker
from src.training.checkpoint import CheckpointReference
from test_helpers.checkpoints import checkpoint_reference


@dataclass(frozen=True)
class FakePosition:
    ply: int


@dataclass
class FakeRoot:
    position: FakePosition
    is_terminal: bool = False
    reset_count: int = 0
    visits: int = 0

    def play(self, action_id: int) -> None:
        assert action_id in (0, 1, 2)
        self.position = FakePosition(self.position.ply + 1)

    def reset(self) -> None:
        self.reset_count += 1

    def discount(self, retained_fraction: float) -> None:
        assert retained_fraction == 0.5


@dataclass(frozen=True)
class FakeRequest:
    root: FakeRoot
    assigned_additional_visits: int | None = None
    policy_checkpoint_visits: list[int] | None = None
    parallel_searches: int | None = None
    root_ply: int = 0


@dataclass(frozen=True)
class FakeResult:
    root_value: float
    highest_visited_child_action_id: int
    highest_visited_child_visit_count: int
    highest_visited_child_q: float
    search_visits: list[GameSearchVisit]
    policy_target_visits: list[GameSearchVisit]
    network_root_value: float
    policy_correction: float
    value_correction: float
    stop_checkpoint_index: int
    parallel_searches: int
    starting_visits: int
    final_visits: int
    stop_reason: NativeSearchStopReason
    root: FakeRoot
    stop_probabilities: tuple[float, ...] = ()
    guard_movements: tuple[float, ...] = ()
    stop_verdicts: tuple[int, ...] = ()
    stop_features: tuple[tuple[float, ...], ...] = ()
    checkpoints: tuple[object, ...] = ()

    @property
    def search_visit_columns(self) -> tuple[list[int], list[int]]:
        return _columns(self.search_visits)

    @property
    def policy_target_columns(self) -> tuple[list[int], list[int]]:
        return _columns(self.policy_target_visits)


def _columns(visits: list[GameSearchVisit]) -> tuple[list[int], list[int]]:
    return ([visit.action_id for visit in visits], [visit.visit_count for visit in visits])


@dataclass(frozen=True)
class FakeBatch:
    results: list[FakeResult]
    simulations_completed: int


@dataclass(frozen=True)
class FakeInferenceStatistics:
    averageNumberOfPositionsInInferenceCall: float = 2.0
    workerUtilization: float = 0.5


class FakeSearch:
    def __init__(self) -> None:
        self.generations: list[int] = []
        self.request_batches: list[tuple[FakeRequest, ...]] = []
        self.capacity_changed = False
        self.search_visits = [GameSearchVisit(0, 3)]
        self.policy_target_visits = [GameSearchVisit(0, 3)]
        self.root_value = 0.25
        self.highest_visited_child_q = 0.2

    def new_root(self, position: FakePosition, maximum_capacity: int = 0) -> FakeRoot:
        del maximum_capacity
        return FakeRoot(position)

    def request(
        self,
        root: FakeRoot,
        assigned_additional_visits: int | None = None,
        policy_checkpoint_visits: list[int] | None = None,
        parallel_searches: int | None = None,
        root_ply: int = 0,
        checkpoint_detail: object = None,
        add_root_noise: bool = True,
        audit: bool = False,
    ) -> FakeRequest:
        del checkpoint_detail, add_root_noise, audit
        return FakeRequest(root, assigned_additional_visits, policy_checkpoint_visits, parallel_searches, root_ply)

    def search(self, requests: list[FakeRequest], collect_statistics: bool = False) -> FakeBatch:
        assert not collect_statistics
        self.request_batches.append(tuple(requests))
        return FakeBatch(
            [
                FakeResult(
                    root_value=self.root_value,
                    highest_visited_child_action_id=self.search_visits[0].action_id,
                    highest_visited_child_visit_count=self.search_visits[0].visit_count,
                    highest_visited_child_q=self.highest_visited_child_q,
                    search_visits=self.search_visits,
                    policy_target_visits=self.policy_target_visits,
                    network_root_value=self.root_value,
                    policy_correction=0.0,
                    value_correction=0.0,
                    stop_checkpoint_index=-1,
                    parallel_searches=1,
                    starting_visits=0,
                    final_visits=3,
                    stop_reason=NativeSearchStopReason.FIXED_LIMIT,
                    root=request.root,
                )
                for request in requests
            ],
            simulations_completed=len(requests) * 3,
        )

    def refresh_model(self, model_generation: int, model_path: str) -> None:
        assert Path(model_path).suffixes == ['.jit', '.pt']
        self.generations.append(model_generation)

    def update_search_schedule(self, search_parameters: ResolvedSelfPlayParameters) -> bool:
        del search_parameters
        return self.capacity_changed

    def inference_statistics(self) -> FakeInferenceStatistics:
        return FakeInferenceStatistics()


class FakeState:
    def initial_position(self) -> FakePosition:
        return FakePosition(0)

    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        del position
        return (0, 1, 2)

    def child_position(self, position: FakePosition, action_id: int) -> FakePosition:
        assert action_id in (0, 1, 2)
        return FakePosition(position.ply + 1)

    def natural_terminal_wdl(self, position: FakePosition) -> WdlTarget | None:
        del position
        return None

    def adjudicated_wdl(self, position: FakePosition, reason: TerminationReason) -> WdlTarget:
        del position, reason
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)


_STOPPING_CONFIGURATION = SearchStoppingConfiguration(
    audit_sample_fraction=Decimal('0.01'),
    paired_audit_fraction=Decimal('0.1'),
    noise_floor_multiple=1.0,
    anchor_fraction=Decimal('0.05'),
    anchor_visit_multiple=4.0,
    checkpoint_multiples=(1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5),
    cap_multiple=2.0,
    eps_pi_minimum=0.02,
    eps_pi_maximum=0.3,
    eps_v=0.3,
    movement_guard_epsilon=0.05,
    excess_cost_ceiling=0.25,
    catastrophic_excess_multiple=5.0,
    catastrophic_stop_ceiling=0.01,
    minimum_evidence_trigger_count=100,
    confidence_level=0.95,
    first_production_generation=10,
    maximum_realized_mean_spend=1.3,
    window_generations=10,
    maximum_unstarted_generation_lag=2,
)


@dataclass(frozen=True)
class FakeLifecycle:
    search_stopping: SearchStoppingConfiguration = _STOPPING_CONFIGURATION


@dataclass(frozen=True)
class FakeTraining:
    random_seed: int = 5
    lifecycle: FakeLifecycle = FakeLifecycle()


class FakeGame:
    training = FakeTraining()
    state = FakeState()

    def __init__(
        self,
        maximum_random_opening_plies: int = 0,
        restart_parameters: RestartStateStartParameters | None = None,
        resignation_configuration: CalibratedResignationConfiguration | None = None,
        maximum_game_plies: int | None = None,
        bootstrap_cut_game_value: bool = False,
        terminal_oracle: TerminalOracle[FakePosition] | None = None,
    ) -> None:
        self.search = FakeSearch()
        self.maximum_random_opening_plies = maximum_random_opening_plies
        self.restart_parameters = restart_parameters
        self.resignation_configuration = resignation_configuration
        self.maximum_game_plies = maximum_game_plies
        self.bootstrap_cut_game_value = bootstrap_cut_game_value
        self.terminal_oracle = terminal_oracle

    def self_play_parameters_at(
        self,
        model_generation: int,
        search_stop_policy: SearchStopPolicy,
    ) -> ResolvedSelfPlayParameters:
        del model_generation
        start_position = self.restart_parameters
        if start_position is None:
            start_position = RandomOpeningStartParameters(
                kind='random_opening', maximum_plies=self.maximum_random_opening_plies
            )
        return ResolvedSelfPlayParameters(
            start_position=start_position,
            baseline_visits=3,
            search_stop_policy=search_stop_policy,
            forced_playout_coefficient=0.0,
            exploration_constant=1.0,
            first_play_urgency=ZeroFirstPlayUrgencyParameters(),
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            retained_root_visit_fraction=0.5,
            starting_temperature=1.0,
            final_temperature=1.0,
            greedy_after_ply=1,
            maximum_game_plies=self.maximum_game_plies,
            primary_sample_weight=1.0,
            value_discount_per_ply=1.0,
            bootstrap_cut_game_value=self.bootstrap_cut_game_value,
        )

    def close(self) -> None:
        if self.terminal_oracle is not None:
            self.terminal_oracle.close()

    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> FakeSearch:
        del device_id, checkpoint, parameters
        return self.search

    def native_search_parameters(self, parameters: ResolvedSelfPlayParameters) -> ResolvedSelfPlayParameters:
        return parameters


class RecordingTerminalOracle(TerminalOracle[FakePosition]):
    def __init__(self, result: WdlTarget | None) -> None:
        self.result = result
        self.probed_positions: list[FakePosition] = []

    def probe_wdl(self, position: FakePosition) -> WdlTarget | None:
        self.probed_positions.append(position)
        return self.result


def checkpoint(path: Path, generation: int) -> CheckpointReference:
    return checkpoint_reference(path, generation, write_inference_model=True)


def resignation_configuration(continuation_probability: float) -> CalibratedResignationConfiguration:
    return CalibratedResignationConfiguration(
        first_production_generation=50,
        false_nonloss_rate_ceiling=0.03,
        continuation_game_probability=continuation_probability,
        triggered_game_window=2000,
        candidate_threshold_minimum=-0.99,
        candidate_threshold_maximum=-0.70,
        candidate_threshold_step=0.01,
        minimum_evidence_trigger_count=100,
        confidence_level=0.95,
        maximum_relaxation_per_generation=0.01,
    )


def test_worker_owns_shared_search_move_selection_and_generation_transition(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=3,
        worker_id=2,
        device_id=0,
        inbox_path=tmp_path,
    )

    with pytest.raises(RuntimeError, match='model must be loaded'):
        worker.run_batch()

    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())
    worker.run_batch()
    worker.refresh_published_model(checkpoint(tmp_path, 1), flat_stop_policy())
    statistics = worker.snapshot_statistics()

    assert [game.root.position.ply for game in worker.active_games] == [1, 1, 1]
    assert [game.root.reset_count for game in worker.active_games] == [1, 1, 1]
    assert all(game.action_ids == [0] for game in worker.active_games)
    assert all(
        game.observations[0].policy_target_visits.native_visits() == (GameSearchVisit(action_id=0, visit_count=3),)
        for game in worker.active_games
    )
    assert game.search.generations == [1]
    assert statistics.model_generation == 1
    assert statistics.completed_searches == 9


def test_terminal_oracle_is_probed_only_at_maximum_ply(tmp_path: Path) -> None:
    oracle_result = WdlTarget(win=1.0, draw=0.0, loss=0.0)
    oracle = RecordingTerminalOracle(oracle_result)
    game = FakeGame(maximum_game_plies=2, terminal_oracle=oracle)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    worker.run_batch()
    assert oracle.probed_positions == []
    worker.run_batch()

    completed = CompletedSelfPlayGame.model_validate_json(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert oracle.probed_positions == [FakePosition(2)]
    assert completed.final_wdl == oracle_result
    assert completed.termination_reason is TerminationReason.MAXIMUM_PLIES


def test_uncovered_maximum_ply_position_uses_game_adjudication(tmp_path: Path) -> None:
    oracle = RecordingTerminalOracle(None)
    game = FakeGame(maximum_game_plies=1, terminal_oracle=oracle)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    worker.run_batch()

    completed = CompletedSelfPlayGame.model_validate_json(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert oracle.probed_positions == [FakePosition(1)]
    assert completed.final_wdl == WdlTarget(win=0.0, draw=1.0, loss=0.0)


def test_worker_replaces_roots_when_search_arena_capacity_changes(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=2,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())
    worker.run_batch()
    original_roots = tuple(active_game.root for active_game in worker.active_games)

    game.search.capacity_changed = True
    worker.refresh_published_model(checkpoint(tmp_path, 1), flat_stop_policy())

    assert all(active_game.root is not original for active_game, original in zip(worker.active_games, original_roots))
    assert [active_game.root.position.ply for active_game in worker.active_games] == [1, 1]
    assert [active_game.action_ids for active_game in worker.active_games] == [[0], [0]]
    assert [active_game.root.reset_count for active_game in worker.active_games] == [0, 0]


def test_worker_uses_learned_budget_search_for_every_position(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    worker.run_batch()
    worker.run_batch()

    assert all(batch[0].assigned_additional_visits is None for batch in game.search.request_batches)
    assert [batch[0].root_ply for batch in game.search.request_batches] == [0, 1]
    assert [
        observation.final_visits - observation.starting_visits for observation in worker.active_games[0].observations
    ] == [3, 3]
    assert [observation.stop_checkpoint_index for observation in worker.active_games[0].observations] == [-1, -1]


def test_worker_selects_from_actual_visits_and_records_pruned_target_visits(tmp_path: Path) -> None:
    game = FakeGame()
    game.search.search_visits = [GameSearchVisit(1, 20)]
    game.search.policy_target_visits = [GameSearchVisit(0, 7)]
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    worker.run_batch()

    assert worker.active_games[0].action_ids == [1]
    assert worker.active_games[0].observations[0].policy_target_visits.native_visits() == (
        GameSearchVisit(action_id=0, visit_count=7),
    )


def test_worker_resigns_with_frozen_published_threshold(tmp_path: Path) -> None:
    game = FakeGame(resignation_configuration=resignation_configuration(0.1))
    game.search.root_value = -0.9
    game.search.highest_visited_child_q = -0.9
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.update_resignation_policy(PublishedResignationPolicy(threshold=-0.85))
    worker.refresh_published_model(checkpoint(tmp_path, 50), flat_stop_policy())
    worker.active_games[0].is_resignation_continuation = False

    worker.run_batch()

    completed = CompletedSelfPlayGame.model_validate_json(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert completed.termination_reason is TerminationReason.RESIGNATION
    assert completed.action_ids == ()
    assert completed.observations[-1].selected_action_id is None
    assert completed.final_wdl == WdlTarget(win=0.0, draw=0.0, loss=1.0)


def test_continuation_game_never_resigns_and_keeps_creation_threshold(tmp_path: Path) -> None:
    game = FakeGame(resignation_configuration=resignation_configuration(1.0))
    game.search.root_value = -0.9
    game.search.highest_visited_child_q = -0.9
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.update_resignation_policy(PublishedResignationPolicy(threshold=-0.85))
    worker.refresh_published_model(checkpoint(tmp_path, 50), flat_stop_policy())
    active = worker.active_games[0]
    worker.update_resignation_policy(PublishedResignationPolicy(threshold=-0.80))

    worker.run_batch()

    assert worker.active_games[0] is active
    assert active.is_resignation_continuation is True
    assert active.resignation_threshold == pytest.approx(-0.85)
    assert active.action_ids == [0]


class FakeOpeningRandom:
    def __init__(self, opening_lengths: tuple[int, ...]) -> None:
        self.opening_lengths = iter(opening_lengths)

    def integers(self, low: int, high: int) -> int:
        assert (low, high) == (0, 13)
        return next(self.opening_lengths)

    def choice(self, values: tuple[int, ...]) -> int:
        return values[0]


def test_worker_samples_random_opening_length_from_zero_through_configured_maximum(tmp_path: Path) -> None:
    game = FakeGame(maximum_random_opening_plies=12)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=2,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.random = cast(np.random.Generator, FakeOpeningRandom((0, 12)))

    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert [active_game.root.position.ply for active_game in worker.active_games] == [0, 12]
    assert [active_game.action_ids for active_game in worker.active_games] == [[], [0] * 12]


def restart_parameters(true_start_probability: float = 0.5) -> RestartStateStartParameters:
    return RestartStateStartParameters(
        kind='restart_state',
        true_start_probability=true_start_probability,
        candidate_visit_mass=0.85,
        minimum_candidates=2,
        maximum_candidates=3,
        maximum_absolute_root_value=0.3,
        minimum_remaining_plies=15,
        maximum_archive_positions=50_000,
        maximum_age_generations=20,
    )


class FakeRestartRandom:
    def __init__(self, probability_draw: float) -> None:
        self.probability_draw = probability_draw

    def random(self) -> float:
        return self.probability_draw


def restart_source_game() -> CompletedSelfPlayGame:
    action_ids = (0,) * 15
    return CompletedSelfPlayGame(
        identity=GameIdentity(worker_id=0, process_instance_id=UUID(int=1), game_number=0),
        created_at_seconds=1.0,
        generation_seconds=1.0,
        action_ids=action_ids,
        observations=(
            SearchObservation(
                ply=0,
                model_generation=0,
                policy_target_visits=SearchVisitCounts.from_native(
                    (
                        GameSearchVisit(action_id=0, visit_count=45),
                        GameSearchVisit(action_id=1, visit_count=30),
                        GameSearchVisit(action_id=2, visit_count=25),
                    )
                ),
                root_value=0.0,
                highest_visited_child_action_id=0,
                highest_visited_child_visit_count=45,
                highest_visited_child_q=0.0,
                selected_action_id=0,
                sample_weight=1.0,
                baseline_visits=256,
                network_root_value=0.0,
                policy_correction=0.0,
                value_correction=0.0,
                stop_checkpoint_index=-1,
                parallel_searches=2,
                starting_visits=0,
                final_visits=256,
                stop_reason=SearchStopReason.FIXED_LIMIT,
            ),
        ),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.MAXIMUM_PLIES,
    )


def test_restart_policy_uses_exact_initial_states_without_random_openings(tmp_path: Path) -> None:
    worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=restart_parameters(1.0))),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'inbox',
    )
    worker.random = cast(np.random.Generator, FakeRestartRandom(0.0))

    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert worker.active_games[0].action_ids == []
    assert worker.active_games[0].root.position == FakePosition(0)
    assert worker.true_starts == 1
    worker.close()


def test_restart_policy_falls_back_to_exact_start_when_archive_is_empty(tmp_path: Path) -> None:
    worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=restart_parameters())),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'inbox',
    )
    worker.random = cast(np.random.Generator, FakeRestartRandom(0.9))

    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert worker.active_games[0].action_ids == []
    assert worker.empty_restart_fallbacks == 1
    worker.close()


def test_restart_root_uses_learned_budget_and_plays_reserved_candidate(tmp_path: Path) -> None:
    parameters = restart_parameters()
    completed_games_path = tmp_path / 'completed-games'
    archive = RestartStateArchive(worker_restart_archive_path(completed_games_path, 0))
    archive.archive_completed_game(restart_source_game(), parameters)
    archive.close()
    game = FakeGame(restart_parameters=parameters)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=completed_games_path / 'inbox',
    )
    worker.random = cast(np.random.Generator, FakeRestartRandom(0.9))
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    worker.run_batch()

    active_game = worker.active_games[0]
    assert active_game.action_ids == [1]
    assert active_game.observations[0].final_visits - active_game.observations[0].starting_visits == 3
    assert active_game.observations[0].selected_action_id == 1
    assert worker.restart_starts == 1
    worker.close()


def test_restart_archives_are_private_to_each_worker_and_survive_replacement(tmp_path: Path) -> None:
    parameters = restart_parameters()
    completed_games_path = tmp_path / 'completed-games'
    first_path = worker_restart_archive_path(completed_games_path, 0)
    archive = RestartStateArchive(first_path)
    archive.archive_completed_game(restart_source_game(), parameters)
    archive.close()

    first_worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=parameters)),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=completed_games_path / 'inbox',
    )
    other_worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=parameters)),
        parallel_game_count=1,
        worker_id=1,
        device_id=0,
        inbox_path=completed_games_path / 'inbox',
    )
    first_worker.random = cast(np.random.Generator, FakeRestartRandom(0.9))
    other_worker.random = cast(np.random.Generator, FakeRestartRandom(0.9))

    first_worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())
    other_worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert first_worker.restart_archive_path == first_path
    assert first_worker.restart_starts == 1
    assert other_worker.restart_archive_path == worker_restart_archive_path(completed_games_path, 1)
    assert other_worker.restart_starts == 0
    assert other_worker.empty_restart_fallbacks == 1
    first_worker.close()
    other_worker.close()

    replacement = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=parameters)),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=completed_games_path / 'inbox',
    )
    replacement.random = cast(np.random.Generator, FakeRestartRandom(0.9))
    replacement.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert replacement.restart_archive_path == first_path
    assert replacement.restart_starts == 1
    replacement.close()


def test_cut_game_bootstraps_its_value_from_the_last_search_root_value(tmp_path: Path) -> None:
    oracle = RecordingTerminalOracle(None)
    game = FakeGame(maximum_game_plies=1, terminal_oracle=oracle, bootstrap_cut_game_value=True)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    # The first batch reaches the cut and defers; the second searches the cut position itself.
    worker.run_batch()
    assert not list(tmp_path.glob('*.json'))
    worker.run_batch()

    completed = CompletedSelfPlayGame.model_validate_json(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert oracle.probed_positions == [FakePosition(1)]
    trailing = completed.observations[-1]
    assert trailing.ply == len(completed.action_ids)
    assert trailing.selected_action_id is None
    assert trailing.final_visits - trailing.starting_visits == 3
    assert completed.final_wdl == WdlTarget.from_scalar(trailing.root_value)


def test_cut_game_falls_back_to_adjudication_when_bootstrapping_is_disabled(tmp_path: Path) -> None:
    game = FakeGame(maximum_game_plies=1, terminal_oracle=RecordingTerminalOracle(None))
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    worker.run_batch()

    completed = CompletedSelfPlayGame.model_validate_json(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert completed.final_wdl == WdlTarget(win=0.0, draw=1.0, loss=0.0)


def _open_stop_policy() -> SearchStopPolicy:
    # Multiples that stay distinct at the fake game's baseline of three visits.
    return SearchStopPolicy(
        checkpoint_multiples=(1.0 / 3.0, 2.0 / 3.0, 1.0, 1.5),
        thresholds=(0.1, 0.1, 0.1, 0.1),
        movement_guard_epsilon=0.05,
        cap_multiple=2.0,
        predictor_path=Path('stop-predictor.jit.pt'),
        predictor_sha256='0' * 64,
        apply_learned=True,
    )


def _closed_capable_policy() -> SearchStopPolicy:
    return SearchStopPolicy(
        checkpoint_multiples=(1.0 / 3.0, 2.0 / 3.0, 1.0, 1.5),
        thresholds=(0.0,) * 4,
        movement_guard_epsilon=0.05,
        cap_multiple=2.0,
        predictor_path=None,
        predictor_sha256=None,
        apply_learned=False,
    )


def test_applied_policy_requests_carry_the_full_checkpoint_set(tmp_path: Path) -> None:
    """The v21 crash-loop regression: under an applied policy the native contract requires one
    checkpoint visit count per configured multiple on EVERY capped request, not only on audits."""
    game = FakeGame()
    worker = SelfPlayWorker(
        game=cast('GameImplementation', game),
        parallel_game_count=3,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'completed-games' / 'inbox',
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), _open_stop_policy())
    worker.run_batch()

    policy = _open_stop_policy()
    for request in worker.search.request_batches[0][:3]:  # type: ignore[union-attr]
        visits = request.policy_checkpoint_visits
        assert visits is not None and len(visits) == len(policy.checkpoint_multiples)
        assert visits == sorted(set(visits)) and visits[0] > 0


def test_closed_policy_non_audit_requests_carry_no_checkpoints(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        game=cast('GameImplementation', game),
        parallel_game_count=3,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'completed-games' / 'inbox',
    )
    worker.audit_fraction = Fraction(0)  # deterministically no audits
    worker.refresh_published_model(checkpoint(tmp_path, 0), _closed_capable_policy())
    worker.run_batch()

    for request in worker.search.request_batches[0]:  # type: ignore[union-attr]
        assert not request.policy_checkpoint_visits


def test_audit_requests_under_a_closed_policy_still_search_to_the_cap(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        game=cast('GameImplementation', game),
        parallel_game_count=3,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'completed-games' / 'inbox',
    )
    worker.audit_fraction = Fraction(1)  # deterministically all audits
    worker.paired_audit_fraction = Fraction(0)
    worker.anchor_fraction = Fraction(0)
    worker.refresh_published_model(checkpoint(tmp_path, 0), _closed_capable_policy())
    worker.run_batch()

    policy = _closed_capable_policy()
    for request in worker.search.request_batches[0][:3]:  # type: ignore[union-attr]
        visits = request.policy_checkpoint_visits
        assert visits is not None and len(visits) == len(policy.checkpoint_multiples)


def test_suspended_games_are_resumed_with_their_plies_and_observations(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    inbox.mkdir(parents=True)

    def worker() -> SelfPlayWorker:
        return SelfPlayWorker(
            cast(GameImplementation, FakeGame()),
            parallel_game_count=3,
            worker_id=0,
            device_id=0,
            inbox_path=inbox,
        )

    first = worker()
    first.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())
    first.run_batch()
    first.run_batch()
    before = [(game.identity, list(game.action_ids), len(game.observations)) for game in first.active_games]
    assert first.suspend_active_games() == 3

    second = worker()
    second.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert second.restored_games == 3
    assert [(game.identity, list(game.action_ids), len(game.observations)) for game in second.active_games] == before
    assert [game.root.position.ply for game in second.active_games] == [2, 2, 2]


def test_a_resumed_worker_starts_fresh_games_when_nothing_was_suspended(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    inbox.mkdir(parents=True)
    worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame()),
        parallel_game_count=2,
        worker_id=0,
        device_id=0,
        inbox_path=inbox,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0), flat_stop_policy())

    assert worker.restored_games == 0
    assert [game.action_ids for game in worker.active_games] == [[], []]
