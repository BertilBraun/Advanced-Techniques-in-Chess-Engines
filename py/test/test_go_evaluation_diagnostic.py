from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.evaluation.contracts import OpeningLine, OpeningSuiteManifest
from src.evaluation.go_diagnostic import (
    DiagnosticEngine,
    GoDiagnosticMatchResult,
    KataGoDiagnosticEngine,
    run_go_diagnostic_match,
    validate_deterministic_identity,
    write_go_diagnostic_match,
)
from src.games.contracts import Player, WdlTarget
from src.games.go.contract import GoStateContract
from src.games.go.katago import KataGoScoreEstimate
from src.self_play.completed_game import TerminationReason


@dataclass(frozen=True)
class FakeEnum:
    name: str


@dataclass(frozen=True)
class FakeAreaScore:
    black_half_points: int
    white_half_points: int
    winner: FakeEnum | None


@dataclass(frozen=True)
class FakePosition:
    actions: tuple[int, ...]

    @property
    def board_size(self) -> int:
        return 7

    @property
    def player(self) -> FakeEnum:
        return FakeEnum('BLACK' if len(self.actions) % 2 == 0 else 'WHITE')

    @property
    def is_terminal(self) -> bool:
        return len(self.actions) == 6

    @property
    def termination_reason(self) -> FakeEnum:
        return FakeEnum('TWO_PASSES' if self.is_terminal else 'NONE')

    def legal_actions(self) -> list[int]:
        return [] if self.is_terminal else [0, 1]

    def child(self, action_id: int) -> FakePosition:
        return FakePosition((*self.actions, action_id))

    def terminal_value(self) -> float:
        return 1.0

    def area_score(self) -> FakeAreaScore:
        return FakeAreaScore(20, 10, FakeEnum('BLACK'))

    def packed_encoding(self) -> bytes:
        return bytes(GoStateContract(7).packed_planes.payload_bytes)


class FakeGoState(GoStateContract):
    def initial_position(self) -> FakePosition:
        return FakePosition(())

    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        return tuple(position.legal_actions())

    def child_position(self, position: FakePosition, action_id: int) -> FakePosition:
        return position.child(action_id)

    def current_player(self, position: FakePosition) -> Player:
        return Player.FIRST if position.player.name == 'BLACK' else Player.SECOND

    def natural_terminal_wdl(self, position: FakePosition) -> WdlTarget | None:
        if not position.is_terminal:
            return None
        return WdlTarget(win=1.0, draw=0.0, loss=0.0)

    def adjudicated_wdl(self, position: FakePosition, reason: TerminationReason) -> WdlTarget:
        return WdlTarget(win=1.0, draw=0.0, loss=0.0)


class FakeSelector:
    def __init__(self, engine: DiagnosticEngine, selected_action: int = 0) -> None:
        self._engine = engine
        self.selected_action = selected_action

    @property
    def engine(self) -> DiagnosticEngine:
        return self._engine

    def choose_actions(
        self,
        positions: tuple[FakePosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        assert len(positions) == len(action_sequences)
        return (self.selected_action,) * len(positions)

    def final_score_estimates(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[KataGoScoreEstimate | None, ...]:
        return (None,) * len(action_sequences)

    def close(self) -> None:
        pass


def _openings(count: int = 2) -> OpeningSuiteManifest:
    return OpeningSuiteManifest(
        game='go',
        rules_digest='0' * 64,
        representation_digest='1' * 64,
        random_seed=1,
        engine_identity='fake',
        engine_artifact_sha256=('2' * 64,),
        label_search_limit=16,
        expanded_actions_per_position=2,
        beam_width=count,
        openings=tuple(
            OpeningLine(
                opening_id=f'opening-{index}',
                action_ids=(0, 0, 0, 0),
                path_probability=0.5,
                final_position_digest=f'{index + 3:064x}',
                human_readable='sgf',
            )
            for index in range(count)
        ),
        builder_source_revision='revision',
    )


def _engine(engine_id: str = 'same') -> KataGoDiagnosticEngine:
    return KataGoDiagnosticEngine(kind='katago', engine_id=engine_id, maximum_visits=16)


def _run(
    measured: FakeSelector,
    opponent: FakeSelector,
) -> GoDiagnosticMatchResult:
    return run_go_diagnostic_match(
        'test',
        FakeGoState(7),
        _openings(),
        2,
        measured,
        opponent,
        20,
        10,
        100,
    )


def test_go_diagnostic_pairs_openings_and_swaps_engine_roles() -> None:
    measured = FakeSelector(_engine('measured'))
    opponent = FakeSelector(_engine('opponent'))

    result = _run(measured, opponent)

    assert tuple(game.game.pair_index for game in result.games) == (0, 0, 1, 1)
    assert tuple(game.players.measured_player for game in result.games) == (
        'first',
        'second',
        'first',
        'second',
    )
    assert result.games[0].players.first.engine_id == 'measured'
    assert result.games[1].players.first.engine_id == 'opponent'
    assert all(len(game.complete_action_ids) == 6 for game in result.games)
    assert all(game.sgf.startswith('(;GM[1]') for game in result.games)


def test_deterministic_identity_requires_equal_paired_histories_and_exact_half_score() -> None:
    engine = _engine()
    result = _run(FakeSelector(engine), FakeSelector(engine))

    validate_deterministic_identity(result)

    games = list(result.games)
    games[1] = games[1].model_copy(update={'complete_action_ids': (*games[1].complete_action_ids[:-1], 1)})
    divergent = result.model_copy(update={'games': tuple(games)})
    with pytest.raises(RuntimeError, match='different action histories'):
        validate_deterministic_identity(divergent)


def test_go_diagnostic_persists_complete_typed_output(tmp_path: Path) -> None:
    engine = _engine()
    result = _run(FakeSelector(engine), FakeSelector(engine))
    output_path = tmp_path / 'diagnostic.json'

    write_go_diagnostic_match(result, output_path)
    loaded = GoDiagnosticMatchResult.model_validate_json(output_path.read_text(encoding='utf-8'))

    assert loaded == result
    assert len(loaded.games[0].complete_action_ids) == 6
    assert loaded.games[0].native_area_score.black_minus_white_points == 5.0
