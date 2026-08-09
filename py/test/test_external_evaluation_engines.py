from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import chess
import chess.engine
import pytest

from src.evaluation.configuration import KataGoEngineConfiguration, StockfishEngineConfiguration
from src.games.chess.stockfish import StockfishClient
from src.games.go.contract import GoStateContract
from src.games.go.katago import KataGoClient, action_id_to_gtp, action_id_to_sgf, gtp_to_action_id


class FakeChessPosition:
    fen = chess.STARTING_FEN

    def action_id_from_uci(self, move_uci: str) -> int:
        return {'e2e4': 1, 'd2d4': 2}[move_uci]


class FakeStockfishEngine:
    id = {'name': 'Fake Stockfish'}

    def __init__(self) -> None:
        self.configurations: list[dict[str, int | bool]] = []

    def configure(self, configuration: dict[str, int | bool]) -> None:
        self.configurations.append(configuration)

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: int,
        info: chess.engine.Info,
    ) -> list[dict[str, object]]:
        assert multipv == 2
        return [
            {'pv': [chess.Move.from_uci('e2e4')], 'score': chess.engine.PovScore(chess.engine.Cp(80), board.turn)},
            {'pv': [chess.Move.from_uci('d2d4')], 'score': chess.engine.PovScore(chess.engine.Cp(20), board.turn)},
        ]

    def quit(self) -> None:
        pass


def test_stockfish_multipv_scores_form_normalized_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / 'stockfish'
    executable.write_bytes(b'engine')
    engine = FakeStockfishEngine()
    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda path: engine)
    configuration = StockfishEngineConfiguration(
        kind='stockfish',
        executable_path=str(executable),
        label_nodes=100,
        match_nodes=10,
        threads=1,
        hash_mib=16,
        multi_pv=2,
        policy_softmax_temperature=0.2,
    )
    state = type(
        'FakeState', (), {'representation': type('R', (), {'channels': 1, 'rows': 1, 'columns': 1})(), 'action_size': 3}
    )()

    client = StockfishClient(configuration, state, executable)
    policy = client.policy(FakeChessPosition(), ())

    assert sum(entry.probability for entry in policy.entries) == pytest.approx(1.0)
    assert policy.top_action_id == 1
    assert engine.configurations[0] == {'Threads': 1, 'Hash': 16, 'UCI_ShowWDL': True}


@pytest.mark.parametrize('board_size', (7, 9))
def test_go_gtp_coordinates_round_trip(board_size: int) -> None:
    for action_id in range(board_size * board_size + 1):
        assert gtp_to_action_id(action_id_to_gtp(action_id, board_size), board_size) == action_id


def test_go_sgf_coordinates_use_sgf_not_gtp_notation() -> None:
    assert action_id_to_sgf(0, 7) == 'aa'
    assert action_id_to_sgf(48, 7) == 'gg'
    assert action_id_to_sgf(49, 7) == ''


class FakeKataGoProcess:
    def __init__(self, output: str) -> None:
        self.stdin = StringIO()
        self.stdout = StringIO(output)
        self.stderr = StringIO()
        self._return_code: int | None = None
        self.was_terminated = False

    def poll(self) -> int | None:
        return self._return_code

    def wait(self, timeout: float | None = None) -> int:
        self._return_code = 0
        return 0

    def terminate(self) -> None:
        self.was_terminated = True
        self._return_code = -15

    def kill(self) -> None:
        self._return_code = -9


def test_katago_analysis_matches_out_of_order_responses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = tuple(tmp_path / name for name in ('katago', 'model.bin.gz', 'analysis.cfg'))
    for path in paths:
        path.write_bytes(path.name.encode('utf-8'))
    responses = '\n'.join(
        (
            json.dumps({'id': 'evaluation-1', 'moveInfos': [{'move': 'pass', 'weight': 1.0}]}),
            json.dumps(
                {
                    'id': 'evaluation-0',
                    'moveInfos': [
                        {'move': 'A7', 'weight': 3.0},
                        {'move': 'B7', 'weight': 1.0},
                    ],
                }
            ),
        )
    )
    process = FakeKataGoProcess(responses + '\n')
    monkeypatch.setattr('subprocess.Popen', lambda *arguments, **keywords: process)
    configuration = KataGoEngineConfiguration(
        kind='katago',
        executable_path=str(paths[0]),
        model_path=str(paths[1]),
        analysis_configuration_path=str(paths[2]),
        label_max_visits=32,
        match_max_visits=16,
    )
    state = GoStateContract(7, komi_half_points=15, maximum_moves=196)
    client = KataGoClient(configuration, state, *paths)

    policies = client.analyze_many(((), (0,)), 32)
    submitted = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    client.close()

    assert policies[0].top_action_id == 0
    assert policies[1].top_action_id == state.pass_action
    assert submitted[1]['moves'] == [['B', 'A7']]
    assert all(request['rules'] == 'chinese' and request['komi'] == 7.5 for request in submitted)
    assert process.was_terminated
