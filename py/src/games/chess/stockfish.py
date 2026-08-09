from __future__ import annotations

import hashlib
import json
from math import exp
from pathlib import Path
from types import TracebackType

import chess
import chess.engine
import chess.pgn

from src.evaluation.configuration import StockfishEngineConfiguration
from src.evaluation.engine import EnginePolicy, EnginePolicyEntry
from src.games.chess.contract import ChessPosition, ChessStateContract


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_payload(payload: dict[str, int | str]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


class StockfishClient:
    def __init__(
        self,
        configuration: StockfishEngineConfiguration,
        state: ChessStateContract,
        executable_path: Path,
    ) -> None:
        if not executable_path.is_file():
            raise ValueError(f'Stockfish executable does not exist: {executable_path}')
        self.configuration = configuration
        self.state = state
        self.executable_path = executable_path
        self._engine = chess.engine.SimpleEngine.popen_uci(str(executable_path))
        self._engine.configure(
            {
                'Threads': configuration.threads,
                'Hash': configuration.hash_mib,
                'UCI_ShowWDL': True,
            }
        )

    @property
    def game_name(self) -> str:
        return 'chess'

    @property
    def rules_digest(self) -> str:
        return _digest_payload({'rules': 'native-chess-training-v1'})

    @property
    def representation_digest(self) -> str:
        representation = self.state.representation
        return _digest_payload(
            {
                'channels': representation.channels,
                'rows': representation.rows,
                'columns': representation.columns,
                'actions': self.state.action_size,
            }
        )

    @property
    def engine_identity(self) -> str:
        return str(self._engine.id.get('name', 'Stockfish'))

    @property
    def engine_artifact_sha256(self) -> tuple[str, ...]:
        return (_file_sha256(self.executable_path),)

    @property
    def label_search_limit(self) -> int:
        return self.configuration.label_nodes

    def policy(self, position: ChessPosition, action_ids: tuple[int, ...]) -> EnginePolicy:
        board = chess.Board(position.fen)
        legal_count = board.legal_moves.count()
        analysis = self._engine.analyse(
            board,
            chess.engine.Limit(nodes=self.configuration.label_nodes),
            multipv=min(self.configuration.multi_pv, legal_count),
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        records = analysis if isinstance(analysis, list) else [analysis]
        scored_actions: list[tuple[int, float]] = []
        for record in records:
            variation = record.get('pv')
            score = record.get('score')
            if not variation or not isinstance(score, chess.engine.PovScore):
                raise ValueError('Stockfish MultiPV result omitted its move or score.')
            expected_score = score.pov(board.turn).wdl(ply=board.ply()).expectation()
            scored_actions.append((position.action_id_from_uci(variation[0].uci()), expected_score))
        maximum = max(score for _, score in scored_actions)
        unnormalized = tuple(
            exp((score - maximum) / self.configuration.policy_softmax_temperature) for _, score in scored_actions
        )
        total = sum(unnormalized)
        return EnginePolicy(
            tuple(
                EnginePolicyEntry(action_id, weight / total)
                for (action_id, _), weight in zip(scored_actions, unnormalized, strict=True)
            )
        )

    def choose_action(self, position: ChessPosition, skill_level: int) -> int:
        self._engine.configure({'Skill Level': skill_level})
        result = self._engine.play(
            chess.Board(position.fen),
            chess.engine.Limit(nodes=self.configuration.match_nodes),
        )
        if result.move is None:
            raise ValueError('Stockfish returned no move for a nonterminal position.')
        return position.action_id_from_uci(result.move.uci())

    def render_game(self, action_ids: tuple[int, ...]) -> str:
        game = chess.pgn.Game()
        board = chess.Board()
        native_position = self.state.initial_position()
        node = game
        for action_id in action_ids:
            move = chess.Move.from_uci(native_position.action_uci(action_id))
            node = node.add_variation(move)
            board.push(move)
            native_position = self.state.child_position(native_position, action_id)
        return str(game)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> StockfishClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
