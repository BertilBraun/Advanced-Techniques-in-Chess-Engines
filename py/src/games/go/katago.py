from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import TextIO

from src.evaluation.configuration import KataGoEngineConfiguration
from src.evaluation.engine import EnginePolicy, EnginePolicyEntry
from src.games.go.contract import GoStateContract, NativeGoPosition


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_payload(payload: dict[str, int | str]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def action_id_to_gtp(action_id: int, board_size: int) -> str:
    if action_id == board_size * board_size:
        return 'pass'
    if not 0 <= action_id < board_size * board_size:
        raise ValueError('Go action ID is outside the board action space.')
    x = action_id % board_size
    y = action_id // board_size
    column = chr(ord('A') + x + (1 if x >= 8 else 0))
    return f'{column}{board_size - y}'


def gtp_to_action_id(coordinate: str, board_size: int) -> int:
    if coordinate.casefold() == 'pass':
        return board_size * board_size
    if len(coordinate) < 2:
        raise ValueError(f'Invalid Go coordinate: {coordinate!r}')
    column = ord(coordinate[0].upper()) - ord('A')
    if column >= 8:
        column -= 1
    row = int(coordinate[1:])
    y = board_size - row
    action_id = y * board_size + column
    if not 0 <= action_id < board_size * board_size:
        raise ValueError(f'Go coordinate is outside the board: {coordinate!r}')
    return action_id


class KataGoClient:
    def __init__(
        self,
        configuration: KataGoEngineConfiguration,
        state: GoStateContract,
        executable_path: Path,
        model_path: Path,
        analysis_configuration_path: Path,
    ) -> None:
        for name, path in (
            ('KataGo executable', executable_path),
            ('KataGo model', model_path),
            ('KataGo analysis configuration', analysis_configuration_path),
        ):
            if not path.is_file():
                raise ValueError(f'{name} does not exist: {path}')
        self.configuration = configuration
        self.state = state
        self.executable_path = executable_path
        self.model_path = model_path
        self.analysis_configuration_path = analysis_configuration_path
        self._next_request_id = 0
        self._process = subprocess.Popen(
            [
                str(executable_path),
                'analysis',
                '-config',
                str(analysis_configuration_path),
                '-model',
                str(model_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError('KataGo analysis process did not expose text pipes.')
        self._input: TextIO = self._process.stdin
        self._output: TextIO = self._process.stdout

    @property
    def game_name(self) -> str:
        return 'go'

    @property
    def rules_digest(self) -> str:
        return _digest_payload(
            {
                'rules': 'chinese',
                'board_size': self.state.board_size,
                'komi_half_points': self.state.komi_half_points,
                'maximum_moves': self.state.maximum_moves or self.state.board_size**2 * 4,
            }
        )

    @property
    def representation_digest(self) -> str:
        return _digest_payload(
            {
                'channels': self.state.channels,
                'board_size': self.state.board_size,
                'actions': self.state.action_size,
            }
        )

    @property
    def engine_identity(self) -> str:
        return 'KataGo analysis'

    @property
    def engine_artifact_sha256(self) -> tuple[str, ...]:
        return (
            _file_sha256(self.executable_path),
            _file_sha256(self.model_path),
            _file_sha256(self.analysis_configuration_path),
        )

    @property
    def label_search_limit(self) -> int:
        return self.configuration.label_max_visits

    def _moves(self, action_ids: tuple[int, ...]) -> list[list[str]]:
        moves: list[list[str]] = []
        for ply, action_id in enumerate(action_ids):
            moves.append(
                [
                    'B' if ply % 2 == 0 else 'W',
                    action_id_to_gtp(action_id, self.state.board_size),
                ]
            )
        return moves

    def _submit(self, action_ids: tuple[int, ...], maximum_visits: int) -> str:
        request_id = f'evaluation-{self._next_request_id}'
        self._next_request_id += 1
        request = {
            'id': request_id,
            'initialPlayer': 'B',
            'moves': self._moves(action_ids),
            'rules': 'chinese',
            'komi': self.state.komi_half_points / 2.0,
            'boardXSize': self.state.board_size,
            'boardYSize': self.state.board_size,
            'maxVisits': maximum_visits,
            'includePolicy': True,
        }
        self._input.write(json.dumps(request, separators=(',', ':')) + '\n')
        self._input.flush()
        return request_id

    def _receive(self, expected_ids: set[str]) -> dict[str, dict[str, object]]:
        responses: dict[str, dict[str, object]] = {}
        while responses.keys() != expected_ids:
            line = self._output.readline()
            if not line:
                raise RuntimeError(f'KataGo analysis process exited with code {self._process.poll()}.')
            response = json.loads(line)
            if not isinstance(response, dict):
                raise ValueError('KataGo response must be a JSON object.')
            request_id = response.get('id')
            if not isinstance(request_id, str) or request_id not in expected_ids:
                raise ValueError('KataGo response has an unexpected request ID.')
            if 'error' in response:
                raise RuntimeError(f'KataGo analysis failed: {response["error"]}')
            responses[request_id] = response
        return responses

    def analyze_many(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
        maximum_visits: int,
    ) -> tuple[EnginePolicy, ...]:
        request_ids = tuple(self._submit(action_ids, maximum_visits) for action_ids in action_sequences)
        responses = self._receive(set(request_ids))
        policies = []
        for request_id in request_ids:
            move_infos = responses[request_id].get('moveInfos')
            if not isinstance(move_infos, list) or not move_infos:
                raise ValueError('KataGo response omitted moveInfos.')
            weighted_actions: list[tuple[int, float]] = []
            for move_info in move_infos:
                if not isinstance(move_info, dict):
                    raise ValueError('KataGo moveInfos entries must be objects.')
                move = move_info.get('move')
                weight = move_info.get('weight')
                if not isinstance(move, str) or not isinstance(weight, int | float) or weight <= 0:
                    continue
                weighted_actions.append((gtp_to_action_id(move, self.state.board_size), float(weight)))
            if not weighted_actions:
                raise ValueError('KataGo response contained no positive move weights.')
            total = sum(weight for _, weight in weighted_actions)
            policies.append(
                EnginePolicy(
                    tuple(EnginePolicyEntry(action_id, weight / total) for action_id, weight in weighted_actions)
                )
            )
        return tuple(policies)

    def policy(self, position: NativeGoPosition, action_ids: tuple[int, ...]) -> EnginePolicy:
        return self.analyze_many((action_ids,), self.configuration.label_max_visits)[0]

    def choose_actions(self, action_sequences: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        return tuple(
            policy.top_action_id for policy in self.analyze_many(action_sequences, self.configuration.match_max_visits)
        )

    def render_game(self, action_ids: tuple[int, ...]) -> str:
        moves = ''.join(
            f';{"B" if ply % 2 == 0 else "W"}[{action_id_to_gtp(action_id, self.state.board_size)}]'
            for ply, action_id in enumerate(action_ids)
        )
        return f'(;GM[1]FF[4]SZ[{self.state.board_size}]KM[{self.state.komi_half_points / 2.0}]{moves})'

    def close(self) -> None:
        if self._process.poll() is None:
            self._input.close()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=10)

    def __enter__(self) -> KataGoClient:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: object
    ) -> None:
        self.close()
