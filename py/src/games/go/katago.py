from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import TextIO
from types import TracebackType

from pydantic import JsonValue, TypeAdapter

from src.evaluation.configuration import KataGoEngineConfiguration
from src.evaluation.engine import EnginePolicy, EnginePolicyEntry
from src.games.go.contract import GoStateContract, NativeGoPosition


JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])
MOVE_INFO_LIST_ADAPTER = TypeAdapter(list[dict[str, JsonValue]])


@dataclass(frozen=True)
class _KataGoPolicyResponse:
    request_id: str
    weighted_actions: tuple[tuple[int, float], ...]


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


def _parse_weighted_actions(move_infos: JsonValue, board_size: int) -> tuple[tuple[int, float], ...]:
    entries = MOVE_INFO_LIST_ADAPTER.validate_python(move_infos)
    weighted_actions: list[tuple[int, float]] = []
    for entry in entries:
        move = entry.get('move')
        weight = entry.get('weight')
        if not isinstance(move, str) or not isinstance(weight, int | float) or isinstance(weight, bool) or weight <= 0:
            continue
        weighted_actions.append((gtp_to_action_id(move, board_size), float(weight)))
    if not weighted_actions:
        raise ValueError('KataGo response contained no positive move weights.')
    return tuple(weighted_actions)


def _parse_policy_response(line: str, expected_ids: set[str], board_size: int) -> _KataGoPolicyResponse:
    response = JSON_OBJECT_ADAPTER.validate_json(line)
    request_id = response.get('id')
    if not isinstance(request_id, str) or request_id not in expected_ids:
        raise ValueError('KataGo response has an unexpected request ID.')
    if 'error' in response:
        raise RuntimeError(f'KataGo analysis failed: {response["error"]}')
    move_infos = response.get('moveInfos')
    if move_infos is None:
        raise ValueError('KataGo response omitted moveInfos.')
    return _KataGoPolicyResponse(request_id, _parse_weighted_actions(move_infos, board_size))


def _normalized_policy(weighted_actions: tuple[tuple[int, float], ...]) -> EnginePolicy:
    total = sum(weight for _, weight in weighted_actions)
    return EnginePolicy(tuple(EnginePolicyEntry(action_id, weight / total) for action_id, weight in weighted_actions))


def action_id_to_sgf(action_id: int, board_size: int) -> str:
    if action_id == board_size * board_size:
        return ''
    if not 0 <= action_id < board_size * board_size:
        raise ValueError('Go action ID is outside the board action space.')
    return f'{chr(ord("a") + action_id % board_size)}{chr(ord("a") + action_id // board_size)}'


class KataGoClient:
    def __init__(
        self,
        configuration: KataGoEngineConfiguration,
        state: GoStateContract,
        executable_path: Path,
        model_path: Path,
        analysis_configuration_path: Path,
        visible_cuda_device: int | None = None,
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
        process_environment = (
            None if visible_cuda_device is None else {**os.environ, 'CUDA_VISIBLE_DEVICES': str(visible_cuda_device)}
        )
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
            stderr=None,
            text=True,
            encoding='utf-8',
            bufsize=1,
            env=process_environment,
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

    def _receive(self, expected_ids: set[str]) -> dict[str, EnginePolicy]:
        responses: dict[str, EnginePolicy] = {}
        while responses.keys() != expected_ids:
            line = self._output.readline()
            if not line:
                raise RuntimeError(f'KataGo analysis process exited with code {self._process.poll()}.')
            response = _parse_policy_response(line, expected_ids, self.state.board_size)
            responses[response.request_id] = _normalized_policy(response.weighted_actions)
        return responses

    def analyze_many(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
        maximum_visits: int,
    ) -> tuple[EnginePolicy, ...]:
        request_ids = tuple(self._submit(action_ids, maximum_visits) for action_ids in action_sequences)
        responses = self._receive(set(request_ids))
        return tuple(responses[request_id] for request_id in request_ids)

    def policy(self, position: NativeGoPosition, action_ids: tuple[int, ...]) -> EnginePolicy:
        return self.analyze_many((action_ids,), self.configuration.label_max_visits)[0]

    def choose_actions(
        self,
        action_sequences: tuple[tuple[int, ...], ...],
        maximum_visits: int,
    ) -> tuple[int, ...]:
        return tuple(policy.top_action_id for policy in self.analyze_many(action_sequences, maximum_visits))

    def render_game(self, action_ids: tuple[int, ...]) -> str:
        moves = ''.join(
            f';{"B" if ply % 2 == 0 else "W"}[{action_id_to_sgf(action_id, self.state.board_size)}]'
            for ply, action_id in enumerate(action_ids)
        )
        return f'(;GM[1]FF[4]SZ[{self.state.board_size}]KM[{self.state.komi_half_points / 2.0}]{moves})'

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._input.close()

    def __enter__(self) -> KataGoClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class KataGoMatchEngine:
    def __init__(self, client: KataGoClient, maximum_visits: int) -> None:
        self.client = client
        self.maximum_visits = maximum_visits

    def choose_actions(
        self,
        positions: tuple[NativeGoPosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if len(positions) != len(action_sequences):
            raise ValueError('KataGo match positions and histories must have equal lengths.')
        return self.client.choose_actions(action_sequences, self.maximum_visits)

    def close(self) -> None:
        self.client.close()
