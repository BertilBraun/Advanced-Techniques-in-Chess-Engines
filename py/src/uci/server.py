from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from threading import Event, Lock, Thread
from typing import Protocol, TextIO

import chess

from src.eval.LegalMoveSelection import select_legal_analysis_move

ENGINE_NAME = 'AlphaZeroCpp'
ENGINE_AUTHOR = 'AlphaZeroCpp contributors'
MINIMUM_MOVE_TIME_MILLISECONDS = 1_000
MAXIMUM_MOVE_TIME_MILLISECONDS = 30_000


class SearchMode(str, Enum):
    POLICY = 'policy'
    MCTS = 'mcts'


@dataclass(frozen=True)
class PositionCommand:
    starting_fen: str
    moves_uci: tuple[str, ...]


@dataclass(frozen=True)
class SearchRequest:
    mode: SearchMode
    time_limit_seconds: int
    stop_event: Event


class CandidateAnalysis(Protocol):
    move_uci: str


class AnalysisResult(Protocol):
    chosen_move_uci: str
    candidates: tuple[CandidateAnalysis, ...]


class InteractiveGame(Protocol):
    def apply_move(self, move_uci: str) -> None: ...

    def analyze(self, request: SearchRequest) -> AnalysisResult: ...


class InteractiveEngine(Protocol):
    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> InteractiveGame: ...


def parse_position(command: str) -> PositionCommand:
    words = command.split()
    if len(words) < 2 or words[0] != 'position':
        raise ValueError('Expected a position command.')

    if words[1] == 'startpos':
        starting_fen = chess.STARTING_FEN
        moves_index = 2
    elif words[1] == 'fen':
        if len(words) < 8:
            raise ValueError('A UCI FEN position requires all six FEN fields.')
        starting_fen = ' '.join(words[2:8])
        moves_index = 8
    else:
        raise ValueError('Position must use startpos or fen.')

    if moves_index == len(words):
        moves_uci: tuple[str, ...] = ()
    else:
        if words[moves_index] != 'moves':
            raise ValueError('Unexpected text after the position.')
        moves_uci = tuple(words[moves_index + 1 :])

    board = chess.Board(starting_fen)
    for move_uci in moves_uci:
        try:
            move = board.parse_uci(move_uci)
        except ValueError as error:
            raise ValueError(f'Invalid UCI move {move_uci}.') from error
        board.push(move)
    return PositionCommand(starting_fen=starting_fen, moves_uci=moves_uci)


def parse_move_time_seconds(command: str) -> int:
    words = command.split()
    if len(words) != 3 or words[:2] != ['go', 'movetime']:
        raise ValueError('Only go movetime <milliseconds> is supported.')
    try:
        move_time_milliseconds = int(words[2])
    except ValueError as error:
        raise ValueError('Move time must be an integer number of milliseconds.') from error
    if not (MINIMUM_MOVE_TIME_MILLISECONDS <= move_time_milliseconds <= MAXIMUM_MOVE_TIME_MILLISECONDS):
        raise ValueError('Move time must be between 1000 and 30000 milliseconds.')
    return move_time_milliseconds // 1_000


def parse_search_mode(command: str) -> SearchMode:
    prefix = 'setoption name SearchMode value '
    if not command.startswith(prefix):
        raise ValueError('Only the SearchMode option is supported.')
    try:
        return SearchMode(command[len(prefix) :].strip().lower())
    except ValueError as error:
        raise ValueError('SearchMode must be policy or mcts.') from error


class UciServer:
    def __init__(
        self,
        engine: InteractiveEngine,
        standard_output: TextIO = sys.stdout,
        standard_error: TextIO = sys.stderr,
    ) -> None:
        self._engine = engine
        self._standard_output = standard_output
        self._standard_error = standard_error
        self._output_lock = Lock()
        self._position = PositionCommand(chess.STARTING_FEN, ())
        self._board = chess.Board()
        self._game: InteractiveGame | None = None
        self._game_position = PositionCommand(chess.STARTING_FEN, ())
        self._search_mode = SearchMode.MCTS
        self._search_thread: Thread | None = None
        self._stop_event: Event | None = None

    def run(self, standard_input: TextIO = sys.stdin) -> None:
        try:
            for raw_line in standard_input:
                if not self.process(raw_line.strip()):
                    break
        finally:
            self._stop_search(wait=True)

    def process(self, command: str) -> bool:
        if not command:
            return True
        try:
            return self._process_valid_command(command)
        except ValueError as error:
            self._diagnostic(str(error))
            return True

    def _process_valid_command(self, command: str) -> bool:
        if command == 'uci':
            self._write(f'id name {ENGINE_NAME}')
            self._write(f'id author {ENGINE_AUTHOR}')
            self._write('option name SearchMode type combo default mcts var mcts var policy')
            self._write('uciok')
        elif command == 'isready':
            self._write('readyok')
        elif command == 'ucinewgame':
            self._stop_search(wait=True)
            self._position = PositionCommand(chess.STARTING_FEN, ())
            self._board = chess.Board()
            self._game = None
        elif command.startswith('setoption '):
            self._search_mode = parse_search_mode(command)
        elif command.startswith('position '):
            self._stop_search(wait=True)
            self._set_position(parse_position(command))
        elif command.startswith('go '):
            self._start_search(parse_move_time_seconds(command))
        elif command == 'stop':
            self._stop_search(wait=False)
        elif command == 'quit':
            return False
        else:
            raise ValueError(f'Unsupported UCI command: {command}')
        return True

    def _set_position(self, position: PositionCommand) -> None:
        self._position = position
        self._board = chess.Board(position.starting_fen)
        for move_uci in position.moves_uci:
            self._board.push_uci(move_uci)

        can_advance = (
            self._game is not None
            and position.starting_fen == self._game_position.starting_fen
            and position.moves_uci[: len(self._game_position.moves_uci)] == self._game_position.moves_uci
        )
        if not can_advance:
            self._game = None
            return
        assert self._game is not None
        for move_uci in position.moves_uci[len(self._game_position.moves_uci) :]:
            self._game.apply_move(move_uci)
        self._game_position = position

    def _start_search(self, time_limit_seconds: int) -> None:
        self._stop_search(wait=True)
        if self._board.legal_moves.count() == 0:
            self._write('bestmove 0000')
            return
        if self._game is None:
            self._game = self._engine.new_game(self._position.starting_fen, self._position.moves_uci)
            self._game_position = self._position

        stop_event = Event()
        request = SearchRequest(self._search_mode, time_limit_seconds, stop_event)
        game = self._game
        board = self._board.copy(stack=True)
        self._stop_event = stop_event
        self._search_thread = Thread(
            target=self._search,
            args=(game, board, request),
            name='uci-search',
            daemon=True,
        )
        self._search_thread.start()

    def _search(self, game: InteractiveGame, board: chess.Board, request: SearchRequest) -> None:
        try:
            result = game.analyze(request)
            ordered_candidates_uci = tuple(candidate.move_uci for candidate in result.candidates)
            selection = select_legal_analysis_move(
                board,
                result.chosen_move_uci,
                ordered_candidates_uci,
            )
            if selection.candidate_rank != 0:
                self._diagnostic('Engine chose an illegal move; using the highest-ranked legal candidate.')
            self._write(f'bestmove {selection.move.uci()}')
        except Exception as error:
            legal_moves = sorted(board.legal_moves, key=lambda legal_move: legal_move.uci())
            self._diagnostic(f'Analysis failed: {error}')
            self._write(f'bestmove {legal_moves[0].uci()}' if legal_moves else 'bestmove 0000')

    def _stop_search(self, wait: bool) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if wait and self._search_thread is not None:
            self._search_thread.join()
        if wait or self._search_thread is None or not self._search_thread.is_alive():
            self._search_thread = None
            self._stop_event = None

    def _write(self, line: str) -> None:
        with self._output_lock:
            print(line, file=self._standard_output, flush=True)

    def _diagnostic(self, line: str) -> None:
        print(line, file=self._standard_error, flush=True)
