from __future__ import annotations

from AlphaZeroCpp import GoPlayer, GoPosition7, GoPosition9, GoRules

from src.games.Board import Board, Player


NativeGoPosition = GoPosition7 | GoPosition9


class NativeGoBoard(Board[int]):
    """Mutable visualization adapter around an immutable native Go position."""

    def __init__(
        self,
        board_size: int = 7,
        komi_half_points: int = 15,
        maximum_moves: int | None = None,
    ) -> None:
        if board_size not in (7, 9):
            raise ValueError('Native Go visualization supports only 7x7 and 9x9 boards')
        move_limit = maximum_moves if maximum_moves is not None else board_size * board_size * 4
        rules = GoRules(komi_half_points, move_limit)
        self._position: NativeGoPosition = GoPosition7(rules) if board_size == 7 else GoPosition9(rules)

    @classmethod
    def from_position(cls, position: NativeGoPosition) -> NativeGoBoard:
        board = cls.__new__(cls)
        board._position = position
        return board

    @property
    def board_dimensions(self) -> tuple[int, int]:
        return (self._position.board_size, self._position.board_size)

    @property
    def board_size(self) -> int:
        return self._position.board_size

    @property
    def pass_action(self) -> int:
        return self.board_size * self.board_size

    @property
    def current_player(self) -> Player:
        return 1 if self._position.player == GoPlayer.BLACK else -1

    @property
    def move_number(self) -> int:
        return self._position.move_number

    @property
    def consecutive_passes(self) -> int:
        return self._position.consecutive_passes

    @property
    def ko_point(self) -> int | None:
        return self._position.ko_point

    def black_points(self) -> list[int]:
        return self._position.black_points(0)

    def white_points(self) -> list[int]:
        return self._position.white_points(0)

    def make_move(self, move: int) -> None:
        if move not in self.get_valid_moves():
            raise ValueError(f'Illegal Go action: {move}')
        self._position = self._position.child(move)

    def is_game_over(self) -> bool:
        return self._position.is_terminal

    def check_winner(self) -> Player | None:
        winner = self._position.terminal_result().winner
        if winner is None:
            return None
        return 1 if winner == GoPlayer.BLACK else -1

    def get_valid_moves(self) -> list[int]:
        return self._position.legal_actions()

    def copy(self) -> NativeGoBoard:
        return self.from_position(self._position)

    def quick_hash(self) -> int:
        return self._position.state_hash()

    def get_approximate_result_score(self) -> float:
        score = self._position.area_score()
        maximum_half_points = self.board_size * self.board_size * 2 + abs(self._position.rules.komi_half_points)
        return (score.black_half_points - score.white_half_points) / maximum_half_points
