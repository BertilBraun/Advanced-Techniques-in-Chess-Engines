from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class OracleStone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class OraclePlayer(IntEnum):
    BLACK = 1
    WHITE = 2


class OracleTermination(IntEnum):
    ONGOING = 0
    TWO_PASSES = 1
    SAFETY_PLY_CAP = 2


@dataclass(frozen=True)
class OracleRules:
    board_size: int
    komi_half_points: int
    safety_ply_cap: int
    history_length: int

    def __post_init__(self) -> None:
        if self.board_size < 3:
            raise ValueError('Go board size must be at least 3')
        if self.board_size * self.board_size >= 2**31 - 1:
            raise ValueError('Go board area and pass action must fit in int32')
        if self.safety_ply_cap < self.board_size * self.board_size:
            raise ValueError('Go safety ply cap must be at least the board area')
        if self.history_length < 1:
            raise ValueError('Go history length must be positive')


@dataclass(frozen=True)
class OracleScore:
    black_twice: int
    white_twice: int

    @property
    def winner(self) -> OraclePlayer | None:
        if self.black_twice > self.white_twice:
            return OraclePlayer.BLACK
        if self.white_twice > self.black_twice:
            return OraclePlayer.WHITE
        return None


class OracleGoState:
    def __init__(self, rules: OracleRules) -> None:
        self.rules = rules
        self.board = [OracleStone.EMPTY] * (rules.board_size * rules.board_size)
        self.current_player = OraclePlayer.BLACK
        self.ply = 0
        self.consecutive_passes = 0
        self.termination = OracleTermination.ONGOING
        self.position_history: list[tuple[OracleStone, ...]] = [tuple(self.board)]

    @property
    def pass_action(self) -> int:
        return self.rules.board_size * self.rules.board_size

    @property
    def action_count(self) -> int:
        return self.pass_action + 1

    def _neighbors(self, point: int) -> tuple[int, ...]:
        size = self.rules.board_size
        row, column = divmod(point, size)
        neighbors: list[int] = []
        if row > 0:
            neighbors.append(point - size)
        if column > 0:
            neighbors.append(point - 1)
        if column + 1 < size:
            neighbors.append(point + 1)
        if row + 1 < size:
            neighbors.append(point + size)
        return tuple(neighbors)

    def _group(self, board: list[OracleStone], origin: int) -> set[int]:
        color = board[origin]
        group: set[int] = set()
        pending = [origin]
        while pending:
            point = pending.pop()
            if point in group:
                continue
            group.add(point)
            pending.extend(
                neighbor for neighbor in self._neighbors(point) if board[neighbor] == color and neighbor not in group
            )
        return group

    def _has_liberty(self, board: list[OracleStone], group: set[int]) -> bool:
        return any(board[neighbor] == OracleStone.EMPTY for point in group for neighbor in self._neighbors(point))

    def _placement(self, action: int) -> list[OracleStone]:
        board = self.board.copy()
        own = OracleStone(self.current_player)
        opponent = OracleStone.WHITE if own == OracleStone.BLACK else OracleStone.BLACK
        board[action] = own
        for neighbor in self._neighbors(action):
            if board[neighbor] != opponent:
                continue
            group = self._group(board, neighbor)
            if not self._has_liberty(board, group):
                for point in group:
                    board[point] = OracleStone.EMPTY
        if not self._has_liberty(board, self._group(board, action)):
            raise ValueError('Go placement is suicide')
        return board

    def is_legal(self, action: int) -> bool:
        if self.termination != OracleTermination.ONGOING:
            return False
        if action < 0 or action >= self.action_count:
            return False
        if action == self.pass_action:
            return True
        if self.board[action] != OracleStone.EMPTY:
            return False
        try:
            board = self._placement(action)
        except ValueError:
            return False
        return tuple(board) not in self.position_history

    def legal_actions(self) -> list[int]:
        return [action for action in range(self.action_count) if self.is_legal(action)]

    def apply(self, action: int) -> None:
        if not self.is_legal(action):
            raise ValueError('Illegal Go action')
        if action == self.pass_action:
            self.consecutive_passes += 1
        else:
            self.board = self._placement(action)
            self.consecutive_passes = 0
        self.ply += 1
        self.current_player = OraclePlayer.WHITE if self.current_player == OraclePlayer.BLACK else OraclePlayer.BLACK
        self.position_history.append(tuple(self.board))
        if self.consecutive_passes == 2:
            self.termination = OracleTermination.TWO_PASSES
        elif self.ply >= self.rules.safety_ply_cap:
            self.termination = OracleTermination.SAFETY_PLY_CAP

    def area_score(self) -> OracleScore:
        black_area = sum(stone == OracleStone.BLACK for stone in self.board)
        white_area = sum(stone == OracleStone.WHITE for stone in self.board)
        seen: set[int] = set()
        for origin, stone in enumerate(self.board):
            if stone != OracleStone.EMPTY or origin in seen:
                continue
            region: set[int] = set()
            borders: set[OracleStone] = set()
            pending = [origin]
            while pending:
                point = pending.pop()
                if point in region:
                    continue
                region.add(point)
                seen.add(point)
                for neighbor in self._neighbors(point):
                    neighbor_stone = self.board[neighbor]
                    if neighbor_stone == OracleStone.EMPTY:
                        if neighbor not in region:
                            pending.append(neighbor)
                    else:
                        borders.add(neighbor_stone)
            if borders == {OracleStone.BLACK}:
                black_area += len(region)
            elif borders == {OracleStone.WHITE}:
                white_area += len(region)
        return OracleScore(
            black_twice=black_area * 2,
            white_twice=white_area * 2 + self.rules.komi_half_points,
        )

    def canonical_encoding(self) -> tuple[int, int, tuple[int, ...]]:
        size = self.rules.board_size
        plane_count = self.rules.history_length * 2 + 1
        plane_size = size * size
        values = [0] * (plane_count * plane_size)
        own = OracleStone(self.current_player)
        opponent = OracleStone.WHITE if own == OracleStone.BLACK else OracleStone.BLACK
        for offset, board in enumerate(reversed(self.position_history)):
            if offset >= self.rules.history_length:
                break
            for point, stone in enumerate(board):
                if stone == own:
                    values[offset * 2 * plane_size + point] = 1
                elif stone == opponent:
                    values[(offset * 2 + 1) * plane_size + point] = 1
        if self.current_player == OraclePlayer.BLACK:
            values[(plane_count - 1) * plane_size :] = [1] * plane_size
        return plane_count, size, tuple(values)
