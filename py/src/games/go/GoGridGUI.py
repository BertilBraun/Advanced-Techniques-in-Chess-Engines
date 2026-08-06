from __future__ import annotations

import pygame

from src.eval.GridGUI import BaseGridGameGUI


class GoGridGUI(BaseGridGameGUI):
    def __init__(self, board_size: int, cell_size: int = 72) -> None:
        super().__init__(board_size, board_size, cell_size, title='Native Go Inspector', checkered=False)

    def draw_background(self) -> None:
        self.screen.fill(pygame.Color('#d7a85b'))
        first = self.cell_size // 2
        last = (self.rows - 1) * self.cell_size + first
        for coordinate in range(self.rows):
            center = coordinate * self.cell_size + first
            pygame.draw.line(self.screen, pygame.Color('#2b2118'), (first, center), (last, center), 2)
            pygame.draw.line(self.screen, pygame.Color('#2b2118'), (center, first), (center, last), 2)
        for row, column in self._star_points():
            pygame.draw.circle(
                self.screen,
                pygame.Color('#2b2118'),
                (column * self.cell_size + first, row * self.cell_size + first),
                max(3, self.cell_size // 16),
            )

    def _star_points(self) -> tuple[tuple[int, int], ...]:
        if self.rows == 9:
            return ((2, 2), (2, 6), (4, 4), (6, 2), (6, 6))
        return ((2, 2), (2, 4), (3, 3), (4, 2), (4, 4))
