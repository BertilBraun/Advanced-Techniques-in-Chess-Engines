from __future__ import annotations

import pygame

from src.eval.GridGUI import BaseGridGameGUI


class GoGridGUI(BaseGridGameGUI):
    control_height = 64

    def __init__(self, board_size: int, cell_size: int = 72) -> None:
        super().__init__(board_size, board_size, cell_size, title='Native Go Inspector', checkered=False)
        self.board_pixels = board_size * cell_size
        self.screen = pygame.display.set_mode((self.board_pixels, self.board_pixels + self.control_height))
        self.pass_button = pygame.Rect(self.board_pixels - 116, self.board_pixels + 12, 100, 40)

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
        pygame.draw.rect(
            self.screen,
            pygame.Color('#30271f'),
            pygame.Rect(0, self.board_pixels, self.board_pixels, self.control_height),
        )

    def draw_controls(self, legal_placement_count: int, pass_enabled: bool) -> None:
        font = pygame.font.Font(None, 24)
        status = font.render(f'{legal_placement_count} legal placement(s)', True, pygame.Color('white'))
        self.screen.blit(status, (16, self.board_pixels + 22))

        button_color = pygame.Color('#d7a85b') if pass_enabled else pygame.Color('#71675e')
        pygame.draw.rect(self.screen, button_color, self.pass_button, border_radius=6)
        pygame.draw.rect(self.screen, pygame.Color('white'), self.pass_button, 2, border_radius=6)
        label = font.render('Pass (Up)', True, pygame.Color('#1d1712'))
        self.screen.blit(label, label.get_rect(center=self.pass_button.center))

    def is_pass_button_clicked(self) -> bool:
        return self.pass_button.collidepoint(pygame.mouse.get_pos())

    def _star_points(self) -> tuple[tuple[int, int], ...]:
        if self.rows == 9:
            return ((2, 2), (2, 6), (4, 4), (6, 2), (6, 6))
        return ((2, 2), (2, 4), (3, 3), (4, 2), (4, 4))
