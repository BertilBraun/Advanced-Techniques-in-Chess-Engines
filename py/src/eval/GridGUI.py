import src.environ_setup  # noqa: F401
import pygame
from dataclasses import dataclass


@dataclass
class Events:
    clicked: bool
    quit: bool
    left: bool
    right: bool


class BaseGridGameGUI:
    def __init__(self, rows: int, cols: int, cell_size: int = 100, title: str = 'Board Game', checkered: bool = True):
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        width = cols * cell_size
        height = rows * cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)

        # Colors could be customizable or default
        self.light_color = pygame.Color('white')
        self.dark_color = pygame.Color('gray')
        self.checkered = checkered

    def draw_cell(self, row: int, col: int, color: str | pygame.Color | tuple[int, int, int]):
        pygame.draw.rect(
            self.screen,
            pygame.Color(color),
            pygame.Rect(
                col * self.cell_size,
                row * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

    def draw_circle(self, row: int, col: int, color: str):
        pygame.draw.circle(
            self.screen,
            pygame.Color(color),
            (
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2,
            ),
            self.cell_size // 3,
        )

    def draw_text(
        self, row: int, col: int, text: str, color: str = 'black', offset: tuple[int, int] = (0, 0), font_size: int = 16
    ):
        font = pygame.font.Font('src/eval/Segoe-UI-Symbol.ttf', font_size)
        text_render = font.render(text, True, pygame.Color(color))
        text_rect = text_render.get_rect(
            center=(
                col * self.cell_size + self.cell_size // 2 + offset[0],
                row * self.cell_size + self.cell_size // 2 + offset[1],
            )
        )
        self.screen.blit(text_render, text_rect)

    def draw_background(self):
        # Default checkerboard pattern
        for r in range(self.rows):
            for c in range(self.cols):
                color = (
                    self.light_color
                    if ((r + c) % 2 == 0 and self.checkered) or (c % 2 == 0 and not self.checkered)
                    else self.dark_color
                )
                self.draw_cell(r, c, color)

    def highlight_cell(self, row: int, col: int, color: str):
        highlight_surf = pygame.Surface((self.cell_size, self.cell_size))
        highlight_surf.set_alpha(100)
        highlight_surf.fill(pygame.Color(color))
        self.screen.blit(highlight_surf, (col * self.cell_size, row * self.cell_size))

    def get_cell_from_click(self) -> tuple[int, int] | None:
        pos = pygame.mouse.get_pos()
        col = pos[0] // self.cell_size
        row = pos[1] // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    def update_display(self):
        pygame.display.flip()

    def update_window_title(self, title: str):
        pygame.display.set_caption(title)

    def events_occurred(self) -> Events:
        # Poll for events
        events = Events(False, False, False, False)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                events.clicked = True
            if event.type == pygame.QUIT:
                pygame.quit()
                events.quit = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    events.left = True
                if event.key == pygame.K_RIGHT:
                    events.right = True

        return events

    def clear_highlights_and_redraw(self, draw_pieces_func):
        # Redraw the board and pieces:
        self.draw_background()
        draw_pieces_func()  # game-specific function to draw pieces
        self.update_display()
