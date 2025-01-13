import pygame


class BaseGridGameGUI:
    def __init__(self, rows: int, cols: int, cell_size: int = 60, title: str = 'Board Game', checkered: bool = True):
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

    def draw_text(self, row: int, col: int, text: str, color: str):
        font = pygame.font.Font(None, 36)
        text_render = font.render(text, True, pygame.Color(color))
        text_rect = text_render.get_rect(
            center=(
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2,
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
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(
                        c * self.cell_size,
                        r * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

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

    def events_occurred(self) -> tuple[bool, bool]:
        # Poll for events and returns if (clicked, quit) occurred
        clicked = False
        quit = False
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked = True
            if event.type == pygame.QUIT:
                pygame.quit()
                quit = True

        return clicked, quit

    def clear_highlights_and_redraw(self, draw_pieces_func):
        # Redraw the board and pieces:
        self.draw_background()
        draw_pieces_func()  # game-specific function to draw pieces
        self.update_display()
