import math
import pygame
from dataclasses import dataclass


@dataclass
class Events:
    clicked: bool
    quit: bool
    left: bool
    right: bool


class HexGridGameGUI:
    """
    Pointy-top hexagonal grid in an (rows × cols) rhombus.
    row → axial r, col → axial q
    """

    def __init__(self, rows: int, cols: int, hex_size: int = 50, title: str = 'Hex Board', light='white', dark='gray'):
        self.rows = rows
        self.cols = cols
        self.size = hex_size  # distance from centre to any corner
        self.light_color = pygame.Color(light)
        self.dark_color = pygame.Color(dark)

        # ── pre-compute pixel centres ───────────────────────────────────────
        self._centres: dict[tuple[int, int], tuple[float, float]] = {}
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for r in range(rows):
            for q in range(cols):
                x, y = self.axial_to_pixel(q, r)
                self._centres[(r, q)] = (x, y)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)

        margin = hex_size  # room for full hex at edges
        self.offset = (margin - min_x, margin - min_y)
        width = int(max_x - min_x + 2 * margin)
        height = int(max_y - min_y + 2 * margin)

        # ── pygame init ────────────────────────────────────────────────────
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)

        # cache regular 6-gon corner vectors (pointy-top, first corner @ 30°)
        ang = [math.radians(30 + 60 * i) for i in range(6)]
        self._corner_vec = [(math.cos(a), math.sin(a)) for a in ang]

    # ╭──────────────────── coordinate converters ─────────────────────╮ #
    def axial_to_pixel(self, q: int, r: int) -> tuple[float, float]:
        """Centre of hex (q,r) in pixels (pointy-top)."""
        x = self.size * math.sqrt(3) * (q + r / 2)
        y = self.size * 1.5 * r
        return x, y

    def pixel_to_axial(self, x: float, y: float) -> tuple[int, int] | None:
        """Reverse mapping + hex rounding (Red Blob algorithm)."""
        x -= self.offset[0]
        y -= self.offset[1]
        q = (math.sqrt(3) / 3 * x - 1 / 3 * y) / self.size
        r = (2 / 3 * y) / self.size
        return self._hex_round(q, r)

    # helpers
    def _hex_round(self, qf: float, rf: float) -> tuple[int, int] | None:
        sf = -qf - rf
        q, r, s = round(qf), round(rf), round(sf)
        if abs(q - qf) > abs(r - rf) and abs(q - qf) > abs(s - sf):
            q = -r - s
        elif abs(r - rf) > abs(s - sf):
            r = -q - s
        if 0 <= r < self.rows and 0 <= q < self.cols:
            return r, q  # our indexing is (row=r, col=q)
        return None

    # ╰─────────────────────────────────────────────────────────────────╯ #

    # ╭──────────────────── basic drawing utilities ───────────────────╮ #
    def _hex_corners(self, centre: tuple[float, float]) -> list[tuple[float, float]]:
        cx, cy = centre
        return [(cx + self.size * vx, cy + self.size * vy) for vx, vy in self._corner_vec]

    def draw_hex(self, centre: tuple[float, float], color: pygame.Color | str):
        pygame.draw.polygon(self.screen, color, self._hex_corners(centre))

    def draw_hex_cell(self, row: int, col: int, color: pygame.Color | str):
        """Draw a hex cell at (row, col) with the given color."""
        cx, cy = self._centres[(row, col)]
        cx, cy = cx + self.offset[0], cy + self.offset[1]
        self.draw_hex((cx, cy), color)

    def draw_background(self):
        for (r, q), centre in self._centres.items():
            color = self.light_color if (r + q) % 2 == 0 else self.dark_color
            shifted = (centre[0] + self.offset[0], centre[1] + self.offset[1])
            self.draw_hex(shifted, color)

    # pieces / text use the pre-computed centres
    def draw_circle(self, row: int, col: int, color: str):
        cx, cy = self._centres[(row, col)]
        cx, cy = cx + self.offset[0], cy + self.offset[1]
        pygame.draw.circle(self.screen, pygame.Color(color), (int(cx), int(cy)), int(self.size * 0.55))

    def draw_text(self, row: int, col: int, text: str, color: str = 'black', font_size: int = 32):
        cx, cy = self._centres[(row, col)]
        cx, cy = cx + self.offset[0], cy + self.offset[1]
        font = pygame.font.Font(None, font_size)
        surf = font.render(text, True, pygame.Color(color))
        rect = surf.get_rect(center=(cx, cy))
        self.screen.blit(surf, rect)

    def highlight_cell(self, row: int, col: int, color: str = 'yellow'):
        tmp = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        overlay = pygame.Color(color)
        overlay.a = 90
        pygame.draw.polygon(
            tmp, overlay, [(self.size + self.size * vx, self.size + self.size * vy) for vx, vy in self._corner_vec]
        )
        cx, cy = self._centres[(row, col)]
        self.screen.blit(tmp, (cx + self.offset[0] - self.size, cy + self.offset[1] - self.size))

    # ╰─────────────────────────────────────────────────────────────────╯ #

    # ╭──────────────────── event helpers ──────────────────────────────╮ #
    def events_occurred(self) -> Events:
        ev = Events(False, False, False, False)
        for e in pygame.event.get():
            if e.type == pygame.MOUSEBUTTONDOWN:
                ev.clicked = True
            if e.type == pygame.QUIT:
                pygame.quit()
                ev.quit = True
            if e.type == pygame.KEYUP:
                if e.key == pygame.K_LEFT:
                    ev.left = True
                if e.key == pygame.K_RIGHT:
                    ev.right = True
        return ev

    def get_cell_from_click(self) -> tuple[int, int] | None:
        x, y = pygame.mouse.get_pos()
        return self.pixel_to_axial(x, y)

    def update_display(self):
        pygame.display.flip()

    def clear_highlights_and_redraw(self, draw_pieces_func):
        # Redraw the board and pieces:
        self.draw_background()
        draw_pieces_func()  # game-specific function to draw pieces
        self.update_display()

    # ╰─────────────────────────────────────────────────────────────────╯ #
