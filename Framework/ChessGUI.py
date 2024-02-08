import pygame

from Framework import Board, SQUARES, square, Square

SCREEN_SIZE = 480
SQUARE_SIZE = SCREEN_SIZE // 8

RESOURCE_FOLDER = 'Framework/Resources/'


class ChessGUI:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption('Chess')
        pygame.display.set_icon(_load_resource('Icon.png'))
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.colors = [pygame.Color('white'), pygame.Color('gray')]

        self.piece_images = {
            'P': _load_resource('WPawn.png'),
            'N': _load_resource('WKnight.png'),
            'B': _load_resource('WBishop.png'),
            'R': _load_resource('WRook.png'),
            'Q': _load_resource('WQueen.png'),
            'K': _load_resource('WKing.png'),
            'p': _load_resource('BPawn.png'),
            'n': _load_resource('BKnight.png'),
            'b': _load_resource('BBishop.png'),
            'r': _load_resource('BRook.png'),
            'q': _load_resource('BQueen.png'),
            'k': _load_resource('BKing.png'),
        }
        # Scale the piece images to the correct size (0.08)
        for piece, img in self.piece_images.items():
            self.piece_images[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))

    def draw_board(self, board: Board) -> None:
        """Draws the chessboard and pieces."""
        for r in range(8):
            for c in range(8):
                color = self.colors[((r + c) % 2)]
                pygame.draw.rect(
                    self.screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )

        # Draw the pieces
        for sq in SQUARES:
            piece = board.piece_at(sq)
            if piece:
                img = self.piece_images[piece.symbol()]
                self.screen.blit(img, ((sq % 8) * SQUARE_SIZE, (7 - sq // 8) * SQUARE_SIZE))

        pygame.display.flip()

    def get_square_from_click(self) -> Square:
        """Translates a mouse click position to a chess square."""
        pos = pygame.mouse.get_pos()
        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE
        return square(col, 7 - row)  # Convert to chess square index

    def highlight_square(self, square: Square, color: str) -> None:
        """Highlights a square on the board."""
        img = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        img.set_alpha(100)
        img.fill(pygame.Color(color))

        self.screen.blit(img, ((square % 8) * SQUARE_SIZE, (7 - square // 8) * SQUARE_SIZE))

        pygame.display.flip()


def _load_resource(path: str) -> pygame.Surface:
    """Loads a resource from the resource folder."""
    return pygame.image.load(RESOURCE_FOLDER + path)
