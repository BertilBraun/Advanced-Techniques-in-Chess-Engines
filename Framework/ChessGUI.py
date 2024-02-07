import pygame
import chess

SCREEN_SIZE = 480
SQUARE_SIZE = SCREEN_SIZE // 8

class ChessGUI:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Chess")
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.colors = [pygame.Color("white"), pygame.Color("gray")]

    def draw_board(self, board: chess.Board) -> None:
        """Draws the chessboard and pieces."""
        for r in range(8):
            for c in range(8):
                color = self.colors[((r + c) % 2)]
                pygame.draw.rect(self.screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

        # Draw the pieces
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                font = pygame.font.SysFont(pygame.font.get_default_font(), SQUARE_SIZE)
                img = font.render(piece.symbol(), True, pygame.Color("Black")) # TODO - Use images for pieces
                self.screen.blit(img, ((sq % 8) * SQUARE_SIZE, (7 - sq // 8) * SQUARE_SIZE))

        pygame.display.flip()

    def get_square_from_click(self) -> chess.Square:
        """Translates a mouse click position to a chess square."""
        pos = pygame.mouse.get_pos()
        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE
        return chess.square(col, 7 - row)  # Convert to chess square index

    def highlight_square(self, square: chess.Square, color: str) -> None:
        """Highlights a square on the board."""
        img = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        img.set_alpha(100)
        img.fill(pygame.Color(color))
        
        self.screen.blit(img, ((square % 8) * SQUARE_SIZE, (7 - square // 8) * SQUARE_SIZE))
        
        pygame.display.flip()
        