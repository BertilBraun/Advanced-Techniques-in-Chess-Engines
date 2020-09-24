import chess
import time

import pyglet
from pyglet import shapes

from Defines import *
  
calls = 0
def timeit(method):
    def timed(*args, **kw):
        ts = time.time_ns()
        result = method(*args, **kw)
        global calls
        calls += 1
        print(method.__name__, (time.time_ns() - ts) / (10 ** 6), "ms", calls)
        return result
    return timed

class Game:
    def __init__(self, board):
        self.board = board

    def userMoveFromTo(self, ox, oy, nx, ny):
        move = chess.Move(ox + oy * 8, nx + ny * 8)
        if move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_game_over():
                # TODO WON Screen
                self.board.reset()

    def botMove(self):

        bestMove = None
        maxVal = float("-inf")

        for move in self.board.generate_legal_moves():
            self.board.push(move)
            val = self.minimax(MAX_DEPTH, float("-inf"), float("inf"), True)
            self.board.pop()

            if val > maxVal:
                maxVal = val
                bestMove = move

        self.board.push(bestMove)

    def minimax(self, depth, alpha, beta, maximizing):
        if depth == 0 or self.board.is_game_over():
            return self.evaluate()

        if maximizing:
            maxVal = float("-inf")
            for move in self.board.generate_legal_moves():
                self.board.push(move)
                val = self.minimax(depth - 1, alpha, beta, False)
                self.board.pop()
                maxVal = max(maxVal, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return maxVal
        else:
            minVal = float("inf")
            for move in self.board.generate_legal_moves():
                self.board.push(move)
                val = self.minimax(depth - 1, alpha, beta, True)
                self.board.pop()
                minVal = min(minVal, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return minVal

    def evaluate(self):

        def evaluatePiece(piece, x, y):
            if piece == None:
                return 0

            symbol = piece.symbol().upper()

            valueArray = pieceValueArrayDict[symbol]
            if piece.color == chess.BLACK:
                valueArray = valueArray[::-1]
        
            value = pieceValueDict[symbol] + valueArray[y][x]

            return value if piece.color == chess.WHITE else -value

        total = 0     
        
        for y in range(8):
            for x in range(8):
                piece = self.board.piece_at(x + y * 8)
                total += evaluatePiece(piece, x, y)

        return total

class GameEventHandler:
    def __init__(self, game):
        self.game = game
        self.selected = None

    def on_key_press(self, symbol, modifiers):
        pass

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        pass # TODO Drag

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            ox, oy = self.selected
            nx, ny = x // 80, y // 80
            if self.game.board.turn == chess.WHITE:
                self.game.userMoveFromTo(ox, oy, nx, ny)

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.selected = (x // 80, y // 80)
            # TODO Draw

class Visualisation:
    def __init__(self, board, game):

        self.window = pyglet.window.Window(8 * 80, 8 * 80, "Chess", False, pyglet.window.Window.WINDOW_STYLE_DIALOG)
        self.board = board
        self.game = game

        @self.window.event
        def on_draw():
            self.window.clear()
            self.draw(board)
          
        self.tiles = []
        darkColor = (209, 139, 71)
        lightColor = (255, 206, 158)

        for x in range(8):
            for y in range(8):
                color = darkColor if (y + x * 8 + x) % 2 == 0 else lightColor
                self.tiles.append(shapes.Rectangle(x * 80, y * 80, 80, 80, color))

        self.imgDict = {
            "P" : pyglet.image.load("Res/WPawn.png"),
            "R" : pyglet.image.load("Res/WRook.png"),
            "N" : pyglet.image.load("Res/WKnight.png"),
            "B" : pyglet.image.load("Res/WBishop.png"),
            "Q" : pyglet.image.load("Res/WQueen.png"),
            "K" : pyglet.image.load("Res/WKing.png"),

            "p" : pyglet.image.load("Res/BPawn.png"),
            "r" : pyglet.image.load("Res/BRook.png"),
            "n" : pyglet.image.load("Res/BKnight.png"),
            "b" : pyglet.image.load("Res/BBishop.png"),
            "q" : pyglet.image.load("Res/BQueen.png"),
            "k" : pyglet.image.load("Res/BKing.png"),            
        }

    def update(self, dt):
        if self.board.turn == chess.BLACK:
            self.game.botMove()

    def run(self):
        self.gameEventHandler = GameEventHandler(self.game)
        self.window.push_handlers(self.gameEventHandler)
        pyglet.clock.schedule_interval(self.update, 1/30.0)
        pyglet.app.run()

    def draw(self, board):
        for tile in self.tiles:
            tile.draw()
            
        for y in range(8):
            for x in range(8):
                piece = board.piece_at(x + y *8)
                if piece != None:
                    img = self.imgDict[piece.symbol()]
                    spr = pyglet.sprite.Sprite(img, x = x * 80, y = y * 80)
                    spr.scale = 0.08
                    spr.draw()
                   
board = chess.Board()
game = Game(board)
vis = Visualisation(board, game)

vis.run()

 