import chess


def human_player(board: chess.Board):
    while True:
        move = input("Input Your Move:")
        if move == "q":
            raise KeyboardInterrupt
        try:
            board.push_san(move)
            break
        except Exception as e:
            print(e)
