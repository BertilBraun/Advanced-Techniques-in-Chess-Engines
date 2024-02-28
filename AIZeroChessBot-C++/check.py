import chess

while True:
    while True:
        fen = input("Enter FEN: ")
        if fen.startswith("FEN: "):
            fen = fen[4:]
            board = chess.Board(fen)
            break
        
    while True:
        policy = input("Enter Policy: ")
        if policy.startswith("Policy: "):
            break
        
    while True:
        move1, score = input("Enter move: ").split(maxsplit=1)
        if move1 == "Board":
            break
        move2, score = input("Enter move: ").split(maxsplit=1)
        if move2 == "Board":
            break
        move1 = chess.Move.from_uci(move1)
        move2 = chess.Move.from_uci(move2)
        if move1 in board.legal_moves or move2 in board.legal_moves:
            print("Legal move", (1, move1.uci()) if move1 in board.legal_moves else (2, move2.uci()))
            pass
        else:
            print("Illegal move (move not in legal moves)", move1.uci(), "and", move2.uci())
        
