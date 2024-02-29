import chess

while True:
    while True:
        fen = input("Enter FEN: ")
        if fen.startswith("FEN: "):
            fen = fen[4:]
            break

    while True:
        policy = input("Enter Policy: ")
        if policy.startswith("Policy: "):
            break

    while True:
        move1, score = input("Enter move: ").split(maxsplit=1)
        if move1 == "Board":
            break
        # move2, score = input("Enter move: ").split(maxsplit=1)
        # if move2 == "Board":
        #     break
        
        move1 = chess.Move.from_uci(move1)
        #move2 = chess.Move.from_uci(move2)
        
        board1 = chess.Board(fen)
        board2 = chess.Board(fen.replace(' w ', ' b '))
        
        move1_matches = move1 in board1.legal_moves or move1 in board2.legal_moves
        #move2_matches = move2 in board1.legal_moves or move2 in board2.legal_moves
        
        if move1 in board1.legal_moves and move1 in board2.legal_moves:
            print("Warning: Both moves are legal")                
        if move1 in board1.legal_moves or move1 in board2.legal_moves:
            print("Legal move", (1, move1.uci()) if move1 in board1.legal_moves else (2, move1.uci()))
        else:
            print("Illegal move (move not in legal moves)", move1.uci(), "and", move1.uci())
