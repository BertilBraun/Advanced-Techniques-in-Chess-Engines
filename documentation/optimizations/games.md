# Game Specific Optimizations

## Symmetric Variations

Most games have symmetric variations, which can be used to reduce the number of states that have to be evaluated and yield more training samples. These symmetries have to ensure, that the orientation is still preserved, i.e. in chess, each encoded board must have the current player on top.

TicTacToe can use 4 rotational symmetries and 2 mirror symmetries, which gives a total of 8 symmetries. This can be used to reduce the number of states that have to be evaluated and yield more training samples.
Connect4 can use a vertical mirror symmetry, all other symmetries are not valid, as they would change the orientation of the board.
Chess does not have any real symmetries, as the orientation of the board is important. However, the board can be mirrored as an approximation, which up to very high levels of play is not distinguishable from the original board. This can be used to reduce the number of states that have to be evaluated and yield more training samples.

## Chess Move Encoding

The most general way to encode a move is to use the start and end square of the move. This yields 64x64=4096 possible moves. However, most of these moves are not legal, as the board state does not allow for them. Therefore, the move encoding can be reduced to the possible moves, which are the legal moves of the board state.

These are all the queen moves (which include the rook, bishop, pawn, and king moves) and the knight moves. The queen moves are 8 directions times the number of squares in that direction (up to the end of the board), 7 in each diagonal, anti-diagonal, row and column, so 28 moves from each square. This yields 28x64=1792 possible moves. The knight moves are up to 8 possible moves from each square unless the knight is on the edge of the board. Including the promotion moves, this yields exactly 1968 possible moves.
