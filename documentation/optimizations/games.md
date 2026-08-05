# Game Specific Optimizations

## Symmetric Variations

Chess does not have exact geometric symmetries because board orientation and piece movement are significant. The board can still be mirrored as an approximation to diversify training samples.

## Chess Encoding

The input representation for the chess engine is a multi-layered structure. It consists of 12 bit-boards for each piece type (six for white and six for black), four planes for castling rights, two for piece occupancy (one for all white pieces, one for all black), a single plane to indicate checkers, and six scalar planes representing the material difference. To simplify the learning process for the neural network, the board is always oriented from the perspective of the white player. This means that whenever it's black's turn to move, the board state is flipped before being fed into the network, so it doesn't have to learn the game from two different perspectives.

The policy head of the network maps the board state to an optimized 1814-dimensional policy vector. This vector is specifically designed to be efficient by excluding moves that are impossible under the rules of chess, such as a piece moving from A1 to H6 in a single turn. Furthermore, because the board is always presented from white's point of view, all potential promotion and castling moves for black are removed from this policy vector, further streamlining the output.

To handle games that might otherwise continue indefinitely, specific termination rules are implemented. Games are automatically declared as draws if they exceed 200 moves or if they continue for an extended period with very few pieces left on the board. In these scenarios, instead of assigning a simple draw score of 0.0, a more nuanced evaluation is used. A weighted count of the remaining pieces provides an approximate result score, offering a better training target for the value head. This score is scaled to a range of -0.5 to 0.5 for excessively long games, and to a range of -1.0 to 1.0 for games with few pieces, as the latter situation is more likely to have a clear winner in the near future.
