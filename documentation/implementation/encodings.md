# Chess Encoding for Neural Networks

This document provides an overview of a chess encoding scheme designed for neural network applications. It includes methods for board state encoding, move encoding, and handling moves with associated probabilities.

## Board Encoding

The `encodeBoard` function translates a chess board into a 13x8x8 tensor representation. Each of the first dimension's 12 layers corresponds to one of the 12 distinct chess piece types, split evenly between each color. A value of 1 in the tensor indicates the presence of a piece type at that cell, and 0 indicates absence. The encoding orients the current player's pieces at the bottom and the opponent's at the top, regardless of the actual game orientation. In this way, the neural network can learn to differentiate between the two players' pieces without needing to consider the current player's color. In addition, the 13th layer represents available castling rights of the current position for both players. This encoding scheme is designed to provide a comprehensive representation of the board state, enabling the neural network to learn the game's intricacies effectively.

## Move Encoding

`encodeMove` compresses a chess move into an index within a predefined mapping, facilitating a compact representation in a 1968-dimensional vector space. This approach significantly reduces the dimensionality required to represent chess moves. The mapping is constructed by iterating through all squares on the board and generating all possible moves for each piece type. The resulting indices are then used to encode moves. This encoding scheme is a lot more efficient than the traditional approach of using a 64x64 matrix to represent all possible moves as also used in AlphaZero. This reduces the dimensionality of the move space by a factor of 2 which means that the neural network has to learn a lot less parameters to predict the best move.

## Moves Encoding with Probabilities

The function `encodeMoves` encodes a list of moves along with their corresponding probabilities into a 1D tensor. This tensor represents the probability of each move, providing a detailed landscape of potential moves for a given board state. Moves are normalized to form a probability distribution, aiding in the interpretation by neural network models.

## Data Augmentation

To enhance training data diversity and volume, we generate two variations of each board state:

1. **Normal**: The original board orientation.
2. **Flipped around column d/e**: Mirrors the board horizontally.

While these augmentations do not fully account for chess's non-symmetric aspects, such as the first-move advantage and differences between the king and queen sides, they aim to effectively expand the dataset. By quadrupling the self-play data through these transformations, we hope the model can still learn effectively without the need for as many self-play games.

In addition two more augmentation variations can be imagined:

1. **Flipped around row 4/5**: Mirrors the board vertically. Scores are negated as the position benefits the other player, with the current player still positioned at the bottom.
2. **Flipped around both column d/e and row 4/5**: Combines the above flips and also negates scores.

These augmentations are wrong though, since the moves would then be based on the wrong perspective.

## Sample Data Input

**Normal Board**:

```text
⭘ ⭘ ♖ ⭘ ⭘ ⭘ ♔ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ♙ ♙ ♙
⭘ ⭘ ⭘ ⭘ ♙ ♘ ⭘ ⭘
♙ ⭘ ♖ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ♛ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ♟ ♞ ⭘ ⭘
⭘ ♟ ⭘ ⭘ ⭘ ♟ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ♚ ⭘

Policy: 2 moves
h2h3
g2g3
```

**Flipped around column d/e**:

```text
⭘ ♔ ⭘ ⭘ ⭘ ♖ ⭘ ⭘
♙ ♙ ♙ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ♘ ♙ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ♖ ⭘ ♙
⭘ ⭘ ⭘ ⭘ ♛ ⭘ ⭘ ⭘
⭘ ⭘ ♞ ♟ ⭘ ⭘ ⭘ ⭘
♟ ♟ ♟ ⭘ ⭘ ⭘ ♟ ⭘
⭘ ♚ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘

Policy: 2 moves
a2a3
b2b3
```
