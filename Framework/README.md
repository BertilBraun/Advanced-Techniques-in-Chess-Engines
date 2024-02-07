# Chess Bot Framework

## Overview

The Chess Bot Framework is a Python toolkit for creating and testing chess bots. It includes a game manager for single matches, a tournament orchestrator, and example bots as templates for developing new bots.

## Getting Started

Run the example via `python -m Framework` to see the framework in action.

## Usage

Two main functions demonstrate the framework's capabilities:

- `example_tournament()`: Showcases how to set up and run a tournament with multiple bots.
- `example_match()`: Demonstrates a single match between two bots.

To perform a chess match or tournament, follow these steps:

1. Import `GameManager` and `Tournament` from the framework.
2. Import or create bots (e.g., `HumanPlayer` and `RandomBot` from `ExampleBots`).
3. Initialize the game manager or tournament with the desired competitors.
4. Run the game or tournament.

## Creating a New Bot

To create a new bot, extend the `ChessBot` class from `ChessBot.py` and implement the `think` method. Here's a template:

```python
from Framework import ChessBot, Board, Move

class YourCustomBot(ChessBot):
    def __init__(self) -> None:
        """Initializes your custom bot."""
        super().__init__("YourCustomBotName")
    
    def think(self, board: Board) -> Move:
        """Implement your bot's logic to select a move."""
        # Your code here to determine the best move
        pass
```

Replace `"YourCustomBotName"` with your bot's name and implement your move selection logic in the `think` method.

## Example Bots

- `HumanPlayer`: Allows a human player to input moves manually via a GUI interface based on pygame.
- `RandomBot`: Chooses moves randomly from the set of legal moves.
