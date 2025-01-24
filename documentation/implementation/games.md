# Abstract Game and Board Classes

To ensure flexibility and support for multiple games, abstract base classes for `Game` and `Board` are defined. These classes enforce a consistent interface across different game implementations. To implement a new game, one must subclass these abstract classes and provide the necessary methods. Other than that, only hyperparameters and game-specific logic need to be defined, the rest of the pipeline remains the same.

```python
class Game(ABC, Generic[_Move]):
    @property
    @abstractmethod
    def null_move(self) -> _Move:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """The number of possible moves in the game."""
        pass

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, int, int]:
        """(num_channels, height, width)"""
        pass

    @abstractmethod
    def get_canonical_board(self, board: Board[_Move]) -> np.ndarray:
        """Returns a canonical representation of the board from the perspective of the current player.
        No matter the current player, the board should always be from the perspective as if the player to move is 1.
        The board should be a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property."""
        pass

    @abstractmethod
    def encode_move(self, move: _Move) -> int:
        """Encodes a move into the index of the action in the policy vector."""
        pass

    @abstractmethod
    def decode_move(self, move: int) -> _Move:
        """Decodes an action index into a move."""
        pass

    @abstractmethod
    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns a list of symmetric variations of the board and the corresponding action probabilities.
        The board is a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property.
        The action_probabilities is a numpy array with shape (action_size)."""
        pass

    @abstractmethod
    def get_initial_board(self) -> Board[_Move]:
        pass

class Board(ABC, Generic[_Move]):
    def __init__(self) -> None:
        self.current_player: Player = 1

    @abstractmethod
    def make_move(self, move: _Move) -> None:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abstractmethod
    def check_winner(self) -> Optional[Player]:
        pass

    @abstractmethod
    def get_valid_moves(self) -> list[_Move]:
        pass

    @abstractmethod
    def copy(self) -> Board[_Move]:
        pass

    @abstractmethod
    def quick_hash(self) -> int:
        pass
```
