import torch
import numpy as np
from typing import Tuple

from src.Network import Network
from src.eval.AlphaZeroBot import AlphaZeroBot
from src.eval.GameManager import GameManager
from src.eval.RandomPlayer import RandomPlayer
from src.settings import CURRENT_BOARD, CURRENT_GAME


class ModelEvaluation:
    # This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself
    def _predict_batch(self, model: Network, board_states: np.ndarray) -> np.ndarray:
        """
        Predicts the policy and value for a batch of board states using a single model.

        Args:
            model (Network): The neural network model to use for predictions.
            board_states (np.ndarray): Array of board states, shape (batch_size, BOARD_SIZE).

        Returns:
            policy_probs_batch: Array of policy probabilities, shape (batch_size, ACTION_SIZE).
        """
        model.eval()
        with torch.no_grad():
            board_tensor = torch.tensor(board_states, dtype=torch.float32).to(next(model.parameters()).device)
            policy_output, _ = model(board_tensor)
            # Apply sigmoid to policy logits to get probabilities
            policy_probs = torch.sigmoid(policy_output).cpu().numpy()
        return policy_probs

    def _play_batch(
        self,
        player1_model: Network,
        player_neg1_model: Network,
        current_model: Network,
        batch_size: int = 64,
    ) -> Tuple[int, int, int]:
        """
        Play a batch of games between two models and collect statistics.

        Args:
            player1_model (Network): Model assigned to Player 1.
            player_neg1_model (Network): Model assigned to Player -1.
            current_model (Network): The current model being evaluated.
            batch_size (int): Number of games in the batch.

        Returns:
            Tuple[int, int, int]: Counts of (wins, losses, draws) from the batch.
        """
        # Initialize batch of games
        games = [CURRENT_BOARD() for _ in range(batch_size)]

        # Initialize statistics
        win = 0
        loss = 0
        draw = 0

        while games:
            assert all(game.current_player == games[0].current_player for game in games)
            # Gather board states for all games that are not over
            board_states = np.array(
                [CURRENT_GAME.get_canonical_board(game) for game in games]
            )  # Shape: (active_batch_size, BOARD_SIZE)

            # Perform predictions model
            model = player1_model if games[0].current_player == 1 else player_neg1_model
            policies = self._predict_batch(model, board_states)

            # Apply moves to the games
            for game, policy in zip(games, policies):
                # Mask invalid moves
                valid_moves_mask = CURRENT_GAME.encode_moves(game.get_valid_moves())
                policy *= valid_moves_mask

                # If no valid moves left, it's a draw
                if policy.sum() == 0:
                    draw += 1
                    game.make_move(game.get_valid_moves()[0])  # Arbitrary move to trigger game over
                    continue

                # Select the move with the highest probability
                move = CURRENT_GAME.decode_move(policy.argmax())

                # Make the move
                game.make_move(move)

                # Check if the game is over after the move
                if game.is_game_over():
                    winner = game.check_winner()
                    if winner is None:
                        draw += 1
                    else:
                        # Determine if the assigned model won
                        if winner == 1:
                            # Player 1 won
                            if player1_model == current_model:
                                win += 1
                            else:
                                loss += 1
                        elif winner == -1:
                            # Player -1 won
                            if player_neg1_model == current_model:
                                win += 1
                            else:
                                loss += 1

            games = [game for game in games if not game.is_game_over()]

        return win, loss, draw

    def play_vs_random(self, model: Network, num_games: int = 64) -> Tuple[int, int, int]:
        # Random vs Random has a result of: 60% Wins, 28% Losses, 12% Draws
        wins = draws = losses = 0

        az_bot = AlphaZeroBot(model, max_time_to_think=0.02)  # 20ms
        random_bot = RandomPlayer()
        for p1, p2 in ((az_bot, random_bot), (random_bot, az_bot)):
            game_manager = GameManager(p1, p2)
            for _ in range(num_games // 2):
                result = game_manager.play_game()
                if result == 1:
                    if p1 == az_bot:
                        wins += 1
                    else:
                        losses += 1
                elif result == -1:
                    if p2 == az_bot:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1

        return wins, losses, draws

    def play_two_models_search(
        self, current_model: Network, previous_model: Network, num_games: int = 64
    ) -> Tuple[int, int, int]:
        wins = draws = losses = 0

        for model1, model2 in [(current_model, previous_model), (previous_model, current_model)]:
            game_manager = GameManager(
                AlphaZeroBot(model1, max_time_to_think=0.02),  # 40ms
                AlphaZeroBot(model2, max_time_to_think=0.02),  # 40ms
            )
            for _ in range(num_games // 2):
                result = game_manager.play_game()
                if result == 1:
                    if model1 == current_model:
                        wins += 1
                    else:
                        losses += 1
                elif result == -1:
                    if model2 == current_model:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1

        return wins, losses, draws

    def play_two_models_batch(
        self, current_model: Network, previous_model: Network, batch_size: int = 64
    ) -> Tuple[int, int, int]:
        """Play two separate batches of games between two most recent models and collect statistics.
        Returns:
            Tuple[int, int, int]: Aggregated counts of (wins, losses, draws) from both batches.
        """
        # Define the two runs
        runs = [(current_model, previous_model), (previous_model, current_model)]

        total_win = 0
        total_loss = 0
        total_draw = 0

        for run_idx, (player1_model, player_neg1_model) in enumerate(runs, start=1):
            win, loss, draw = self._play_batch(player1_model, player_neg1_model, current_model, batch_size)
            total_win += win
            total_loss += loss
            total_draw += draw

            starting_model = 'Current Model' if run_idx == 1 else 'Previous Model'
            print(f'Batch {run_idx} Results (Starting Model: {starting_model}):')
            print(f'Wins: {win}, Losses: {loss}, Draws: {draw}\n')

        print(f'Total Results after {2 * batch_size} games:')
        print(f'Wins: {total_win}, Losses: {total_loss}, Draws: {total_draw}')

        return total_win, total_loss, total_draw
