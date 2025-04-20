from __future__ import annotations
from os import PathLike

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Network import Network
from src.dataset.SelfPlayTrainDataset import SelfPlayTrainDataset
from src.train.TrainingArgs import TrainingArgs
from src.settings import USE_GPU, TORCH_DTYPE
from src.util.save_paths import load_model, model_save_path

import AlphaZeroCpp


@dataclass
class Results:
    wins: int
    losses: int
    draws: int

    @staticmethod
    def from_cpp(wld: tuple[int, int, int]) -> Results:
        return Results(wld[0], wld[1], wld[2])


# The device to use for evaluation. Since Training is done on device 0, we can use device 1 for evaluation
EVAL_DEVICE = 0


class ModelEvaluation:
    """This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself"""

    def __init__(self, iteration: int, args: TrainingArgs, run: int, num_games: int = 64) -> None:
        self.iteration = iteration
        self.num_games = num_games
        self.args = args
        self.run = run

    def evaluate_model_vs_dataset(self, dataset: SelfPlayTrainDataset) -> tuple[float, float, float, float]:
        device = torch.device(f'cuda:{EVAL_DEVICE}' if USE_GPU else 'cpu')
        model = load_model(model_save_path(self.iteration, self.args.save_path), self.args.network, device)

        return self._evaluate_model_vs_dataset(model, dataset.as_dataloader(batch_size=128, num_workers=1))

    @staticmethod
    def _evaluate_model_vs_dataset(model: Network, dataloader: DataLoader) -> tuple[float, float, float, float]:
        model.eval()

        total_top1_correct = 0
        total_top5_correct = 0
        total_top10_correct = 0
        total_policy_total = 0
        total_value_loss = 0.0

        with torch.no_grad():
            for boards, moves_list, outcomes in dataloader:
                boards = boards.to(device=model.device, dtype=TORCH_DTYPE)
                moves_list = moves_list.to(device=model.device, dtype=TORCH_DTYPE)
                outcomes = outcomes.to(device=model.device, dtype=TORCH_DTYPE).unsqueeze(1)

                policy_outputs, value_outputs = model(boards)

                policy_preds = torch.softmax(policy_outputs, dim=1)

                for i in range(len(boards)):
                    top1 = policy_preds[i].topk(1).indices
                    top5 = policy_preds[i].topk(5).indices
                    top10 = policy_preds[i].topk(10).indices
                    true_moves = moves_list[i].nonzero().squeeze()

                    if torch.any(top1 == true_moves):
                        total_top1_correct += 1
                    if torch.any(top5.unsqueeze(1) == true_moves):
                        total_top5_correct += 1
                    if torch.any(top10.unsqueeze(1) == true_moves):
                        total_top10_correct += 1

                    total_policy_total += 1

                total_value_loss += F.mse_loss(value_outputs, outcomes).item()

        top1_accuracy = total_top1_correct / total_policy_total
        top5_accuracy = total_top5_correct / total_policy_total
        top10_accuracy = total_top10_correct / total_policy_total
        avg_value_loss = total_value_loss / len(dataloader)

        return top1_accuracy, top5_accuracy, top10_accuracy, avg_value_loss

    def play_vs_random(self) -> Results:
        # Random vs Random has a result of: 60% Wins, 28% Losses, 12% Draws
        res = AlphaZeroCpp.evaluate_model(self.iteration, self.args.save_path, self.num_games, self.run, 'random')
        return Results.from_cpp(res)

    def play_two_models_search(self, model_path: str | PathLike) -> Results:
        res = AlphaZeroCpp.evaluate_model(self.iteration, self.args.save_path, self.num_games, self.run, model_path)
        return Results.from_cpp(res)
