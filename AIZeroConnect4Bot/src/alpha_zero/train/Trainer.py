import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.Network import Network
from src.settings import TORCH_DTYPE, USE_GPU, log_scalar
from src.alpha_zero.train.TrainingArgs import TrainingParams
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.util.log import log

# DONE AlphaZero simply maintains a single neural network that is updated continually, rather than waiting for an iteration to complete
# DONE do so, save model after each epoch, use smaller num_parallel_games
# DONE save deduplicated dataset for the previous iteration
# DONE simply set num_epochs to 1 and increase the num_iterations, while decreasing how fast the window size grows but increasing the base and max window size
# DONE game and inference nodes? So that 100% GPU is used on inference nodes and as many nodes as needed can supply the self-play nodes
# DONE current MCTS -> BatchedMCTS and create new ClientServerMCTS - based on what protocol?
# DONE make sure to instantly update the inference servers once a new model is available
# DONE infrence server caching
# DONE make sure, that each process logs their CPU and RAM usage and the root logs usage for all GPUs - Display all data and averages in visualization


# DONE something between Deduplication and Training Batches takes 10min?!
# DONE [17.03.24] [INFO] Exception in Training: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

# DONE /memory_19_deduplicated.pt (deflated 96%) Why is the memory able to be compressed so much? What about the memory is so compressible? Yeah.. just a lot of either 0 or 1 values.. Does it make sense to store it as a bit array?
# DONE with the encoded boards, inter process comunication of the states should be way faster and more efficient
# DONE optimize encode and decode with numpy
# DONE load and save via numpy not torch

# NOT_REQUIRED Not always reload each iteration, but only if the memory is not already loaded - only takes a second to load the memory
# DONE single GPU for training, multiple GPUs for self-play?

# TODO use both the final game result as well as the MCTS root value_sum / num_visits as the value target for the NN training. Averaging f.e.

# DONE reduce board size to 6x7

# TODO parallel MCTS search? - searching multiple states at once by blocking nodes: https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf

# TODO Int8 for inference. In that case the trainer and self play nodes need different models and after training the model needs to be quantized but apparently up to 4x faster inference. https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization


class Trainer:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingParams,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def train(self, dataset: SelfPlayDataset, iteration: int) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [len(dataset) - self.args.batch_size, self.args.batch_size]
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=USE_GPU,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        train_stats = TrainingStats()
        base_lr = self.args.learning_rate(iteration)
        log_scalar('learning_rate', base_lr, iteration)

        self.model.train()

        def calculate_loss_for_batch(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state, policy_targets, value_targets = batch

            state = state.to(dtype=TORCH_DTYPE, device=self.model.device, non_blocking=True)
            policy_targets = policy_targets.to(dtype=TORCH_DTYPE, device=self.model.device, non_blocking=True)
            value_targets = value_targets.to(dtype=TORCH_DTYPE, device=self.model.device, non_blocking=True)
            value_targets = value_targets.unsqueeze(1)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            return policy_loss, value_loss, loss

        total_policy_loss = torch.tensor(0.0, device=self.model.device)
        total_value_loss = torch.tensor(0.0, device=self.model.device)
        total_loss = torch.tensor(0.0, device=self.model.device)

        for batchIdx, batch in tqdm(enumerate(train_dataloader), desc='Training batches', total=len(train_dataloader)):
            policy_loss, value_loss, loss = calculate_loss_for_batch(batch)

            # Update learning rate before stepping the optimizer
            batch_percentage = batchIdx / len(train_dataloader)
            lr = self.args.learning_rate_scheduler(batch_percentage, base_lr)
            log_scalar(f'learning_rate/iteration_{iteration}', lr, batchIdx)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss

        train_stats.update(total_policy_loss.item(), total_value_loss.item(), total_loss.item(), len(train_dataloader))

        with torch.no_grad():
            validation_stats = TrainingStats()
            for i, validation_batch in enumerate(validation_dataloader):
                val_policy_loss, val_value_loss, val_loss = calculate_loss_for_batch(validation_batch)

                log_scalar('val_policy_loss', val_policy_loss.item(), i + (iteration * len(validation_dataloader)))
                log_scalar('val_value_loss', val_value_loss.item(), i + (iteration * len(validation_dataloader)))
                log_scalar('val_loss', val_loss.item(), i + (iteration * len(validation_dataloader)))
                validation_stats.update(val_policy_loss.item(), val_value_loss.item(), val_loss.item())

            log(f'Validation stats: {validation_stats}')

        return train_stats
