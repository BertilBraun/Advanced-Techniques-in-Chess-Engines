import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.Network import Network
from src.settings import TORCH_DTYPE, log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import log


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

    def train(
        self, dataloader: DataLoader, validation_dataloader: DataLoader, iteration: int
    ) -> tuple[TrainingStats, TrainingStats]:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        base_lr = self.args.learning_rate(iteration)
        log_scalar('learning_rate', base_lr, iteration)

        self.model.train()

        out_value_mean = torch.tensor(0.0, device=self.model.device)
        out_value_std = torch.tensor(0.0, device=self.model.device)

        def calculate_loss_for_batch(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state, policy_targets, value_targets = batch

            state = state.to(device=self.model.device, dtype=TORCH_DTYPE)
            policy_targets = policy_targets.to(device=self.model.device, dtype=TORCH_DTYPE)
            value_targets = value_targets.to(device=self.model.device, dtype=TORCH_DTYPE)

            value_targets = value_targets.unsqueeze(1)

            out_policy, out_value = self.model(state)

            # Binary cross entropy loss for the policy is definitely not correct, as the policy has multiple classes
            # torch.cross_entropy applies softmax internally, so we don't need to apply it to the output
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            # policy_loss = F.kl_div(F.log_softmax(out_policy, dim=1), policy_targets, reduction='batchmean')
            # policy_loss = F.mse_loss(F.softmax(out_policy, dim=1), policy_targets)
            # policy_loss = F.l1_loss(F.softmax(out_policy, dim=1) * 100, policy_targets * 100)
            # policy_loss = F.mse_loss(F.softmax(out_policy, dim=1) * 10, policy_targets * 10)
            # value_loss = F.l1_loss(out_value, value_targets)  # l1_loss = mean_absolute_error
            value_loss = F.mse_loss(out_value, value_targets)  # mse_loss = mean_squared_error
            # value_loss = F.mse_loss(torch.tanh(out_value), value_targets)  # mse_loss = mean_squared_error

            if False and batchIdx % 50 == 49:
                for value, target in zip(out_value, value_targets[:10]):
                    print('Value:', value.item(), 'Target:', target.item())

                for policy, target in zip(out_policy, policy_targets):
                    break
                    if F.cross_entropy(policy.unsqueeze(0), target.unsqueeze(0)) > 0.5:
                        print('Policy        :', [f'{p:.2f}' for p in policy.tolist()])
                        print('Policy softmax:', [f'{p:.2f}' for p in F.softmax(policy, dim=0).tolist()])
                        print('Target        :', [f'{p:.2f}' for p in target.tolist()])
                        print('Policy loss:', F.cross_entropy(policy.unsqueeze(0), target.unsqueeze(0)))
                        print()

            nonlocal out_value_mean, out_value_std
            out_value_mean += out_value.mean().detach()
            out_value_std += out_value.std().detach()
            # NOTE: Taking the log of the policy targets is necessary because the cross entropy loss expects the logit values, which can be reverted from the softmax distribution in the policy targets by applying the log function: https://math.stackexchange.com/a/3162684
            # optimal_policy_input = torch.full(policy_targets.shape, -1e6, device=self.model.device)
            # mask = policy_targets != 0
            # optimal_policy_input[mask] = torch.log(policy_targets[mask])
            # optimal_policy_loss += F.cross_entropy(optimal_policy_input, policy_targets).detach()
            # current_policy_loss += F.cross_entropy(out_policy, policy_targets).detach()

            # Apparently just as in AZ Paper, give more weight to the policy loss
            # loss = torch.lerp(value_loss, policy_loss, 0.66)
            loss = policy_loss + value_loss

            return policy_loss, value_loss, loss

        total_policy_loss = torch.tensor(0.0, device=self.model.device)
        total_value_loss = torch.tensor(0.0, device=self.model.device)
        total_loss = torch.tensor(0.0, device=self.model.device)

        for batchIdx, batch in enumerate(tqdm(dataloader, desc='Training batches')):
            policy_loss, value_loss, loss = calculate_loss_for_batch(batch)

            # Update learning rate before stepping the optimizer
            # TODO? if batchIdx % 100 == 0:
            # TODO?     batch_percentage = batchIdx / len(dataloader)
            # TODO?     lr = self.args.learning_rate_scheduler(batch_percentage, base_lr)
            # TODO?     for param_group in self.optimizer.param_groups:
            # TODO?         param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 1.5
            # TODO magic hyperparameter and sensible like this?

            self.optimizer.step()

            total_policy_loss += policy_loss.detach()
            total_value_loss += value_loss.detach()
            total_loss += loss.detach()

        train_stats = TrainingStats()
        train_stats.update(
            total_policy_loss.item(),
            total_value_loss.item(),
            total_loss.item(),
            out_value_mean.item(),
            out_value_std.item(),
            len(dataloader),
        )

        log(f'Training stats: {train_stats}')

        # print('Optimal policy loss:', optimal_policy_loss.item() / len(dataloader))
        # print('Current policy loss:', current_policy_loss.item() / len(dataloader))

        # Reset the mean and std for the validation batch
        out_value_mean = torch.tensor(0.0, device=self.model.device)
        out_value_std = torch.tensor(0.0, device=self.model.device)

        self.model.eval()

        validation_total_policy_loss = torch.tensor(0.0, device=self.model.device)
        validation_total_value_loss = torch.tensor(0.0, device=self.model.device)
        validation_total_loss = torch.tensor(0.0, device=self.model.device)

        with torch.no_grad():
            for validation_batch in tqdm(validation_dataloader, desc='Validation batches'):
                val_policy_loss, val_value_loss, val_loss = calculate_loss_for_batch(validation_batch)

                validation_total_policy_loss += val_policy_loss.detach()
                validation_total_value_loss += val_value_loss.detach()
                validation_total_loss += val_loss.detach()

        validation_stats = TrainingStats()
        validation_stats.update(
            validation_total_policy_loss.item(),
            validation_total_value_loss.item(),
            validation_total_loss.item(),
            out_value_mean.item(),
            out_value_std.item(),
            len(validation_dataloader),
        )

        log(f'Validation stats: {validation_stats}')

        return train_stats, validation_stats
