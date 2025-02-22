import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.Network import Network
from src.settings import TORCH_DTYPE, log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import log
from src.util.timing import timeit


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

    @timeit
    def train(self, dataloader: DataLoader, iteration: int) -> TrainingStats:
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

            if batchIdx % 50 == 49:
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

        for batchIdx, batch in enumerate(tqdm(dataloader, desc='Training batches', total=len(dataloader) - 1)):
            if batchIdx == len(dataloader) - 1:
                # Use the last batch as validation batch
                validation_batch = batch
                break

            policy_loss, value_loss, loss = calculate_loss_for_batch(batch)

            # Update learning rate before stepping the optimizer
            if batchIdx % 100 == 0:
                batch_percentage = batchIdx / len(dataloader)
                lr = self.args.learning_rate_scheduler(batch_percentage, base_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
            # TODO magic hyperparameter and sensible like this?

            self.optimizer.step()

            total_policy_loss += policy_loss.detach()
            total_value_loss += value_loss.detach()
            total_loss += loss.detach()

        total_norm = 0
        for p in self.model.parameters():
            assert p.grad is not None
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        print('Total norm:', total_norm)

        train_stats = TrainingStats()
        train_stats.update(
            total_policy_loss.item(),
            total_value_loss.item(),
            total_loss.item(),
            out_value_mean.item(),
            out_value_std.item(),
            len(dataloader),
        )

        # print('Optimal policy loss:', optimal_policy_loss.item() / len(dataloader))
        # print('Current policy loss:', current_policy_loss.item() / len(dataloader))

        # Reset the mean and std for the validation batch
        out_value_mean = torch.tensor(0.0, device=self.model.device)
        out_value_std = torch.tensor(0.0, device=self.model.device)

        self.model.eval()

        with torch.no_grad():
            validation_stats = TrainingStats()
            val_policy_loss, val_value_loss, val_loss = calculate_loss_for_batch(validation_batch)

            log_scalar('validation/policy_loss', val_policy_loss.item(), iteration)
            log_scalar('validation/value_loss', val_value_loss.item(), iteration)
            log_scalar('validation/loss', val_loss.item(), iteration)
            validation_stats.update(
                val_policy_loss.item(),
                val_value_loss.item(),
                val_loss.item(),
                out_value_mean.item(),
                out_value_std.item(),
                1,
            )

            log(f'Validation stats: {validation_stats}')

        return train_stats
