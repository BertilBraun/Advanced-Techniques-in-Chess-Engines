import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.Network import Network
from src.settings import TORCH_DTYPE, log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import error, log
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
    def train(
        self, dataloader: DataLoader, validation_dataloader: DataLoader, iteration: int
    ) -> tuple[TrainingStats, TrainingStats]:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        base_lr = self.args.learning_rate(iteration, self.args.optimizer)
        log_scalar('learning_rate', base_lr, iteration)
        log(f'Setting learning rate to {base_lr} for iteration {iteration}')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

        self.model.train()

        out_value_mean = torch.tensor(0.0, device=self.model.device)
        out_value_std = torch.tensor(0.0, device=self.model.device)

        torch.autograd.set_detect_anomaly(True)

        def calculate_loss_for_batch(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state, policy_targets, value_targets = batch

            state = state.to(device=self.model.device, dtype=TORCH_DTYPE)
            policy_targets = policy_targets.to(device=self.model.device, dtype=TORCH_DTYPE)
            value_targets = value_targets.to(device=self.model.device, dtype=TORCH_DTYPE)

            value_targets = value_targets.unsqueeze(1)

            policy_logits, value_logits = self.model.logit_forward(state)
            value_output = torch.tanh(value_logits)

            # Binary cross entropy loss for the policy is definitely not correct, as the policy has multiple classes
            # torch.cross_entropy applies softmax internally, so we don't need to apply it to the output
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
            # policy_loss = F.kl_div(F.log_softmax(out_policy, dim=1), policy_targets, reduction='batchmean')
            # policy_loss = F.mse_loss(F.softmax(out_policy, dim=1), policy_targets)
            # policy_loss = F.l1_loss(F.softmax(out_policy, dim=1) * 100, policy_targets * 100)
            # policy_loss = F.mse_loss(F.softmax(out_policy, dim=1) * 10, policy_targets * 10)
            # value_loss = F.l1_loss(out_value, value_targets)  # l1_loss = mean_absolute_error
            # value_loss = F.mse_loss(out_value, value_targets)  # mse_loss = mean_squared_error

            # NOTE:
            # At |logit| ≈ 4 the derivative of tanh is already < 0.002, so the gradient almost vanishes.
            # Smoothing the value targets to avoid overfitting to the extreme values which no longer give sensible gradients
            # Non-saturated logits: 0.95 corresponds to logit ≈ +2.94, 0.05 to logit ≈ −2.94 – well inside the linear part of tanh. The gradient w.r.t. the logits is therefore still ≈ 0.05–0.1 instead of ≈ 0.001.
            value_targets = 0.95 * value_targets

            # BCE is not suitable for regression tasks and produces spikey outputs at the extremes, so we use MSE instead
            # value_targets = (value_targets + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1] range for binary cross entropy
            # value_loss = F.binary_cross_entropy_with_logits(value_logits, value_targets)
            value_loss = F.mse_loss(value_output, value_targets)

            print(f'Value output shape: {value_output.shape}, value targets shape: {value_targets.shape}')
            # is finite check
            print(
                f'Value output finite: {torch.isfinite(value_output).all()}, value targets finite: {torch.isfinite(value_targets).all()}'
            )
            print('Value output:')
            for i, v in enumerate(value_output):
                print(f'Value output[{i}]: {v.item()}')
                if abs(v.item()) > 1.0:
                    error(f'Value output[{i}] is out of bounds: {v.item()}')
            print('Value targets:')
            for i, v in enumerate(value_targets):
                print(f'Value targets[{i}]: {v.item()}')
                if abs(v.item()) > 1.0:
                    error(f'Value targets[{i}] is out of bounds: {v.item()}')

            if False and (batchIdx % 50 == 1 or True):
                count_unique_values_in_value_targets = torch.unique(value_targets.to(torch.float32)).numel()
                count_unique_values_in_out_value = torch.unique(value_logits.to(torch.float32)).numel()
                unique_input_states_by_batch = torch.unique(state.to(torch.float32), dim=0).numel()

                print(
                    'Batch:',
                    batchIdx,
                    'State shape:',
                    state.shape,
                    'Unique input states:',
                    unique_input_states_by_batch,
                    'Unique value targets:',
                    count_unique_values_in_value_targets,
                    'Unique out value:',
                    count_unique_values_in_out_value,
                )
                seen = set()
                seen_again = set()
                for s, value, target in zip(state, value_logits, value_targets):
                    value = value.item()
                    target = target.item()
                    if (value, target) in seen_again:
                        continue
                    # compress 14x8x8 binary tensor into 14 64-bit integers
                    compressed = []
                    for i in range(14):
                        num = 0
                        for j in range(8):
                            for k in range(8):
                                num = (num << 1) | int(s[i, j, k])
                        compressed.append(num)
                    compressed = tuple(compressed)
                    if (value, target, compressed) in seen:
                        seen_again.add((value, target, compressed))
                    seen.add((value, target, compressed))

                    print('Value:', value, 'Target:', target, 'Compressed:', compressed)

                for policy, target in zip(policy_logits, policy_targets):
                    break
                    if F.cross_entropy(policy.unsqueeze(0), target.unsqueeze(0)) > 0.5:
                        print('Policy        :', [f'{p:.2f}' for p in policy.tolist()])
                        print('Policy softmax:', [f'{p:.2f}' for p in F.softmax(policy, dim=0).tolist()])
                        print('Target        :', [f'{p:.2f}' for p in target.tolist()])
                        print('Policy loss:', F.cross_entropy(policy.unsqueeze(0), target.unsqueeze(0)))
                        print()

            nonlocal out_value_mean, out_value_std
            out_value_mean += value_output.detach().mean()
            out_value_std += value_output.detach().std()

            # Apparently just as in AZ Paper, give more weight to the policy loss
            loss = policy_loss + 0.3 * value_loss  # TODO move to hyperparameters

            return policy_loss, value_loss, loss

        total_policy_loss = torch.tensor(0.0, device=self.model.device)
        total_value_loss = torch.tensor(0.0, device=self.model.device)
        total_loss = torch.tensor(0.0, device=self.model.device)
        total_gradient_norm = torch.tensor(0.0, device=self.model.device)

        for batchIdx, batch in enumerate(tqdm(dataloader, desc='Training batches')):
            try:
                policy_loss, value_loss, loss = calculate_loss_for_batch(batch)

                # Update learning rate before stepping the optimizer
                # TODO? if batchIdx % 100 == 0:
                # TODO?     batch_percentage = batchIdx / len(dataloader)
                # TODO?     lr = self.args.learning_rate_scheduler(batch_percentage, base_lr)
                # TODO?     for param_group in self.optimizer.param_groups:
                # TODO?         param_group['lr'] = lr

                self.optimizer.zero_grad()
                loss.backward()
                # TODO magic hyperparameter and sensible like this?
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                total_gradient_norm += norm.detach()

                self.optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_loss += loss.detach()
            except Exception as e:
                print(f'Error calculating value loss: {e}')
                print('Value loss calculation failed.')
                exit()

        train_stats = TrainingStats(
            total_policy_loss.item(),
            total_value_loss.item(),
            total_loss.item(),
            out_value_mean.item(),
            out_value_std.item(),
            total_gradient_norm.item(),
            num_batches=len(dataloader),
        )

        log(f'Training stats: {train_stats}')

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

        validation_stats = TrainingStats(
            validation_total_policy_loss.item(),
            validation_total_value_loss.item(),
            validation_total_loss.item(),
            out_value_mean.item(),
            out_value_std.item(),
            0.0,  # No gradient norm for validation
            num_batches=len(validation_dataloader),
        )

        log(f'Validation stats: {validation_stats}')

        return train_stats, validation_stats
