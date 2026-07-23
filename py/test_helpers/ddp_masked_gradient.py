from multiprocessing.connection import Connection

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.Network import Network
from src.self_play.SelfPlayDataset import TrainingBatch
from src.self_play.value_target import FinalOutcome, TerminationReason
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingParams


class MaskedValueNetwork(Network):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.device = torch.device('cpu')
        self.policy_logits = nn.Parameter(torch.zeros(2))
        self.value_logits = nn.Parameter(torch.zeros(3))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        return (
            self.policy_logits.unsqueeze(0).expand(batch_size, -1),
            self.value_logits.unsqueeze(0).expand(batch_size, -1),
        )

    def logit_forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(states)


def masked_value_gradient_rank(
    rank: int,
    initialization_method: str,
    result_connection: Connection,
) -> None:
    distributed.init_process_group(
        backend='gloo',
        init_method=initialization_method,
        rank=rank,
        world_size=2,
    )
    try:
        model = MaskedValueNetwork()
        distributed_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        training_parameters = TrainingParams(
            num_epochs=1,
            global_batch_size=4,
            local_batch_size=2,
            optimizer='sgd',
            sampling_window=lambda _: 1,
            learning_rate=lambda _iteration, _optimizer: 0.0,
            learning_rate_scheduler=lambda _progress, learning_rate: learning_rate,
            policy_loss_weight=0.0,
            value_loss_weight=1.0,
            outcome_value_loss_weight=0.85,
            mcts_value_loss_weight=0.15,
        )
        trainer = Trainer(
            model,
            optimizer,
            training_parameters,
            training_model=distributed_model,
            rank=rank,
        )
        if rank == 0:
            outcomes = (FinalOutcome.WIN, FinalOutcome.DRAW)
            eligibility = (True, False)
            reasons = (TerminationReason.NATURAL, TerminationReason.PLY_CAP)
        else:
            outcomes = (FinalOutcome.LOSS, FinalOutcome.LOSS)
            eligibility = (True, True)
            reasons = (TerminationReason.NATURAL, TerminationReason.RESIGNATION)
        batch = TrainingBatch(
            states=torch.zeros((2, 1)),
            policy_targets=torch.full((2, 2), 0.5),
            final_outcomes=torch.tensor(tuple(int(outcome) for outcome in outcomes)),
            mcts_root_values=torch.zeros(2),
            outcome_target_eligible=torch.tensor(eligibility),
            termination_reasons=torch.tensor(tuple(int(reason) for reason in reasons)),
            plies=torch.arange(2, dtype=torch.int32),
            current_player_piece_counts=torch.full((2,), 8, dtype=torch.int8),
            opponent_piece_counts=torch.full((2,), 8, dtype=torch.int8),
        )

        loss = trainer._calculate_loss_for_batch(batch)
        loss.total_loss.backward()
        assert model.value_logits.grad is not None
        result_connection.send(tuple(float(value) for value in model.value_logits.grad))
    finally:
        result_connection.close()
        distributed.destroy_process_group()
