from __future__ import annotations

import uuid
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from pydantic import Field
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.Network import Network
from src.experiment.chess_experiment import GoExperimentConfiguration
from src.games.go.contract import GoStateContract
from src.games.go.training import calculate_go_loss_from_logits
from src.train.GoReplay import GoReplaySnapshot, GoTrainingBatchLoader
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.save_paths import (
    load_checkpoint_manifest,
    load_model,
    load_optimizer,
    save_model_and_optimizer,
)


class GoDistributedQuantumMetrics(FrozenModel):
    policy_loss: float = Field(ge=0.0)
    value_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)


class _LogitForward(nn.Module):
    def __init__(self, model: Network) -> None:
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.logit_forward(states)


def train_go_quantum_distributed(
    configuration: GoExperimentConfiguration,
    snapshot: GoReplaySnapshot,
    current_model_generation: int,
    next_model_generation: int,
    global_step: int,
) -> GoDistributedQuantumMetrics:
    topology = configuration.training.topology.trainer
    world_size = len(topology.ddp_device_ids)
    if world_size < 2:
        raise ValueError('Distributed Go training requires at least two ranks.')
    run_path = Path(configuration.training.save_path)
    identity = uuid.uuid4().hex
    rendezvous_path = (run_path / 'distributed' / f'go-rendezvous-{identity}').resolve()
    metrics_path = run_path / 'distributed' / f'go-metrics-{identity}.json'
    rendezvous_path.parent.mkdir(parents=True, exist_ok=True)
    multiprocessing.spawn(
        _train_rank,
        args=(
            configuration,
            snapshot,
            current_model_generation,
            next_model_generation,
            global_step,
            rendezvous_path.as_uri(),
            metrics_path,
        ),
        nprocs=world_size,
        join=True,
    )
    metrics = GoDistributedQuantumMetrics.model_validate_json(metrics_path.read_text(encoding='utf-8'))
    metrics_path.unlink()
    if rendezvous_path.exists():
        rendezvous_path.unlink()
    return metrics


def _train_rank(
    rank: int,
    configuration: GoExperimentConfiguration,
    snapshot: GoReplaySnapshot,
    current_model_generation: int,
    next_model_generation: int,
    global_step: int,
    rendezvous_uri: str,
    metrics_path: Path,
) -> None:
    training = configuration.training
    topology = training.topology.trainer
    device = (
        torch.device('cpu') if topology.device_type == 'cpu' else torch.device('cuda', topology.ddp_device_ids[rank])
    )
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    distributed.init_process_group(
        backend=topology.process_group_backend,
        init_method=rendezvous_uri,
        rank=rank,
        world_size=len(topology.ddp_device_ids),
    )
    try:
        contract = GoStateContract(
            configuration.go.representation.board_size,
            configuration.go.representation.history_length,
        )
        manifest = load_checkpoint_manifest(current_model_generation, training.save_path)
        model = load_model(
            Path(training.save_path) / manifest.model_path,
            training.network,
            device,
            contract.network_dimensions,
        )
        optimizer = load_optimizer(
            Path(training.save_path) / manifest.optimizer_path,
            model,
            training.trainer.optimizer,
            device,
        )
        distributed_model = DistributedDataParallel(
            _LogitForward(model),
            device_ids=[device.index] if device.type == 'cuda' else None,
        )
        parameters = training.lifecycle.credit
        batches = GoTrainingBatchLoader(
            snapshot,
            global_step=global_step,
            optimizer_steps=parameters.optimizer_steps_per_quantum,
            global_batch_size=training.trainer.global_batch_size,
            world_size=len(topology.ddp_device_ids),
            rank=rank,
            pin_memory=device.type == 'cuda',
        )
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for optimizer_step, batch in enumerate(batches, start=global_step):
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = training.trainer.learning_rate(
                    optimizer_step,
                    training.trainer.optimizer,
                )
            batch = batch.to_device(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            policy_logits, value_logits = distributed_model(batch.states)
            loss = calculate_go_loss_from_logits(
                policy_logits,
                value_logits,
                batch,
                configuration.go.objective,
                device,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.trainer.max_grad_norm)
            optimizer.step()
            totals += torch.tensor(
                (loss.policy.item(), loss.value.item(), loss.total.item()),
                device=device,
                dtype=torch.float64,
            )
        distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
        totals /= len(topology.ddp_device_ids) * parameters.optimizer_steps_per_quantum
        if rank == 0:
            save_model_and_optimizer(model, optimizer, next_model_generation, training.save_path)
            write_text_atomically(
                metrics_path,
                GoDistributedQuantumMetrics(
                    policy_loss=float(totals[0].item()),
                    value_loss=float(totals[1].item()),
                    total_loss=float(totals[2].item()),
                ).model_dump_json(indent=2)
                + '\n',
            )
        distributed.barrier()
    finally:
        distributed.destroy_process_group()
