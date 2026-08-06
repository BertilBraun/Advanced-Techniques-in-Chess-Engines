from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from src.Network import Network
from src.experiment.chess_experiment import GoExperimentConfiguration
from src.games.go.contract import GoStateContract
from src.games.go.training import calculate_go_loss
from src.self_play.GoSelfPlay import GoSelfPlay
from src.self_play.go_completed_game import GoCompletedGamePublisher
from src.train.CreditPublication import (
    CreditPublicationManifest,
    create_credit_publication_manifest,
    write_credit_publication_manifest,
)
from src.train.CreditTrainingLedger import CreditTrainingLedger, CreditTrainingProgress
from src.train.GoDistributedTraining import train_go_quantum_distributed
from src.train.GoReplay import GoReplayMaintainer, GoReplaySnapshot, GoTrainingBatchLoader
from src.util.save_paths import (
    checkpoint_manifest_path,
    inference_model_path,
    load_checkpoint_manifest,
    load_model,
    load_optimizer,
    model_save_path,
    save_model_and_optimizer,
)


@dataclass(frozen=True)
class GoQuantumResult:
    progress: CreditTrainingProgress
    policy_loss: float
    value_loss: float
    total_loss: float
    publication: CreditPublicationManifest


class GoTrainingLifecycle:
    def __init__(self, run_id: int, configuration: GoExperimentConfiguration) -> None:
        self.run_id = run_id
        self.configuration = configuration
        self.training = configuration.training
        self.run_path = Path(self.training.save_path)
        self.contract = GoStateContract(
            configuration.go.representation.board_size,
            configuration.go.representation.history_length,
        )
        trainer_topology = self.training.topology.trainer
        self.device = (
            torch.device('cpu')
            if trainer_topology.device_type == 'cpu'
            else torch.device('cuda', trainer_topology.rank_zero_device_id)
        )
        parameters = self.training.lifecycle.credit
        self.ledger = CreditTrainingLedger(
            self.run_path,
            parameters,
            self.training.trainer.global_batch_size,
        )
        self.replay = GoReplayMaintainer(
            self.run_path,
            self.contract,
            parameters.replay_capacity_for_model_version(self.ledger.progress.model_version),
            self.training.random_seed,
        )
        self.model, self.optimizer = self._load_training_state(self.ledger.progress.model_version)
        model_path = inference_model_path(model_save_path(self.ledger.progress.model_version, self.run_path))
        self.self_play_workers = tuple(
            GoSelfPlay(
                configuration,
                model_path,
                self.ledger.progress.model_version,
                GoCompletedGamePublisher(self.run_path, run_id, worker_id),
                device_id,
            )
            for worker_id, device_id in enumerate(self.training.topology.self_play.device_ids)
        )
        self._recover_prepared_quantum()

    def run(self) -> Iterator[GoQuantumResult]:
        parameters = self.training.lifecycle.credit
        while self.ledger.progress.completed_optimizer_steps < parameters.maximum_optimizer_steps:
            snapshot = self._earn_training_credits()
            yield self._train_quantum(snapshot)

    def run_one_quantum(self) -> GoQuantumResult:
        return self._train_quantum(self._earn_training_credits())

    def _earn_training_credits(self) -> GoReplaySnapshot:
        parameters = self.training.lifecycle.credit
        required = parameters.presentation_credits_per_quantum(self.training.trainer.global_batch_size)
        capacity = parameters.replay_capacity_for_model_version(self.ledger.progress.model_version)
        while True:
            snapshot = self.replay.maintain(capacity)
            progress = self.ledger.reconcile_credited_samples(snapshot.credited_samples)
            if progress.can_train(required) and len(snapshot.samples) >= self.training.trainer.global_batch_size:
                return snapshot
            for worker in self.self_play_workers:
                worker.generate(self.training.topology.self_play.parallel_games_per_process)

    def _train_quantum(self, snapshot: GoReplaySnapshot) -> GoQuantumResult:
        next_generation = self.ledger.progress.model_version + 1
        world_size = len(self.training.topology.trainer.ddp_device_ids)
        if world_size == 1:
            policy_loss, value_loss, total_loss = self._train_local(snapshot)
            save_model_and_optimizer(self.model, self.optimizer, next_generation, self.run_path)
        else:
            metrics = train_go_quantum_distributed(
                self.configuration,
                snapshot,
                self.ledger.progress.model_version,
                next_generation,
                self.ledger.progress.sampler_global_step,
            )
            policy_loss = metrics.policy_loss
            value_loss = metrics.value_loss
            total_loss = metrics.total_loss
            self.model, self.optimizer = self._load_training_state(next_generation)
        prepared = self.ledger.prepare_quantum(checkpoint_manifest_path(next_generation, self.run_path))
        progress = self.ledger.commit_prepared_quantum()
        publication = create_credit_publication_manifest(
            self.run_path,
            progress,
            self.training.trainer.global_batch_size,
        )
        write_credit_publication_manifest(self.run_path, publication)
        published_model = inference_model_path(model_save_path(next_generation, self.run_path))
        for worker in self.self_play_workers:
            worker.refresh_model(next_generation, published_model)
        assert prepared.prepared_progress == progress
        return GoQuantumResult(
            progress=progress,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            publication=publication,
        )

    def _train_local(self, snapshot: GoReplaySnapshot) -> tuple[float, float, float]:
        parameters = self.training.lifecycle.credit
        batches = GoTrainingBatchLoader(
            snapshot,
            global_step=self.ledger.progress.sampler_global_step,
            optimizer_steps=parameters.optimizer_steps_per_quantum,
            global_batch_size=self.training.trainer.global_batch_size,
            world_size=1,
            rank=0,
            pin_memory=self.device.type == 'cuda',
        )
        policy_total = 0.0
        value_total = 0.0
        total = 0.0
        self.model.train()
        for optimizer_step, batch in enumerate(batches, start=self.ledger.progress.completed_optimizer_steps):
            learning_rate = self.training.trainer.learning_rate(
                optimizer_step,
                self.training.trainer.optimizer,
            )
            for parameter_group in self.optimizer.param_groups:
                parameter_group['lr'] = learning_rate
            self.optimizer.zero_grad(set_to_none=True)
            loss = calculate_go_loss(
                self.model, batch.to_device(self.device, non_blocking=True), self.configuration.go.objective
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.trainer.max_grad_norm)
            self.optimizer.step()
            policy_total += float(loss.policy.detach().cpu())
            value_total += float(loss.value.detach().cpu())
            total += float(loss.total.detach().cpu())
        step_count = parameters.optimizer_steps_per_quantum
        return (
            policy_total / step_count,
            value_total / step_count,
            total / step_count,
        )

    def _load_training_state(self, model_generation: int) -> tuple[Network, torch.optim.Optimizer]:
        manifest = load_checkpoint_manifest(model_generation, self.run_path)
        model = load_model(
            self.run_path / manifest.model_path,
            self.training.network,
            self.device,
            self.contract.network_dimensions,
        )
        optimizer = load_optimizer(
            self.run_path / manifest.optimizer_path,
            model,
            self.training.trainer.optimizer,
            self.device,
        )
        return model, optimizer

    def _recover_prepared_quantum(self) -> None:
        prepared = self.ledger.prepared_quantum
        if prepared is None:
            return
        progress = self.ledger.commit_prepared_quantum()
        publication = create_credit_publication_manifest(
            self.run_path,
            progress,
            self.training.trainer.global_batch_size,
        )
        write_credit_publication_manifest(self.run_path, publication)
        self.model, self.optimizer = self._load_training_state(progress.model_version)
        published_model = inference_model_path(model_save_path(progress.model_version, self.run_path))
        for worker in self.self_play_workers:
            worker.refresh_model(progress.model_version, published_model)
