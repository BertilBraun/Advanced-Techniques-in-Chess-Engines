from __future__ import annotations

import io

import torch
from torch import nn

from src.az.games.go.augmentation import GoSymmetry, transform_sample
from src.az.games.go.configuration import (
    GoGameConfiguration,
    GoObjectiveConfiguration,
    ResidualGoModelConfiguration,
)
from src.az.games.go.losses import GoLossResult, calculate_go_loss
from src.az.games.go.model import ResidualGoModel
from src.az.games.go.replay_codec import GoReplayCodec
from src.az.games.go.samples import GoBatch, GoSample
from src.az.games.api import GameIdentifier
from src.az.replay.envelope import ReplayRecord


class GoTrainingModule:
    def __init__(
        self,
        game_configuration: GoGameConfiguration,
        model_configuration: ResidualGoModelConfiguration,
        objective_configuration: GoObjectiveConfiguration,
        payload_schema_version: int,
        device: torch.device,
        model_initialization_seed: int,
    ) -> None:
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Go training supports CPU or CUDA devices.')
        self._game_configuration = game_configuration
        self._objective_configuration = objective_configuration
        self._codec = GoReplayCodec(game_configuration, payload_schema_version)
        self._device = device
        with torch.random.fork_rng(devices=()):
            torch.manual_seed(model_initialization_seed)
            self._model = ResidualGoModel(game_configuration, model_configuration).to(device)

    @property
    def model(self) -> ResidualGoModel:
        return self._model

    def create_training_batch(
        self,
        records: tuple[ReplayRecord, ...],
        augmentation_seeds: tuple[int, ...],
    ) -> GoBatch:
        if not records or len(records) != len(augmentation_seeds):
            raise ValueError('Go training records and augmentation seeds must be equally sized and nonempty.')
        samples: list[GoSample] = []
        for record, augmentation_seed in zip(records, augmentation_seeds, strict=True):
            if record.envelope.game_identifier is not GameIdentifier.GO:
                raise ValueError('Go training cannot decode another game.')
            if record.envelope.payload_schema_version != self._codec.payload_schema_version:
                raise ValueError('Go training record has an incompatible payload schema.')
            sample = self._codec.decode(record.payload)
            symmetry = GoSymmetry(augmentation_seed % len(GoSymmetry))
            samples.append(transform_sample(sample, symmetry))
        return self._codec.create_batch(tuple(samples))

    def move_batch(self, batch: GoBatch) -> GoBatch:
        return GoBatch(
            inputs=batch.inputs.to(self._device),
            legal_action_masks=batch.legal_action_masks.to(self._device),
            policy_targets=batch.policy_targets.to(self._device),
            value_targets=batch.value_targets.to(self._device),
            policy_weights=batch.policy_weights.to(self._device),
            value_weights=batch.value_weights.to(self._device),
        )

    def calculate_loss(self, batch: GoBatch) -> GoLossResult:
        return calculate_go_loss(
            self._model(batch.inputs),
            batch,
            self._model,
            self._objective_configuration,
        )

    def calculate_loss_with_model(
        self,
        batch: GoBatch,
        model: nn.Module,
    ) -> GoLossResult:
        outputs = model(batch.inputs)
        return calculate_go_loss(
            outputs,
            batch,
            self._model,
            self._objective_configuration,
        )

    def serialize_model(self) -> bytes:
        stream = io.BytesIO()
        torch.save(self._model.state_dict(), stream)
        return stream.getvalue()

    def restore_model(self, artifact: bytes) -> None:
        if not artifact:
            raise ValueError('Go model artifact cannot be empty.')
        self._model.load_state_dict(torch.load(io.BytesIO(artifact), map_location=self._device, weights_only=True))
