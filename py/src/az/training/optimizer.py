from __future__ import annotations

import io
import math
from base64 import b64decode, b64encode
from typing import Annotated, Literal

import torch
from pydantic import Field, TypeAdapter, model_validator
from torch import nn
from torch.optim import Optimizer

from src.az.config.base import FrozenModel
from src.az.config.training import (
    AdamWOptimizerConfiguration,
    ConstantLearningRate,
    LearningRateConfiguration,
    OptimizerConfiguration,
    PiecewiseLearningRate,
    SgdOptimizerConfiguration,
)


class LearningRateState(FrozenModel):
    completed_optimizer_steps: int = Field(ge=0)
    current_learning_rate: float = Field(gt=0)

    @model_validator(mode='after')
    def validate_finite(self) -> LearningRateState:
        if not math.isfinite(self.current_learning_rate):
            raise ValueError('Current learning rate must be finite.')
        return self


def create_optimizer(model: nn.Module, configuration: OptimizerConfiguration) -> Optimizer:
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError('Cannot optimize a model without parameters.')
    match configuration:
        case AdamWOptimizerConfiguration(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        ):
            return torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=(beta_1, beta_2),
                eps=epsilon,
                weight_decay=weight_decay,
            )
        case SgdOptimizerConfiguration(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        ):
            return torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )


def optimizer_base_learning_rate(configuration: OptimizerConfiguration) -> float:
    return configuration.learning_rate


def _multiplier(configuration: LearningRateConfiguration, optimizer_step: int) -> float:
    match configuration:
        case ConstantLearningRate(multiplier=multiplier):
            return multiplier
        case PiecewiseLearningRate(stages=stages):
            selected = stages[0].multiplier
            for stage in stages[1:]:
                if stage.start_optimizer_step > optimizer_step:
                    break
                selected = stage.multiplier
            return selected


class LearningRateController:
    def __init__(
        self,
        optimizer: Optimizer,
        base_learning_rate: float,
        configuration: LearningRateConfiguration,
        state: LearningRateState | None = None,
    ) -> None:
        if not math.isfinite(base_learning_rate) or base_learning_rate <= 0:
            raise ValueError('Base learning rate must be finite and positive.')
        self._optimizer = optimizer
        self._base_learning_rate = base_learning_rate
        self._configuration = configuration
        if state is None:
            current = self._learning_rate(0)
            self._state = LearningRateState(completed_optimizer_steps=0, current_learning_rate=current)
        else:
            expected = self._learning_rate(state.completed_optimizer_steps)
            if state.current_learning_rate != expected:
                raise ValueError('Checkpoint learning-rate state does not match the configured schedule.')
            self._state = state
        self._apply(self._state.current_learning_rate)

    @property
    def state(self) -> LearningRateState:
        return self._state

    def advance(self) -> LearningRateState:
        completed = self._state.completed_optimizer_steps + 1
        learning_rate = self._learning_rate(completed)
        self._state = LearningRateState(
            completed_optimizer_steps=completed,
            current_learning_rate=learning_rate,
        )
        self._apply(learning_rate)
        return self._state

    def _learning_rate(self, optimizer_step: int) -> float:
        return self._base_learning_rate * _multiplier(self._configuration, optimizer_step)

    def _apply(self, learning_rate: float) -> None:
        for parameter_group in self._optimizer.param_groups:
            parameter_group['lr'] = learning_rate


def serialize_optimizer(optimizer: Optimizer) -> bytes:
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    return stream.getvalue()


def restore_optimizer(optimizer: Optimizer, artifact: bytes) -> None:
    if not artifact:
        raise ValueError('Optimizer artifact cannot be empty.')
    optimizer.load_state_dict(torch.load(io.BytesIO(artifact), map_location='cpu', weights_only=True))


def serialize_torch_random_state() -> bytes:
    return torch.get_rng_state().numpy().tobytes()


def restore_torch_random_state(artifact: bytes) -> None:
    if not artifact:
        raise ValueError('Torch random-state artifact cannot be empty.')
    state = torch.frombuffer(bytearray(artifact), dtype=torch.uint8).clone()
    torch.set_rng_state(state)


class NoCudaRandomStream(FrozenModel):
    kind: Literal['none']


class AssignedCudaRandomStream(FrozenModel):
    kind: Literal['assigned_cuda']
    device_index: int = Field(ge=0)
    device_name: str = Field(min_length=1)
    state_base64: str = Field(min_length=1)


CudaRandomStream = Annotated[
    NoCudaRandomStream | AssignedCudaRandomStream,
    Field(discriminator='kind'),
]


def serialize_assigned_cuda_random_state(device: torch.device) -> bytes:
    if device.type != 'cuda':
        return (NoCudaRandomStream(kind='none').model_dump_json() + '\n').encode()
    if device.index is None:
        raise ValueError('CUDA training device must have an explicit index.')
    state = torch.cuda.get_rng_state(device)
    artifact = AssignedCudaRandomStream(
        kind='assigned_cuda',
        device_index=device.index,
        device_name=torch.cuda.get_device_name(device),
        state_base64=b64encode(state.cpu().numpy().tobytes()).decode('ascii'),
    )
    return (artifact.model_dump_json() + '\n').encode()


def restore_assigned_cuda_random_state(artifact: bytes, device: torch.device) -> None:
    if not artifact:
        raise ValueError('CUDA random-state artifact cannot be empty.')
    state = TypeAdapter(CudaRandomStream).validate_json(artifact)
    match state:
        case NoCudaRandomStream():
            if device.type == 'cuda':
                raise ValueError('CUDA trainer cannot restore an absent CUDA random stream.')
        case AssignedCudaRandomStream(
            device_index=device_index,
            device_name=device_name,
            state_base64=state_base64,
        ):
            if device.type != 'cuda' or device.index != device_index:
                raise ValueError('CUDA random stream device identity does not match trainer assignment.')
            if torch.cuda.get_device_name(device) != device_name:
                raise ValueError('CUDA random stream model identity does not match trainer assignment.')
            state_bytes = b64decode(state_base64, validate=True)
            random_state = torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).clone()
            torch.cuda.set_rng_state(random_state, device)


def serialize_gradient_scaler(scaler: torch.amp.GradScaler) -> bytes:
    stream = io.BytesIO()
    torch.save(scaler.state_dict(), stream)
    return stream.getvalue()


def restore_gradient_scaler(scaler: torch.amp.GradScaler, artifact: bytes) -> None:
    if not artifact:
        raise ValueError('Gradient-scaler artifact cannot be empty.')
    scaler.load_state_dict(torch.load(io.BytesIO(artifact), map_location='cpu', weights_only=True))
