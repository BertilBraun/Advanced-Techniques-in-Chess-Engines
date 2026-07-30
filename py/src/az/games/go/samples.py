from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from src.az.games.go.configuration import GoGameConfiguration
from src.az.replay.envelope import GameTermination


FloatArray = npt.NDArray[np.float32]
IntegerArray = npt.NDArray[np.int32]
BooleanArray = npt.NDArray[np.bool_]


def _immutable_array(array: npt.NDArray[np.generic], dtype: np.dtype[np.generic]) -> npt.NDArray[np.generic]:
    copied = np.asarray(array, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class DensePolicyTarget:
    probabilities: FloatArray

    def __post_init__(self) -> None:
        probabilities = _immutable_array(self.probabilities, np.dtype('<f4'))
        if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError('Dense policy probabilities must be a finite nonnegative vector.')
        object.__setattr__(self, 'probabilities', probabilities)


@dataclass(frozen=True)
class SparsePolicyTarget:
    actions: IntegerArray
    weights: FloatArray

    def __post_init__(self) -> None:
        actions = _immutable_array(self.actions, np.dtype('<i4'))
        weights = _immutable_array(self.weights, np.dtype('<f4'))
        if actions.ndim != 1 or weights.ndim != 1 or len(actions) != len(weights) or not len(actions):
            raise ValueError('Sparse policy actions and weights must be equally sized nonempty vectors.')
        if len(np.unique(actions)) != len(actions) or np.any(actions < 0):
            raise ValueError('Sparse policy actions must be unique and nonnegative.')
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('Sparse policy weights must be finite and positive.')
        object.__setattr__(self, 'actions', actions)
        object.__setattr__(self, 'weights', weights)


PolicyTarget = DensePolicyTarget | SparsePolicyTarget


@dataclass(frozen=True)
class PendingGoSearchSample:
    input_planes: FloatArray
    legal_action_mask: BooleanArray
    policy_target: PolicyTarget
    policy_weight: float


@dataclass(frozen=True)
class GoSample:
    input_planes: FloatArray
    legal_action_mask: BooleanArray
    policy_target: PolicyTarget
    policy_weight: float
    value_target: float | None
    value_weight: float

    def __post_init__(self) -> None:
        planes = _immutable_array(self.input_planes, np.dtype('<f4'))
        legal = _immutable_array(self.legal_action_mask, np.dtype('?'))
        if planes.ndim != 3 or not np.all((planes == 0) | (planes == 1)):
            raise ValueError('Go input planes must be a binary rank-three array.')
        if legal.ndim != 1 or not np.any(legal):
            raise ValueError('Go legal-action mask must be a nonempty vector with at least one legal action.')
        if not np.isfinite(self.policy_weight) or self.policy_weight < 0:
            raise ValueError('Policy weight must be finite and nonnegative.')
        if not np.isfinite(self.value_weight) or self.value_weight < 0:
            raise ValueError('Value weight must be finite and nonnegative.')
        if self.value_target is not None and (not np.isfinite(self.value_target) or not -1 <= self.value_target <= 1):
            raise ValueError('Value target must be finite and between -1 and 1.')
        if (self.value_target is None) != (self.value_weight == 0):
            raise ValueError('A value target is required exactly when value weight is positive.')
        action_count = len(legal)
        match self.policy_target:
            case DensePolicyTarget(probabilities=probabilities):
                if len(probabilities) != action_count:
                    raise ValueError('Dense policy target length must equal the action count.')
                if np.any(probabilities[~legal] != 0):
                    raise ValueError('Dense policy target assigns weight to an illegal action.')
                target_total = float(probabilities.sum(dtype=np.float64))
            case SparsePolicyTarget(actions=actions, weights=weights):
                if np.any(actions >= action_count) or np.any(~legal[actions]):
                    raise ValueError('Sparse policy target contains an invalid or illegal action.')
                target_total = float(weights.sum(dtype=np.float64))
        if self.policy_weight > 0 and target_total <= 0:
            raise ValueError('Eligible policy targets must have positive target mass.')
        if not np.isfinite(target_total):
            raise ValueError('Policy target total mass must be finite.')
        object.__setattr__(self, 'input_planes', planes)
        object.__setattr__(self, 'legal_action_mask', legal)

    def validate_configuration(self, configuration: GoGameConfiguration) -> None:
        expected_shape = (
            configuration.input_plane_count,
            configuration.board_size,
            configuration.board_size,
        )
        if self.input_planes.shape != expected_shape:
            raise ValueError(f'Go input shape must be {expected_shape}.')
        if len(self.legal_action_mask) != configuration.action_count:
            raise ValueError('Go legal-action mask length must equal N squared plus one.')


@dataclass(frozen=True)
class GoBatch:
    inputs: Tensor
    legal_action_masks: Tensor
    policy_targets: Tensor
    value_targets: Tensor
    policy_weights: Tensor
    value_weights: Tensor

    @property
    def size(self) -> int:
        return self.inputs.shape[0]


class NativeGoEncoding(Protocol):
    @property
    def planes(self) -> int: ...

    @property
    def board_size(self) -> int: ...

    @property
    def values(self) -> list[int]: ...


class NativeSearchTelemetry(Protocol):
    @property
    def policy_target_eligible(self) -> bool: ...

    @property
    def policy_target_weight(self) -> float: ...


class NativeGoSearchResult(Protocol):
    @property
    def root_visits(self) -> list[int]: ...

    @property
    def telemetry(self) -> NativeSearchTelemetry: ...


def pending_sample_from_native(
    encoding: NativeGoEncoding,
    legal_actions: tuple[int, ...],
    result: NativeGoSearchResult,
    configuration: GoGameConfiguration,
) -> PendingGoSearchSample:
    expected_values = configuration.input_plane_count * configuration.board_size**2
    if encoding.planes != configuration.input_plane_count or encoding.board_size != configuration.board_size:
        raise ValueError('Native encoding shape does not match the Go configuration.')
    if len(encoding.values) != expected_values:
        raise ValueError('Native encoding has an invalid flattened length.')
    visits = np.asarray(result.root_visits, dtype=np.float32)
    if len(visits) != configuration.action_count:
        raise ValueError('Native root visits do not match the Go action count.')
    if not legal_actions or any(action < 0 or action >= configuration.action_count for action in legal_actions):
        raise ValueError('Native legal actions are outside the Go action space.')
    if len(set(legal_actions)) != len(legal_actions):
        raise ValueError('Native legal actions contain a duplicate.')
    if result.telemetry.policy_target_eligible != (result.telemetry.policy_target_weight > 0):
        raise ValueError('Native policy eligibility must exactly match positive policy weight.')
    legal_mask = np.zeros(configuration.action_count, dtype=np.bool_)
    legal_mask[np.asarray(legal_actions, dtype=np.int32)] = True
    if np.any(visits[~legal_mask] != 0):
        raise ValueError('Native search assigned visits to an illegal action.')
    policy_weight = result.telemetry.policy_target_weight if result.telemetry.policy_target_eligible else 0.0
    return PendingGoSearchSample(
        input_planes=np.asarray(encoding.values, dtype=np.float32).reshape(
            configuration.input_plane_count,
            configuration.board_size,
            configuration.board_size,
        ),
        legal_action_mask=legal_mask,
        policy_target=DensePolicyTarget(visits),
        policy_weight=policy_weight,
    )


def finalize_sample(
    pending: PendingGoSearchSample,
    value_target: float | None,
    configured_value_weight: float,
    termination: GameTermination,
) -> GoSample:
    """Attach the terminal value from the encoded position's player-to-move perspective."""
    if configured_value_weight <= 0 or not np.isfinite(configured_value_weight):
        raise ValueError('Configured value weight must be finite and positive.')
    if termination is GameTermination.SAFETY_PLY_CAP:
        resolved_target = None
        resolved_weight = 0.0
    else:
        if value_target is None:
            raise ValueError('A completed non-censored game requires a value target.')
        resolved_target = value_target
        resolved_weight = configured_value_weight
    return GoSample(
        input_planes=pending.input_planes,
        legal_action_mask=pending.legal_action_mask,
        policy_target=pending.policy_target,
        policy_weight=pending.policy_weight,
        value_target=resolved_target,
        value_weight=resolved_weight,
    )


def create_batch(samples: tuple[GoSample, ...], configuration: GoGameConfiguration) -> GoBatch:
    if not samples:
        raise ValueError('Cannot create an empty Go batch.')
    for sample in samples:
        sample.validate_configuration(configuration)
    dense_targets = np.zeros((len(samples), configuration.action_count), dtype=np.float32)
    for index, sample in enumerate(samples):
        match sample.policy_target:
            case DensePolicyTarget(probabilities=probabilities):
                dense_targets[index] = probabilities
            case SparsePolicyTarget(actions=actions, weights=weights):
                dense_targets[index, actions] = weights
    target_totals = dense_targets.sum(axis=1, keepdims=True, dtype=np.float64)
    positive_mass = target_totals[:, 0] > 0
    dense_targets[positive_mass] /= target_totals[positive_mass]
    value_targets = np.asarray(
        [0.0 if sample.value_target is None else sample.value_target for sample in samples],
        dtype=np.float32,
    )
    return GoBatch(
        inputs=torch.from_numpy(np.stack([sample.input_planes for sample in samples]).copy()),
        legal_action_masks=torch.from_numpy(np.stack([sample.legal_action_mask for sample in samples]).copy()),
        policy_targets=torch.from_numpy(dense_targets),
        value_targets=torch.from_numpy(value_targets),
        policy_weights=torch.tensor([sample.policy_weight for sample in samples], dtype=torch.float32),
        value_weights=torch.tensor([sample.value_weight for sample in samples], dtype=torch.float32),
    )
