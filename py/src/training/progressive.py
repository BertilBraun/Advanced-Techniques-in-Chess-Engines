from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from src.training.checkpoint import CheckpointReference
from src.training.network import NetworkParams
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel

if TYPE_CHECKING:
    from src.replay.manager import ReplayDescription


SECONDS_PER_DAY = 86_400.0


class ProgressiveModelDefinition(FrozenModel):
    model_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    training_start_days: Decimal = Field(ge=0)
    network: NetworkParams

    def is_eligible(self, elapsed_seconds: float) -> bool:
        if elapsed_seconds < 0.0:
            raise ValueError('Elapsed run time cannot be negative.')
        return Decimal(str(elapsed_seconds)) >= self.training_start_days * Decimal(str(SECONDS_PER_DAY))


class TotalLossEmaPromotionConfiguration(FrozenModel):
    decay: float = Field(gt=0.0, lt=1.0)
    warmup_quanta: int = Field(gt=0)
    maximum_relative_loss: float = Field(default=1.01, ge=1.0)


class ProgressiveModelSizingConfiguration(FrozenModel):
    models: tuple[ProgressiveModelDefinition, ProgressiveModelDefinition, ProgressiveModelDefinition]
    promotion: TotalLossEmaPromotionConfiguration

    @model_validator(mode='after')
    def validate_models(self) -> ProgressiveModelSizingConfiguration:
        model_ids = tuple(model.model_id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError('Progressive model IDs must be unique.')
        starts = tuple(model.training_start_days for model in self.models)
        if starts[0] != Decimal(0):
            raise ValueError('The initial progressive model must start at day zero.')
        if any(starts[index] <= starts[index - 1] for index in range(1, len(starts))):
            raise ValueError('Progressive model training starts must be strictly increasing.')
        return self

    def model(self, model_id: str) -> ProgressiveModelDefinition:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise ValueError(f'Unknown progressive model ID: {model_id}')

    def eligible_model_ids(self, elapsed_seconds: float) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models if model.is_eligible(elapsed_seconds))

    def successor(self, model_id: str) -> ProgressiveModelDefinition | None:
        for index, model in enumerate(self.models):
            if model.model_id == model_id:
                return self.models[index + 1] if index + 1 < len(self.models) else None
        raise ValueError(f'Unknown progressive model ID: {model_id}')


class ComparableLossEma(FrozenModel):
    value: float
    observations: int = Field(gt=0)

    def update(self, loss: float, decay: float) -> ComparableLossEma:
        if loss < 0.0:
            raise ValueError('Comparable training loss cannot be negative.')
        return ComparableLossEma(
            value=decay * self.value + (1.0 - decay) * loss,
            observations=self.observations + 1,
        )


class ProgressiveCandidateState(FrozenModel):
    model_id: str
    completed_optimizer_steps: int = Field(default=0, ge=0)
    checkpoint: CheckpointReference | None = None
    comparable_loss_ema: ComparableLossEma | None = None


class ReplayBatchIdentity(FrozenModel):
    source_optimizer_steps: int = Field(ge=0)
    path: Path
    head: int = Field(ge=0)
    size: int = Field(ge=0)
    logical_capacity: int = Field(gt=0)
    maximum_capacity: int = Field(gt=0)
    layout_digest: str

    @classmethod
    def from_replay(
        cls,
        replay: ReplayDescription,
        source_optimizer_steps: int,
    ) -> ReplayBatchIdentity:
        return cls(
            source_optimizer_steps=source_optimizer_steps,
            path=replay.path,
            head=replay.head,
            size=replay.size,
            logical_capacity=replay.logical_capacity,
            maximum_capacity=replay.maximum_capacity,
            layout_digest=replay.layout.digest,
        )


class CompletedCandidateTraining(FrozenModel):
    model_id: str
    completed_optimizer_steps: int = Field(gt=0)
    checkpoint: CheckpointReference
    comparable_total_loss: float = Field(ge=0.0)


class PendingProgressiveQuantum(FrozenModel):
    target_global_optimizer_steps: int = Field(gt=0)
    replay_batch: ReplayBatchIdentity
    required_model_ids: tuple[str, ...] = Field(min_length=1)
    completed: tuple[CompletedCandidateTraining, ...] = ()

    @model_validator(mode='after')
    def validate_completion_prefix(self) -> PendingProgressiveQuantum:
        completed_ids = tuple(result.model_id for result in self.completed)
        if completed_ids != self.required_model_ids[: len(completed_ids)]:
            raise ValueError('Progressive candidate results must be recorded in configured training order.')
        return self

    @property
    def next_model_id(self) -> str | None:
        if len(self.completed) == len(self.required_model_ids):
            return None
        return self.required_model_ids[len(self.completed)]


class ProgressiveTrainingState(FrozenModel):
    schema_version: Literal[1] = 1
    active_model_id: str
    candidates: tuple[ProgressiveCandidateState, ...]
    pending_quantum: PendingProgressiveQuantum | None = None


class ProgressiveTrainingStateStore:
    def __init__(self, path: Path, configuration: ProgressiveModelSizingConfiguration) -> None:
        self.path = path
        self.configuration = configuration
        if path.exists():
            self.state = ProgressiveTrainingState.model_validate_json(path.read_text(encoding='utf-8'))
            self._validate_state(self.state)
        else:
            self.state = ProgressiveTrainingState(
                active_model_id=configuration.models[0].model_id,
                candidates=tuple(ProgressiveCandidateState(model_id=model.model_id) for model in configuration.models),
            )
            self.save()

    def begin_quantum(
        self,
        elapsed_seconds: float,
        replay: ReplayDescription,
        source_optimizer_steps: int,
        optimizer_steps_per_quantum: int,
    ) -> PendingProgressiveQuantum:
        replay_batch = ReplayBatchIdentity.from_replay(replay, source_optimizer_steps)
        if self.state.pending_quantum is not None:
            if self.state.pending_quantum.replay_batch != replay_batch:
                raise ValueError('Pending progressive quantum replay batches changed across restart.')
            return self.state.pending_quantum
        eligible = self.configuration.eligible_model_ids(elapsed_seconds)
        active_index = eligible.index(self.state.active_model_id)
        required_model_ids = eligible[active_index:]
        pending = PendingProgressiveQuantum(
            target_global_optimizer_steps=source_optimizer_steps + optimizer_steps_per_quantum,
            replay_batch=replay_batch,
            required_model_ids=required_model_ids,
        )
        self.state = self.state.model_copy(update={'pending_quantum': pending})
        self.save()
        return pending

    def record_candidate(self, result: CompletedCandidateTraining) -> None:
        pending = self.state.pending_quantum
        if pending is None:
            raise ValueError('No progressive training quantum is pending.')
        if pending.next_model_id != result.model_id:
            raise ValueError('Progressive candidate result is out of training order.')
        pending = pending.model_copy(update={'completed': (*pending.completed, result)})
        self.state = self.state.model_copy(update={'pending_quantum': pending})
        self.save()

    def complete_quantum(self) -> str:
        pending = self.state.pending_quantum
        if pending is None or pending.next_model_id is not None:
            raise ValueError('Every required progressive model must finish before completing the quantum.')
        results = {result.model_id: result for result in pending.completed}
        candidates: list[ProgressiveCandidateState] = []
        for candidate in self.state.candidates:
            result = results.get(candidate.model_id)
            if result is None:
                candidates.append(candidate)
                continue
            ema = candidate.comparable_loss_ema
            updated_ema = (
                ComparableLossEma(value=result.comparable_total_loss, observations=1)
                if ema is None
                else ema.update(result.comparable_total_loss, self.configuration.promotion.decay)
            )
            candidates.append(
                candidate.model_copy(
                    update={
                        'completed_optimizer_steps': result.completed_optimizer_steps,
                        'checkpoint': result.checkpoint,
                        'comparable_loss_ema': updated_ema,
                    }
                )
            )
        active_model_id = self._promoted_model_id(tuple(candidates))
        self.state = ProgressiveTrainingState(
            active_model_id=active_model_id,
            candidates=tuple(candidates),
        )
        self.save()
        return active_model_id

    def save(self) -> None:
        write_text_atomically(self.path, self.state.model_dump_json(indent=2) + '\n')

    def _promoted_model_id(self, candidates: tuple[ProgressiveCandidateState, ...]) -> str:
        successor = self.configuration.successor(self.state.active_model_id)
        if successor is None:
            return self.state.active_model_id
        by_id = {candidate.model_id: candidate for candidate in candidates}
        active = by_id[self.state.active_model_id]
        candidate = by_id[successor.model_id]
        active_ema = active.comparable_loss_ema
        candidate_ema = candidate.comparable_loss_ema
        warmup = self.configuration.promotion.warmup_quanta
        if active_ema is None or candidate_ema is None or candidate_ema.observations < warmup:
            return self.state.active_model_id
        if candidate_ema.value <= active_ema.value * self.configuration.promotion.maximum_relative_loss:
            return successor.model_id
        return self.state.active_model_id

    def _validate_state(self, state: ProgressiveTrainingState) -> None:
        expected_ids = tuple(model.model_id for model in self.configuration.models)
        actual_ids = tuple(candidate.model_id for candidate in state.candidates)
        if actual_ids != expected_ids:
            raise ValueError('Persisted progressive candidates do not match configured model order.')
        if state.active_model_id not in expected_ids:
            raise ValueError('Persisted active progressive model is not configured.')
