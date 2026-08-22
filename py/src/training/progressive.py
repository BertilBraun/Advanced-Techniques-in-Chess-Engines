from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from src.replay.description import ReplayDescription
from src.training.checkpoint import CheckpointReference
from src.training.network import NetworkConfiguration
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel

SECONDS_PER_DAY = 86_400.0


class ProgressiveModelDefinition(FrozenModel):
    model_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    training_start_days: Decimal = Field(ge=0)
    network: NetworkConfiguration

    def is_eligible(self, elapsed_seconds: float) -> bool:
        if elapsed_seconds < 0.0:
            raise ValueError('Elapsed run time cannot be negative.')
        return Decimal(str(elapsed_seconds)) >= self.training_start_days * Decimal(str(SECONDS_PER_DAY))


class TotalLossEmaPromotionConfiguration(FrozenModel):
    decay: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    warmup_quanta: int = Field(gt=0)
    maximum_relative_loss: float = Field(default=1.01, ge=1.0, allow_inf_nan=False)


class ProgressiveModelSizingConfiguration(FrozenModel):
    models: tuple[ProgressiveModelDefinition, ...] = Field(min_length=1)
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

    @property
    def is_progressive(self) -> bool:
        return len(self.models) > 1

    def eligible_model_ids(self, elapsed_seconds: float) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models if model.is_eligible(elapsed_seconds))

    def successor(self, model_id: str) -> ProgressiveModelDefinition | None:
        for index, model in enumerate(self.models):
            if model.model_id == model_id:
                return self.models[index + 1] if index + 1 < len(self.models) else None
        raise ValueError(f'Unknown progressive model ID: {model_id}')


class ComparableLossEma(FrozenModel):
    value: float = Field(ge=0.0, allow_inf_nan=False)
    observations: int = Field(gt=0)

    def update(self, loss: float, decay: float) -> ComparableLossEma:
        if loss < 0.0:
            raise ValueError('Comparable training loss cannot be negative.')
        return ComparableLossEma(
            value=decay * self.value + (1.0 - decay) * loss,
            observations=self.observations + 1,
        )


class PromotionLossComparison(FrozenModel):
    active_model_id: str
    active_loss_ema: ComparableLossEma
    candidate_loss_ema: ComparableLossEma


class ProgressiveCandidateState(FrozenModel):
    model_id: str
    completed_optimizer_steps: int = Field(default=0, ge=0)
    checkpoint: CheckpointReference | None = None
    training_loss_ema: ComparableLossEma | None = None
    promotion_comparison: PromotionLossComparison | None = None


class ReplayBatchIdentity(FrozenModel):
    source_optimizer_steps: int = Field(ge=0)
    replay: ReplayDescription


class CompletedCandidateTraining(FrozenModel):
    model_id: str
    completed_optimizer_steps: int = Field(gt=0)
    checkpoint: CheckpointReference
    comparable_total_loss: float = Field(ge=0.0, allow_inf_nan=False)


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
        replay_batch = ReplayBatchIdentity(replay=replay, source_optimizer_steps=source_optimizer_steps)
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
        self.state = self.state.validated_copy(update={'pending_quantum': pending})
        self.save()
        return pending

    def initialize_candidate(
        self,
        model_id: str,
        completed_optimizer_steps: int,
        checkpoint: CheckpointReference,
    ) -> None:
        candidate = self.candidate(model_id)
        if candidate.checkpoint is not None:
            if candidate.completed_optimizer_steps != completed_optimizer_steps or candidate.checkpoint != checkpoint:
                raise ValueError('Progressive candidate initialization disagrees with persisted state.')
            return
        candidates = tuple(
            item.validated_copy(
                update={
                    'completed_optimizer_steps': completed_optimizer_steps,
                    'checkpoint': checkpoint,
                }
            )
            if item.model_id == model_id
            else item
            for item in self.state.candidates
        )
        self.state = self.state.validated_copy(update={'candidates': candidates})
        self.save()

    def record_candidate(self, result: CompletedCandidateTraining) -> None:
        pending = self.state.pending_quantum
        if pending is None:
            raise ValueError('No progressive training quantum is pending.')
        if pending.next_model_id != result.model_id:
            raise ValueError('Progressive candidate result is out of training order.')
        candidate = self.candidate(result.model_id)
        quantum_steps = pending.target_global_optimizer_steps - pending.replay_batch.source_optimizer_steps
        if result.completed_optimizer_steps != candidate.completed_optimizer_steps + quantum_steps:
            raise ValueError('Progressive candidate result must advance exactly one optimizer quantum.')
        if result.completed_optimizer_steps % quantum_steps:
            raise ValueError('Progressive candidate optimizer progress must align with complete quanta.')
        if result.checkpoint.generation != result.completed_optimizer_steps // quantum_steps:
            raise ValueError('Progressive candidate checkpoint generation disagrees with optimizer progress.')
        pending = pending.validated_copy(update={'completed': (*pending.completed, result)})
        self.state = self.state.validated_copy(update={'pending_quantum': pending})
        self.save()

    def complete_quantum(self) -> str:
        pending = self.state.pending_quantum
        if pending is None or pending.next_model_id is not None:
            raise ValueError('Every required progressive model must finish before completing the quantum.')
        candidates = self._completed_candidate_states(pending)
        active_model_id = self._promoted_model_id(candidates)
        self.state = ProgressiveTrainingState(
            active_model_id=active_model_id,
            candidates=candidates,
        )
        self.save()
        return active_model_id

    def preview_active_model_id(self) -> str:
        pending = self.state.pending_quantum
        if pending is None or pending.next_model_id is not None:
            raise ValueError('Every required progressive model must finish before selecting publication.')
        return self._promoted_model_id(self._completed_candidate_states(pending))

    def completed_result(self, model_id: str) -> CompletedCandidateTraining:
        pending = self.state.pending_quantum
        if pending is None:
            raise ValueError('No progressive training quantum is pending.')
        for result in pending.completed:
            if result.model_id == model_id:
                return result
        raise ValueError(f'Progressive model did not train in the pending quantum: {model_id}')

    def candidate(self, model_id: str) -> ProgressiveCandidateState:
        for candidate in self.state.candidates:
            if candidate.model_id == model_id:
                return candidate
        raise ValueError(f'Unknown progressive model ID: {model_id}')

    def _completed_candidate_states(
        self,
        pending: PendingProgressiveQuantum,
    ) -> tuple[ProgressiveCandidateState, ...]:
        results = {result.model_id: result for result in pending.completed}
        candidates: list[ProgressiveCandidateState] = []
        for candidate in self.state.candidates:
            result = results.get(candidate.model_id)
            if result is None:
                candidates.append(candidate)
                continue
            ema = candidate.training_loss_ema
            updated_ema = (
                ComparableLossEma(value=result.comparable_total_loss, observations=1)
                if ema is None
                else ema.update(result.comparable_total_loss, self.configuration.promotion.decay)
            )
            candidates.append(
                candidate.validated_copy(
                    update={
                        'completed_optimizer_steps': result.completed_optimizer_steps,
                        'checkpoint': result.checkpoint,
                        'training_loss_ema': updated_ema,
                    }
                )
            )
        completed_candidates = tuple(candidates)
        successor = self.configuration.successor(self.state.active_model_id)
        if successor is None or successor.model_id not in results:
            return completed_candidates
        active_loss = results[self.state.active_model_id].comparable_total_loss
        candidate_loss = results[successor.model_id].comparable_total_loss
        successor_state = next(item for item in completed_candidates if item.model_id == successor.model_id)
        comparison = successor_state.promotion_comparison
        if comparison is None or comparison.active_model_id != self.state.active_model_id:
            comparison = PromotionLossComparison(
                active_model_id=self.state.active_model_id,
                active_loss_ema=ComparableLossEma(value=active_loss, observations=1),
                candidate_loss_ema=ComparableLossEma(value=candidate_loss, observations=1),
            )
        else:
            comparison = comparison.validated_copy(
                update={
                    'active_loss_ema': comparison.active_loss_ema.update(
                        active_loss,
                        self.configuration.promotion.decay,
                    ),
                    'candidate_loss_ema': comparison.candidate_loss_ema.update(
                        candidate_loss,
                        self.configuration.promotion.decay,
                    ),
                }
            )
        return tuple(
            item.validated_copy(update={'promotion_comparison': comparison})
            if item.model_id == successor.model_id
            else item
            for item in completed_candidates
        )

    def save(self) -> None:
        write_text_atomically(self.path, self.state.model_dump_json(indent=2) + '\n')

    def _promoted_model_id(self, candidates: tuple[ProgressiveCandidateState, ...]) -> str:
        successor = self.configuration.successor(self.state.active_model_id)
        if successor is None:
            return self.state.active_model_id
        by_id = {candidate.model_id: candidate for candidate in candidates}
        candidate = by_id[successor.model_id]
        comparison = candidate.promotion_comparison
        warmup = self.configuration.promotion.warmup_quanta
        if comparison is None or comparison.candidate_loss_ema.observations < warmup:
            return self.state.active_model_id
        if (
            comparison.candidate_loss_ema.value
            <= comparison.active_loss_ema.value * self.configuration.promotion.maximum_relative_loss
        ):
            return successor.model_id
        return self.state.active_model_id

    def _validate_state(self, state: ProgressiveTrainingState) -> None:
        expected_ids = tuple(model.model_id for model in self.configuration.models)
        actual_ids = tuple(candidate.model_id for candidate in state.candidates)
        if actual_ids != expected_ids:
            raise ValueError('Persisted progressive candidates do not match configured model order.')
        if state.active_model_id not in expected_ids:
            raise ValueError('Persisted active progressive model is not configured.')


def retain_progressive_candidate_checkpoints(run_path: Path, state: ProgressiveTrainingState) -> None:
    models_path = run_path / 'models'
    if not models_path.is_dir():
        return
    retained_paths: set[Path] = set()
    for candidate in state.candidates:
        if candidate.checkpoint is not None:
            retained_paths.update(_checkpoint_paths(candidate.checkpoint))
    if state.pending_quantum is not None:
        for completed in state.pending_quantum.completed:
            retained_paths.update(_checkpoint_paths(completed.checkpoint))
    for model_path in models_path.iterdir():
        if not model_path.is_dir():
            continue
        for artifact in model_path.iterdir():
            if artifact.is_file() and artifact not in retained_paths:
                if artifact.name.startswith(('checkpoint_', 'model_', 'optimizer_')):
                    artifact.unlink()


def _checkpoint_paths(checkpoint: CheckpointReference) -> tuple[Path, Path, Path, Path]:
    return (
        checkpoint.manifest_path,
        checkpoint.model_path,
        checkpoint.optimizer_path,
        checkpoint.inference_model_path,
    )
