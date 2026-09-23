from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator
from src.evaluation.ladder import CandidateMatchObservation, PrimaryLadderEloObservation
from src.replay.description import ReplayDescription
from src.training.checkpoint import CheckpointReference
from src.training.network import NetworkConfiguration
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import FloatGenerationSchedule

# The decay trades detecting a plateau late against mistaking a lull for one. At 0.95 the average
# spans roughly twenty boundaries, nearly seven hours at a twenty-minute cadence, and is still
# climbing towards a curve that has already flattened, so no run ever detected a plateau: V79
# reported 30 Elo per hour while its curve gained 15.5. At 0.80 it spans about five boundaries and
# opened V89's candidate during a flat stretch that turned out to be noise rather than a ceiling.
# 0.90 spans about ten. The stage thresholds are raised alongside it, because a longer average lags
# further behind a flattening curve and so reports a higher gain for the same plateau.
ELO_EMA_DECAY = 0.90
# The gain is measured across a window of the bias-corrected Elo EMA rather than between
# consecutive boundaries. A single step of the EMA moves one or two Elo, which at a twenty-minute
# cadence is three to six Elo per hour, so the per-step slope straddles any sensible threshold and
# changes sign every few observations; V93 never held a run of five below five Elo per hour even
# while its curve was flat for ten hours. Over six observations the noise averages out: the same
# stretch stayed inside a couple of Elo per two hours without a single crossing.
ELO_PLATEAU_WINDOW_OBSERVATIONS = 6
ELO_PLATEAU_CONFIRMATION_OBSERVATIONS = 2
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 86_400.0


class ProgressiveModelDefinition(FrozenModel):
    model_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    network: NetworkConfiguration


class EloPlateauCandidateStartConfiguration(FrozenModel):
    kind: Literal['elo_plateau']
    minimum_worthwhile_gain_per_hour: float = Field(gt=0.0, allow_inf_nan=False)


class EloPlateauStageConfiguration(FrozenModel):
    candidate_model_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    minimum_worthwhile_gain_per_hour: float = Field(gt=0.0, allow_inf_nan=False)


class StagedEloPlateauCandidateStartConfiguration(FrozenModel):
    kind: Literal['staged_elo_plateau']
    stages: tuple[EloPlateauStageConfiguration, ...] = Field(min_length=1)

    def stage(self, candidate_model_id: str) -> EloPlateauStageConfiguration:
        for stage in self.stages:
            if stage.candidate_model_id == candidate_model_id:
                return stage
        raise ValueError(f'No Elo plateau stage is configured for candidate model: {candidate_model_id}')


class ElapsedCandidateStartConfiguration(FrozenModel):
    kind: Literal['elapsed']
    start_days: tuple[Decimal, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_starts(self) -> ElapsedCandidateStartConfiguration:
        if self.start_days[0] <= 0 or any(
            self.start_days[index] <= self.start_days[index - 1] for index in range(1, len(self.start_days))
        ):
            raise ValueError('Elapsed candidate starts must be positive and strictly increasing.')
        return self


CandidateStartConfiguration: TypeAlias = Annotated[
    EloPlateauCandidateStartConfiguration
    | StagedEloPlateauCandidateStartConfiguration
    | ElapsedCandidateStartConfiguration,
    Field(discriminator='kind'),
]


class CandidateMatchGateConfiguration(FrozenModel):
    """What the candidate must prove over the board before it replaces the active model.

    Training loss cannot decide this: a candidate trained at a step multiplier sees each replay
    sample more often than the model it must overtake, so its loss is lower at equal strength.
    """

    definition_id: str = Field(min_length=1)
    minimum_score: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    consecutive_evaluations: int = Field(gt=0)


class PromotionConfiguration(FrozenModel):
    candidate_match_gate: CandidateMatchGateConfiguration
    # A schedule here runs on the candidate's own clock: its generation zero is the quantum it began
    # training, not the run's. A bare number is still accepted and stays constant, as before.
    candidate_catchup_learning_rate: FloatGenerationSchedule
    # A candidate is outside the credit ledger: credits are consumed once per quantum for the active
    # model alone, so extra candidate quanta cost wall-clock rather than replay credits. A stage that
    # trains from scratch needs far more steps than the model it must overtake, and at one quantum
    # per generation it cannot close that in a useful time.
    candidate_step_multiplier: float = Field(default=1.0, ge=1.0)


class FixedModelSizingConfiguration(FrozenModel):
    kind: Literal['fixed']
    model: ProgressiveModelDefinition

    @property
    def models(self) -> tuple[ProgressiveModelDefinition, ...]:
        return (self.model,)

    @property
    def is_progressive(self) -> Literal[False]:
        return False


class ProgressiveModelSizingConfiguration(FrozenModel):
    kind: Literal['progressive']
    models: tuple[ProgressiveModelDefinition, ...] = Field(min_length=2)
    candidate_start: CandidateStartConfiguration
    promotion: PromotionConfiguration

    @model_validator(mode='after')
    def validate_models(self) -> ProgressiveModelSizingConfiguration:
        model_ids = tuple(model.model_id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError('Progressive model IDs must be unique.')
        match self.candidate_start:
            case ElapsedCandidateStartConfiguration(start_days=start_days):
                if len(start_days) != len(self.models) - 1:
                    raise ValueError('Elapsed candidate starts must contain one entry per candidate model.')
            case StagedEloPlateauCandidateStartConfiguration(stages=stages):
                expected_candidate_ids = model_ids[1:]
                actual_candidate_ids = tuple(stage.candidate_model_id for stage in stages)
                if actual_candidate_ids != expected_candidate_ids:
                    raise ValueError('Staged Elo plateau entries must match candidate model order exactly.')
            case EloPlateauCandidateStartConfiguration():
                pass
        return self

    def model(self, model_id: str) -> ProgressiveModelDefinition:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise ValueError(f'Unknown progressive model ID: {model_id}')

    @property
    def is_progressive(self) -> Literal[True]:
        return True

    def successor(self, model_id: str) -> ProgressiveModelDefinition | None:
        for index, model in enumerate(self.models):
            if model.model_id == model_id:
                return self.models[index + 1] if index + 1 < len(self.models) else None
        raise ValueError(f'Unknown progressive model ID: {model_id}')


ModelSizingConfiguration: TypeAlias = Annotated[
    FixedModelSizingConfiguration | ProgressiveModelSizingConfiguration,
    Field(discriminator='kind'),
]


def candidate_quanta_at(generation: int, multiplier: float) -> int:
    """How many quanta a candidate trains during the run's given generation.

    A quantum is indivisible, so a fractional multiplier alternates: at 1.5 the candidate trains one
    quantum then two, averaging exactly 1.5. The clock must be the run's generation, which advances
    by one per call; reading the candidate's own generation makes the index advance by whatever this
    returned and settle on a fixed point at the larger step count.
    """
    return math.floor((generation + 1) * multiplier) - math.floor(generation * multiplier)


class ProgressiveCandidateState(FrozenModel):
    model_id: str
    completed_optimizer_steps: int = Field(default=0, ge=0)
    checkpoint: CheckpointReference | None = None


class EloEmaSample(FrozenModel):
    boundary_seconds: int = Field(ge=0)
    ema_elo: float = Field(ge=0.0, allow_inf_nan=False)


class EloPlateauCandidateStartState(FrozenModel):
    kind: Literal['elo_plateau']
    latest_boundary_seconds: int = Field(ge=0)
    ema_observations: int = Field(ge=0)
    ema_elo: float = Field(ge=0.0, allow_inf_nan=False)
    instantaneous_ema_gain_per_hour: float | None = Field(default=None, allow_inf_nan=False)
    recent_ema_samples: tuple[EloEmaSample, ...] = ()
    consecutive_below_threshold_observations: int = Field(ge=0)
    latched: bool

    @model_validator(mode='after')
    def validate_observation_state(self) -> EloPlateauCandidateStartState:
        if self.ema_observations == 0:
            if self.latest_boundary_seconds != 0 or self.ema_elo != 0.0:
                raise ValueError('An empty Elo EMA must remain at its zero baseline.')
            if self.instantaneous_ema_gain_per_hour is not None:
                raise ValueError('An empty Elo EMA cannot have a gain rate.')
            if self.consecutive_below_threshold_observations != 0:
                raise ValueError('An empty Elo EMA cannot have below-threshold observations.')
        elif self.latest_boundary_seconds == 0:
            raise ValueError('An observed Elo EMA must have a positive evaluation boundary.')
        if self.consecutive_below_threshold_observations > self.ema_observations:
            raise ValueError('Below-threshold observations cannot exceed total Elo EMA observations.')
        return self


class StagedEloPlateauCandidateStartState(FrozenModel):
    kind: Literal['staged_elo_plateau']
    candidate_model_id: str
    latest_boundary_seconds: int = Field(ge=0)
    ema_observations: int = Field(ge=0)
    ema_elo: float = Field(ge=0.0, allow_inf_nan=False)
    latest_observed_elo: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    instantaneous_ema_gain_per_hour: float | None = Field(default=None, allow_inf_nan=False)
    recent_ema_samples: tuple[EloEmaSample, ...] = ()
    consecutive_below_threshold_observations: int = Field(ge=0)
    latched: bool

    @model_validator(mode='after')
    def validate_observation_state(self) -> StagedEloPlateauCandidateStartState:
        if self.ema_observations == 0:
            if self.latest_boundary_seconds != 0 or self.ema_elo != 0.0 or self.latest_observed_elo is not None:
                raise ValueError('An empty staged Elo EMA must remain at its zero baseline.')
            if self.instantaneous_ema_gain_per_hour is not None:
                raise ValueError('An empty staged Elo EMA cannot have a gain rate.')
            if self.consecutive_below_threshold_observations != 0:
                raise ValueError('An empty staged Elo EMA cannot have below-threshold observations.')
        elif self.latest_boundary_seconds == 0 or self.latest_observed_elo is None:
            raise ValueError('An observed staged Elo EMA must retain its latest observation and boundary.')
        if self.consecutive_below_threshold_observations > self.ema_observations:
            raise ValueError('Below-threshold observations cannot exceed total Elo EMA observations.')
        return self


class ElapsedCandidateStartState(FrozenModel):
    kind: Literal['elapsed']


CandidateStartState: TypeAlias = Annotated[
    EloPlateauCandidateStartState | StagedEloPlateauCandidateStartState | ElapsedCandidateStartState,
    Field(discriminator='kind'),
]


class ReplayBatchIdentity(FrozenModel):
    source_optimizer_steps: int = Field(ge=0)
    replay: ReplayDescription


class CompletedCandidateTraining(FrozenModel):
    model_id: str
    completed_optimizer_steps: int = Field(gt=0)
    checkpoint: CheckpointReference


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


class CandidateMatchGateState(FrozenModel):
    active_model_id: str
    candidate_model_id: str
    consecutive_passes: int = Field(default=0, ge=0)
    latest_boundary_seconds: int = Field(default=0, ge=0)
    recent_observations: tuple[CandidateMatchObservation, ...] = ()


class ProgressiveTrainingState(FrozenModel):
    schema_version: Literal[5] = 5
    active_model_id: str
    candidates: tuple[ProgressiveCandidateState, ...]
    candidate_start: CandidateStartState
    pending_quantum: PendingProgressiveQuantum | None = None
    match_gate: CandidateMatchGateState | None = None


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
                candidate_start=self._initial_candidate_start_state(),
            )
            self.save()

    def observe_primary_ladder_elos(
        self,
        observations: tuple[PrimaryLadderEloObservation, ...],
    ) -> tuple[EloPlateauCandidateStartState | StagedEloPlateauCandidateStartState, ...]:
        updates: list[EloPlateauCandidateStartState | StagedEloPlateauCandidateStartState] = []
        match self.state.candidate_start, self.configuration.candidate_start:
            case EloPlateauCandidateStartState() as candidate_start, EloPlateauCandidateStartConfiguration() as start:
                threshold = start.minimum_worthwhile_gain_per_hour
            case (
                StagedEloPlateauCandidateStartState() as candidate_start,
                StagedEloPlateauCandidateStartConfiguration() as start,
            ):
                successor = self.configuration.successor(self.state.active_model_id)
                if successor is None:
                    return ()
                if candidate_start.candidate_model_id != successor.model_id:
                    raise ValueError('Staged Elo plateau state does not target the active model successor.')
                threshold = start.stage(successor.model_id).minimum_worthwhile_gain_per_hour
            case ElapsedCandidateStartState(), ElapsedCandidateStartConfiguration():
                return ()
            case _:
                raise ValueError('Persisted candidate-start policy does not match configuration.')
        for observation in sorted(observations, key=lambda item: item.boundary_seconds):
            if observation.boundary_seconds <= candidate_start.latest_boundary_seconds:
                continue
            ema_observations = candidate_start.ema_observations + 1
            previous_weight = 1.0 - ELO_EMA_DECAY**candidate_start.ema_observations
            current_weight = 1.0 - ELO_EMA_DECAY**ema_observations
            ema_elo = (
                ELO_EMA_DECAY * candidate_start.ema_elo * previous_weight + (1.0 - ELO_EMA_DECAY) * observation.elo
            ) / current_weight
            recent_ema_samples = (
                *candidate_start.recent_ema_samples,
                EloEmaSample(boundary_seconds=observation.boundary_seconds, ema_elo=ema_elo),
            )[-(ELO_PLATEAU_WINDOW_OBSERVATIONS + 1) :]
            gain_per_hour = None
            if len(recent_ema_samples) > ELO_PLATEAU_WINDOW_OBSERVATIONS:
                oldest = recent_ema_samples[0]
                window_hours = (observation.boundary_seconds - oldest.boundary_seconds) / SECONDS_PER_HOUR
                gain_per_hour = (ema_elo - oldest.ema_elo) / window_hours
            consecutive_below_threshold_observations = (
                candidate_start.consecutive_below_threshold_observations + 1
                if gain_per_hour is not None and gain_per_hour < threshold
                else 0
            )
            latched = (
                candidate_start.latched
                or consecutive_below_threshold_observations >= ELO_PLATEAU_CONFIRMATION_OBSERVATIONS
            )
            match candidate_start:
                case EloPlateauCandidateStartState():
                    candidate_start = EloPlateauCandidateStartState(
                        kind='elo_plateau',
                        latest_boundary_seconds=observation.boundary_seconds,
                        ema_observations=ema_observations,
                        ema_elo=ema_elo,
                        instantaneous_ema_gain_per_hour=gain_per_hour,
                        recent_ema_samples=recent_ema_samples,
                        consecutive_below_threshold_observations=consecutive_below_threshold_observations,
                        latched=latched,
                    )
                case StagedEloPlateauCandidateStartState(candidate_model_id=candidate_model_id):
                    candidate_start = StagedEloPlateauCandidateStartState(
                        kind='staged_elo_plateau',
                        candidate_model_id=candidate_model_id,
                        latest_boundary_seconds=observation.boundary_seconds,
                        ema_observations=ema_observations,
                        ema_elo=ema_elo,
                        latest_observed_elo=observation.elo,
                        instantaneous_ema_gain_per_hour=gain_per_hour,
                        recent_ema_samples=recent_ema_samples,
                        consecutive_below_threshold_observations=consecutive_below_threshold_observations,
                        latched=latched,
                    )
            updates.append(candidate_start)
        if updates:
            self.state = self.state.validated_copy(update={'candidate_start': candidate_start.model_dump(mode='json')})
            self.save()
        return tuple(updates)

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
        required_model_ids = self._required_model_ids(elapsed_seconds)
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
        # A candidate may train several quanta per generation; the active model always trains one,
        # because its progress is what the credit ledger has paid for.
        expected_quanta = (
            1
            if result.model_id == self.state.active_model_id
            else candidate_quanta_at(
                pending.replay_batch.source_optimizer_steps // quantum_steps,
                self.configuration.promotion.candidate_step_multiplier,
            )
        )
        if result.completed_optimizer_steps != candidate.completed_optimizer_steps + quantum_steps * expected_quanta:
            raise ValueError('Progressive candidate result must advance its configured optimizer quanta.')
        if result.completed_optimizer_steps % quantum_steps:
            raise ValueError('Progressive candidate optimizer progress must align with complete quanta.')
        if result.checkpoint.generation != result.completed_optimizer_steps // quantum_steps:
            raise ValueError('Progressive candidate checkpoint generation disagrees with optimizer progress.')
        pending = pending.validated_copy(update={'completed': (*pending.completed, result)})
        self.state = self.state.validated_copy(update={'pending_quantum': pending})
        self.save()

    def observe_candidate_matches(
        self,
        observations: tuple[CandidateMatchObservation, ...],
    ) -> CandidateMatchGateState | None:
        successor = self.configuration.successor(self.state.active_model_id)
        if successor is None:
            return None
        gate = self.state.match_gate
        if gate is None or gate.active_model_id != self.state.active_model_id:
            gate = CandidateMatchGateState(
                active_model_id=self.state.active_model_id,
                candidate_model_id=successor.model_id,
            )
        minimum_score = self.configuration.promotion.candidate_match_gate.minimum_score
        for observation in sorted(observations, key=lambda item: item.boundary_seconds):
            if observation.boundary_seconds <= gate.latest_boundary_seconds:
                continue
            # A run of passes must be unbroken: one failure is evidence the candidate is not there
            # yet, and starting again is what keeps a lucky pair of results from promoting it.
            passes = gate.consecutive_passes + 1 if observation.score >= minimum_score else 0
            gate = gate.validated_copy(
                update={
                    'consecutive_passes': passes,
                    'latest_boundary_seconds': observation.boundary_seconds,
                    'recent_observations': (*gate.recent_observations, observation)[
                        -self.configuration.promotion.candidate_match_gate.consecutive_evaluations :
                    ],
                }
            )
        if gate == self.state.match_gate:
            return gate
        self.state = self.state.validated_copy(update={'match_gate': gate})
        self.save()
        return gate

    def complete_quantum(self) -> str:
        pending = self.state.pending_quantum
        if pending is None or pending.next_model_id is not None:
            raise ValueError('Every required progressive model must finish before completing the quantum.')
        candidates = self._completed_candidate_states(pending)
        active_model_id = self._promoted_model_id()
        candidate_start = self._candidate_start_after_promotion(active_model_id)
        self.state = ProgressiveTrainingState(
            active_model_id=active_model_id,
            candidates=candidates,
            candidate_start=candidate_start,
            match_gate=None if active_model_id != self.state.active_model_id else self.state.match_gate,
        )
        self.save()
        return active_model_id

    def preview_active_model_id(self) -> str:
        pending = self.state.pending_quantum
        if pending is None or pending.next_model_id is not None:
            raise ValueError('Every required progressive model must finish before selecting publication.')
        return self._promoted_model_id()

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
        return tuple(
            candidate
            if candidate.model_id not in results
            else candidate.validated_copy(
                update={
                    'completed_optimizer_steps': results[candidate.model_id].completed_optimizer_steps,
                    'checkpoint': results[candidate.model_id].checkpoint,
                }
            )
            for candidate in self.state.candidates
        )

    def save(self) -> None:
        write_text_atomically(self.path, self.state.model_dump_json(indent=2) + '\n')

    def _promoted_model_id(self) -> str:
        successor = self.configuration.successor(self.state.active_model_id)
        if successor is None:
            return self.state.active_model_id
        gate = self.state.match_gate
        if gate is None or gate.active_model_id != self.state.active_model_id:
            return self.state.active_model_id
        if gate.candidate_model_id != successor.model_id:
            return self.state.active_model_id
        if gate.consecutive_passes < self.configuration.promotion.candidate_match_gate.consecutive_evaluations:
            return self.state.active_model_id
        return successor.model_id

    def _validate_state(self, state: ProgressiveTrainingState) -> None:
        expected_ids = tuple(model.model_id for model in self.configuration.models)
        actual_ids = tuple(candidate.model_id for candidate in state.candidates)
        if actual_ids != expected_ids:
            raise ValueError('Persisted progressive candidates do not match configured model order.')
        if state.active_model_id not in expected_ids:
            raise ValueError('Persisted active progressive model is not configured.')
        if state.candidate_start.kind != self.configuration.candidate_start.kind:
            raise ValueError('Persisted candidate-start policy does not match configuration.')
        match state.candidate_start:
            case StagedEloPlateauCandidateStartState(candidate_model_id=candidate_model_id):
                successor = self.configuration.successor(state.active_model_id)
                expected_candidate_id = state.active_model_id if successor is None else successor.model_id
                if candidate_model_id != expected_candidate_id:
                    raise ValueError('Persisted staged Elo plateau target does not match the active model successor.')
            case _:
                pass

    def _initial_candidate_start_state(self) -> CandidateStartState:
        match self.configuration.candidate_start:
            case EloPlateauCandidateStartConfiguration():
                return EloPlateauCandidateStartState(
                    kind='elo_plateau',
                    latest_boundary_seconds=0,
                    ema_observations=0,
                    ema_elo=0.0,
                    consecutive_below_threshold_observations=0,
                    latched=False,
                )
            case StagedEloPlateauCandidateStartConfiguration():
                return StagedEloPlateauCandidateStartState(
                    kind='staged_elo_plateau',
                    candidate_model_id=self.configuration.models[1].model_id,
                    latest_boundary_seconds=0,
                    ema_observations=0,
                    ema_elo=0.0,
                    consecutive_below_threshold_observations=0,
                    latched=False,
                )
            case ElapsedCandidateStartConfiguration():
                return ElapsedCandidateStartState(kind='elapsed')

    def _candidate_start_after_promotion(self, active_model_id: str) -> CandidateStartState:
        candidate_start = self.state.candidate_start
        if active_model_id == self.state.active_model_id:
            return candidate_start
        match candidate_start:
            case StagedEloPlateauCandidateStartState():
                successor = self.configuration.successor(active_model_id)
                if successor is None:
                    return candidate_start.validated_copy(
                        update={'candidate_model_id': active_model_id, 'latched': False}
                    )
                if candidate_start.latest_observed_elo is None:
                    return StagedEloPlateauCandidateStartState(
                        kind='staged_elo_plateau',
                        candidate_model_id=successor.model_id,
                        latest_boundary_seconds=0,
                        ema_observations=0,
                        ema_elo=0.0,
                        consecutive_below_threshold_observations=0,
                        latched=False,
                    )
                return StagedEloPlateauCandidateStartState(
                    kind='staged_elo_plateau',
                    candidate_model_id=successor.model_id,
                    latest_boundary_seconds=candidate_start.latest_boundary_seconds,
                    ema_observations=1,
                    ema_elo=candidate_start.latest_observed_elo,
                    latest_observed_elo=candidate_start.latest_observed_elo,
                    consecutive_below_threshold_observations=0,
                    latched=False,
                )
            case _:
                return candidate_start

    def _required_model_ids(self, elapsed_seconds: float) -> tuple[str, ...]:
        if elapsed_seconds < 0.0:
            raise ValueError('Elapsed run time cannot be negative.')
        match self.configuration.candidate_start, self.state.candidate_start:
            case EloPlateauCandidateStartConfiguration(), EloPlateauCandidateStartState() as state:
                required_model_ids = (self.state.active_model_id,)
                successor = self.configuration.successor(self.state.active_model_id)
                if state.latched and successor is not None:
                    required_model_ids = (*required_model_ids, successor.model_id)
                return required_model_ids
            case (
                StagedEloPlateauCandidateStartConfiguration(),
                StagedEloPlateauCandidateStartState() as state,
            ):
                required_model_ids = (self.state.active_model_id,)
                successor = self.configuration.successor(self.state.active_model_id)
                if state.latched and successor is not None:
                    required_model_ids = (*required_model_ids, successor.model_id)
                return required_model_ids
            case ElapsedCandidateStartConfiguration(start_days=start_days), ElapsedCandidateStartState():
                elapsed_days = Decimal(str(elapsed_seconds)) / Decimal(str(SECONDS_PER_DAY))
                eligible_model_ids = (self.configuration.models[0].model_id,) + tuple(
                    model.model_id
                    for model, start_day in zip(self.configuration.models[1:], start_days, strict=True)
                    if elapsed_days >= start_day
                )
                active_index = eligible_model_ids.index(self.state.active_model_id)
                return eligible_model_ids[active_index:]
            case _:
                raise ValueError('Persisted candidate-start policy does not match configuration.')


def retain_progressive_candidate_checkpoints(
    run_path: Path,
    state: ProgressiveTrainingState,
    pinned: tuple[CheckpointReference, ...] = (),
) -> None:
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
    # A promotion match runs in another process against a candidate checkpoint the next generation
    # would otherwise delete underneath it.
    for checkpoint in pinned:
        retained_paths.update(_checkpoint_paths(checkpoint))
    for model_path in models_path.iterdir():
        if not model_path.is_dir():
            continue
        for artifact in model_path.iterdir():
            if artifact.is_file() and artifact not in retained_paths:
                if artifact.name.startswith(('checkpoint_', 'model_', 'optimizer_', 'qat_state_')):
                    artifact.unlink()


def _checkpoint_paths(checkpoint: CheckpointReference) -> tuple[Path, ...]:
    paths = (
        checkpoint.manifest_path,
        checkpoint.model_path,
        checkpoint.optimizer_path,
        checkpoint.inference_model_path,
    )
    if checkpoint.qat_state is not None:
        return (*paths, checkpoint.qat_state.path)
    return paths
