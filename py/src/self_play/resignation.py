from __future__ import annotations

import hashlib
import math
import random
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ResignationTerminationReason(str, Enum):
    NATURAL = 'natural'
    RESIGNATION = 'resignation'
    PLY_CAP = 'ply_cap'
    LOW_MATERIAL = 'low_material'


@dataclass(frozen=True)
class ResignationParams:
    audit_enabled: bool = False
    production_enabled: bool = False
    audit_game_probability: float = 0.10
    audit_cutoff_threshold: float | None = None
    audit_cutoff_increase_per_model: float = 0.0
    minimum_eligible_ply: int = 0
    minimum_published_model_version: int = 30
    minimum_completed_audit_triggers: int = 100
    false_non_loss_upper_bound: float = 0.03
    confidence_level: float = 0.95
    calibration_window_size: int = 2_000
    calibration_interval: int = 100
    threshold_candidate_minimum: float = -0.99
    threshold_candidate_maximum: float = -0.80
    threshold_candidate_resolution: float = 0.01
    maximum_threshold_change_per_calibration: float = 0.01
    production_resignation_enable_fraction: float = 1.0
    production_resignation_fade_versions: int = 30
    require_external_safety_approval: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.audit_game_probability <= 1.0:
            raise ValueError('Audit-game probability must be in [0, 1].')
        if self.audit_cutoff_threshold is not None and not -1.0 <= self.audit_cutoff_threshold < 0.0:
            raise ValueError('Audit cutoff threshold must be in [-1, 0).')
        if self.audit_cutoff_increase_per_model < 0.0:
            raise ValueError('Audit cutoff increase per model cannot be negative.')
        if self.audit_cutoff_increase_per_model > 0.0 and self.audit_cutoff_threshold is None:
            raise ValueError('Audit cutoff increase requires a starting cutoff threshold.')
        if self.minimum_eligible_ply < 0:
            raise ValueError('Minimum eligible ply cannot be negative.')
        if self.minimum_published_model_version < 0:
            raise ValueError('Minimum published model version cannot be negative.')
        if self.minimum_completed_audit_triggers <= 0:
            raise ValueError('Minimum completed audit triggers must be positive.')
        if not 0.0 < self.false_non_loss_upper_bound < 1.0:
            raise ValueError('False-non-loss upper bound must be in (0, 1).')
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError('Confidence level must be in (0, 1).')
        if self.calibration_window_size < self.minimum_completed_audit_triggers:
            raise ValueError('Calibration window must contain the minimum completed audit triggers.')
        if self.calibration_interval <= 0:
            raise ValueError('Calibration interval must be positive.')
        if not -1.0 <= self.threshold_candidate_minimum < self.threshold_candidate_maximum < 0.0:
            raise ValueError('Threshold candidates must be an increasing negative interval within [-1, 0).')
        if self.threshold_candidate_resolution <= 0.0:
            raise ValueError('Threshold candidate resolution must be positive.')
        if self.maximum_threshold_change_per_calibration <= 0.0:
            raise ValueError('Maximum threshold change must be positive.')
        if not 0.0 <= self.production_resignation_enable_fraction <= 1.0:
            raise ValueError('Production resignation fraction must be in [0, 1].')
        if self.production_resignation_fade_versions < 0:
            raise ValueError('Production resignation fade length cannot be negative.')


class ResignationObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model_version: int = Field(ge=0)
    ply: int = Field(ge=0)
    side_to_move: Literal[-1, 1]
    root_value: float = Field(ge=-1.0, le=1.0)
    best_child_value: float = Field(ge=-1.0, le=1.0)


class CompletedResignationAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    game_id: str
    observations: tuple[ResignationObservation, ...]
    audit_cutoff_threshold: float | None
    audit_cutoff_ply: int | None
    final_current_player: Literal[-1, 1]
    game_outcome: float = Field(ge=-1.0, le=1.0)
    termination_reason: ResignationTerminationReason


class ThresholdStatistics(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    threshold: float
    completed_triggers: int
    recovered_wins: int
    recovered_draws: int
    false_non_loss_rate: float
    false_non_loss_upper_confidence: float


class ResignationCalibrationState(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int = 1
    configuration_fingerprint: str
    selected_threshold: float | None = None
    selected_threshold_is_safe: bool = False
    total_completed_trigger_count: int = 0
    completed_trigger_count_at_last_calibration: int = 0
    total_completed_audit_count: int = 0
    completed_audit_count_at_last_calibration: int = 0
    audit_record_count: int = 0
    threshold_statistics: tuple[ThresholdStatistics, ...] = ()


@dataclass(frozen=True)
class ResignationAssignment:
    is_audit_game: bool
    production_resignation_enabled: bool
    governing_threshold: float | None


class ExternalResignationSafetyApproval(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    approved: bool
    approved_through_model_version: int = Field(ge=0)
    reason: str = Field(min_length=1)


class MCTSChildStatistics(Protocol):
    visits: int
    result_sum: float


def best_child_value_from_root_perspective(
    children: Sequence[MCTSChildStatistics],
) -> float | None:
    visited_children = [child for child in children if child.visits > 0]
    if not visited_children:
        return None
    best_child = max(visited_children, key=lambda child: child.visits)
    child_perspective_value = best_child.result_sum / best_child.visits
    return max(-1.0, min(1.0, -child_perspective_value))


def one_sided_binomial_upper_bound(
    false_non_losses: int,
    completed_triggers: int,
    confidence_level: float,
) -> float:
    if completed_triggers <= 0:
        return 1.0
    if not 0 <= false_non_losses <= completed_triggers:
        raise ValueError('False non-losses must be between zero and the completed-trigger count.')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError('Confidence level must be in (0, 1).')
    if false_non_losses == completed_triggers:
        return 1.0

    target_cumulative_probability = 1.0 - confidence_level
    lower = false_non_losses / completed_triggers
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        cumulative_probability = _binomial_cumulative_probability(
            false_non_losses,
            completed_triggers,
            midpoint,
        )
        if cumulative_probability > target_cumulative_probability:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _binomial_cumulative_probability(maximum_failures: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0

    logarithms = [
        math.lgamma(trials + 1)
        - math.lgamma(failures + 1)
        - math.lgamma(trials - failures + 1)
        + failures * math.log(probability)
        + (trials - failures) * math.log1p(-probability)
        for failures in range(maximum_failures + 1)
    ]
    maximum_logarithm = max(logarithms)
    return math.exp(maximum_logarithm) * sum(math.exp(logarithm - maximum_logarithm) for logarithm in logarithms)


class ResignationManager:
    def __init__(self, save_path: str, parameters: ResignationParams) -> None:
        self.parameters = parameters
        self._state_path = Path(save_path) / 'resignation' / 'calibration.json'
        self._audit_database_path = self._state_path.with_name('completed-audits.sqlite3')
        self._external_approval_path = self._state_path.with_name('external-safety-approval.json')
        self._lock = FileLock(str(self._state_path.with_suffix('.lock')))
        self._configuration_fingerprint = _configuration_fingerprint(parameters)
        self._state = self._load_state()
        if self._audit_database_path.exists():
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                loaded_state = self._load_state()
                recovered_state = self._recover_interrupted_database_commit(loaded_state)
                if recovered_state != loaded_state:
                    self._write_state(recovered_state)
                self._state = recovered_state

    @property
    def state(self) -> ResignationCalibrationState:
        return self._state

    def assignment(self, model_version: int) -> ResignationAssignment:
        if not self.parameters.audit_enabled and not self.parameters.production_enabled:
            return ResignationAssignment(False, False, None)

        self._state = self._load_state()
        is_audit_game = (
            self.parameters.audit_enabled
            and model_version >= self.parameters.minimum_published_model_version
            and random.random() < self.parameters.audit_game_probability
        )
        if is_audit_game:
            return ResignationAssignment(True, False, self._audit_threshold(model_version))

        production_fraction = self._production_fraction(model_version)
        bootstrap_enabled = self.parameters.production_enabled and not self._state.threshold_statistics
        calibrated_enabled = (
            self.parameters.production_enabled
            and self._state.selected_threshold_is_safe
            and self._state.selected_threshold is not None
        )
        production_enabled = (
            (bootstrap_enabled or calibrated_enabled)
            and self._external_safety_gate_is_open(model_version)
            and random.random() < production_fraction
        )
        governing_threshold = (
            self._audit_threshold(model_version)
            if production_enabled and bootstrap_enabled
            else self._state.selected_threshold
            if production_enabled
            else None
        )
        return ResignationAssignment(
            False,
            production_enabled,
            governing_threshold,
        )

    def _audit_threshold(self, model_version: int) -> float | None:
        if self._state.selected_threshold is not None:
            return self._state.selected_threshold
        starting_threshold = self.parameters.audit_cutoff_threshold
        if starting_threshold is None:
            return None
        elapsed_models = model_version - self.parameters.minimum_published_model_version
        scheduled_threshold = starting_threshold + elapsed_models * self.parameters.audit_cutoff_increase_per_model
        return min(self.parameters.threshold_candidate_maximum, scheduled_threshold)

    def record_completed_audit(
        self,
        audit: CompletedResignationAudit,
    ) -> ResignationCalibrationState:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            state = self._recover_interrupted_database_commit(self._load_state())
            inserted = self._insert_completed_audit(audit)
            if not inserted:
                self._write_state(state)
                self._state = state
                return state
            total_audit_count = state.total_completed_audit_count + 1
            total_trigger_count = state.total_completed_trigger_count + int(
                audit.termination_reason is ResignationTerminationReason.NATURAL
                and _first_trigger(audit, self.parameters.threshold_candidate_maximum) is not None
            )
            should_calibrate = (
                total_trigger_count - state.completed_trigger_count_at_last_calibration
                >= self.parameters.calibration_interval
                or total_audit_count - state.completed_audit_count_at_last_calibration
                >= self.parameters.calibration_interval
            )
            if should_calibrate:
                completed_audits = self._load_recent_completed_audits()
                state = self._calibrate(
                    state,
                    completed_audits,
                    total_trigger_count,
                    total_audit_count,
                )
            else:
                state = state.model_copy(
                    update={
                        'total_completed_trigger_count': total_trigger_count,
                        'total_completed_audit_count': total_audit_count,
                        'audit_record_count': total_audit_count,
                    }
                )
            self._write_state(state)
            self._state = state
            return state

    def _calibrate(
        self,
        previous_state: ResignationCalibrationState,
        completed_audits: tuple[CompletedResignationAudit, ...],
        trigger_count: int,
        audit_count: int,
    ) -> ResignationCalibrationState:
        statistics = tuple(
            _threshold_statistics(completed_audits, threshold, self.parameters.confidence_level)
            for threshold in _threshold_candidates(self.parameters)
        )
        safe_candidates = [
            candidate
            for candidate in statistics
            if candidate.completed_triggers >= self.parameters.minimum_completed_audit_triggers
            and candidate.false_non_loss_upper_confidence <= self.parameters.false_non_loss_upper_bound
        ]
        selected_target = max((candidate.threshold for candidate in safe_candidates), default=None)
        selected_threshold = _rate_limited_threshold(
            previous_state.selected_threshold,
            selected_target,
            self.parameters,
        )
        selected_statistics = next(
            (candidate for candidate in statistics if candidate.threshold == selected_threshold),
            None,
        )
        selected_is_safe = (
            selected_statistics is not None
            and selected_statistics.completed_triggers >= self.parameters.minimum_completed_audit_triggers
            and selected_statistics.false_non_loss_upper_confidence <= self.parameters.false_non_loss_upper_bound
        )
        return ResignationCalibrationState(
            configuration_fingerprint=self._configuration_fingerprint,
            selected_threshold=selected_threshold,
            selected_threshold_is_safe=selected_is_safe,
            total_completed_trigger_count=trigger_count,
            completed_trigger_count_at_last_calibration=trigger_count,
            total_completed_audit_count=audit_count,
            completed_audit_count_at_last_calibration=audit_count,
            audit_record_count=audit_count,
            threshold_statistics=statistics,
        )

    def _production_fraction(self, model_version: int) -> float:
        if model_version < self.parameters.minimum_published_model_version:
            return 0.0
        fade_versions = self.parameters.production_resignation_fade_versions
        if fade_versions == 0:
            return self.parameters.production_resignation_enable_fraction
        elapsed_versions = model_version - self.parameters.minimum_published_model_version + 1
        fade_progress = min(1.0, elapsed_versions / fade_versions)
        return self.parameters.production_resignation_enable_fraction * fade_progress

    def _external_safety_gate_is_open(self, model_version: int) -> bool:
        if not self.parameters.require_external_safety_approval:
            return True
        if not self._external_approval_path.exists():
            return False
        try:
            approval = ExternalResignationSafetyApproval.model_validate_json(
                self._external_approval_path.read_text(encoding='utf-8')
            )
        except (OSError, ValidationError):
            return False
        return approval.approved and model_version <= approval.approved_through_model_version

    def _load_state(self) -> ResignationCalibrationState:
        if not self._state_path.exists():
            return ResignationCalibrationState(configuration_fingerprint=self._configuration_fingerprint)
        state = ResignationCalibrationState.model_validate_json(self._state_path.read_text(encoding='utf-8'))
        if state.configuration_fingerprint != self._configuration_fingerprint:
            raise ValueError('Resignation calibration state belongs to a different configuration.')
        return state

    def _write_state(self, state: ResignationCalibrationState) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._state_path.with_suffix('.json.tmp')
        temporary_path.write_text(state.model_dump_json(indent=2) + '\n', encoding='utf-8')
        temporary_path.replace(self._state_path)

    def _insert_completed_audit(self, audit: CompletedResignationAudit) -> bool:
        with closing(self._open_audit_database()) as audit_database, audit_database:
            is_natural_trigger = int(
                audit.termination_reason is ResignationTerminationReason.NATURAL
                and _first_trigger(audit, self.parameters.threshold_candidate_maximum) is not None
            )
            cursor = audit_database.execute(
                """
                INSERT OR IGNORE INTO completed_audits (
                    game_id,
                    payload,
                    is_natural_maximum_threshold_trigger
                ) VALUES (?, ?, ?)
                """,
                (
                    audit.game_id,
                    audit.model_dump_json(),
                    is_natural_trigger,
                ),
            )
            if cursor.rowcount == 1:
                audit_database.execute(
                    """
                    UPDATE audit_metadata
                    SET
                        record_count = record_count + 1,
                        natural_trigger_count = natural_trigger_count + ?
                    WHERE singleton = 1
                    """,
                    (is_natural_trigger,),
                )
            return cursor.rowcount == 1

    def _load_recent_completed_audits(self) -> tuple[CompletedResignationAudit, ...]:
        with closing(self._open_audit_database()) as audit_database:
            rows = audit_database.execute(
                """
                SELECT payload
                FROM completed_audits
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (self.parameters.calibration_window_size,),
            ).fetchall()
        return tuple(CompletedResignationAudit.model_validate_json(payload) for (payload,) in reversed(rows))

    def _recover_interrupted_database_commit(
        self,
        state: ResignationCalibrationState,
    ) -> ResignationCalibrationState:
        with closing(self._open_audit_database()) as audit_database:
            audit_count, trigger_count = audit_database.execute(
                """
                SELECT record_count, natural_trigger_count
                FROM audit_metadata
                WHERE singleton = 1
                """
            ).fetchone()
        if audit_count == state.audit_record_count:
            return state

        completed_audits = self._load_recent_completed_audits()
        if (
            trigger_count - state.completed_trigger_count_at_last_calibration >= self.parameters.calibration_interval
            or audit_count - state.completed_audit_count_at_last_calibration >= self.parameters.calibration_interval
        ):
            return self._calibrate(
                state,
                completed_audits,
                trigger_count,
                audit_count,
            )
        return state.model_copy(
            update={
                'total_completed_trigger_count': trigger_count,
                'total_completed_audit_count': audit_count,
                'audit_record_count': audit_count,
            }
        )

    def _open_audit_database(self) -> sqlite3.Connection:
        audit_database = sqlite3.connect(self._audit_database_path)
        audit_database.execute(
            """
            CREATE TABLE IF NOT EXISTS completed_audits (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                is_natural_maximum_threshold_trigger INTEGER NOT NULL
            )
            """
        )
        audit_database.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                record_count INTEGER NOT NULL,
                natural_trigger_count INTEGER NOT NULL
            )
            """
        )
        audit_database.execute(
            """
            INSERT OR IGNORE INTO audit_metadata (
                singleton,
                record_count,
                natural_trigger_count
            ) VALUES (1, 0, 0)
            """
        )
        audit_database.commit()
        return audit_database


def _configuration_fingerprint(parameters: ResignationParams) -> str:
    calibration_configuration = (
        parameters.audit_cutoff_threshold,
        parameters.audit_cutoff_increase_per_model,
        parameters.minimum_published_model_version,
        parameters.minimum_eligible_ply,
        parameters.minimum_completed_audit_triggers,
        parameters.false_non_loss_upper_bound,
        parameters.confidence_level,
        parameters.calibration_window_size,
        parameters.calibration_interval,
        parameters.threshold_candidate_minimum,
        parameters.threshold_candidate_maximum,
        parameters.threshold_candidate_resolution,
        parameters.maximum_threshold_change_per_calibration,
    )
    encoded = repr(calibration_configuration).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _threshold_candidates(parameters: ResignationParams) -> tuple[float, ...]:
    span = parameters.threshold_candidate_maximum - parameters.threshold_candidate_minimum
    candidate_count = int(math.floor(span / parameters.threshold_candidate_resolution + 1e-9))
    return tuple(
        round(
            parameters.threshold_candidate_minimum + index * parameters.threshold_candidate_resolution,
            8,
        )
        for index in range(candidate_count + 1)
    )


def _first_trigger(
    audit: CompletedResignationAudit,
    threshold: float,
) -> ResignationObservation | None:
    return next(
        (
            observation
            for observation in audit.observations
            if observation.root_value < threshold and observation.best_child_value < threshold
        ),
        None,
    )


def _threshold_statistics(
    completed_audits: tuple[CompletedResignationAudit, ...],
    threshold: float,
    confidence_level: float,
) -> ThresholdStatistics:
    triggered_audits = [
        audit
        for audit in completed_audits
        if audit.termination_reason is ResignationTerminationReason.NATURAL
        and _first_trigger(audit, threshold) is not None
    ]
    trigger_outcomes = [
        _outcome_for_triggering_player(audit, _first_trigger(audit, threshold)) for audit in triggered_audits
    ]
    recovered_wins = sum(outcome == 1.0 for outcome in trigger_outcomes)
    recovered_draws = sum(outcome == 0.0 for outcome in trigger_outcomes)
    false_non_losses = recovered_wins + recovered_draws
    completed_triggers = len(triggered_audits)
    return ThresholdStatistics(
        threshold=threshold,
        completed_triggers=completed_triggers,
        recovered_wins=recovered_wins,
        recovered_draws=recovered_draws,
        false_non_loss_rate=false_non_losses / completed_triggers if completed_triggers else 0.0,
        false_non_loss_upper_confidence=one_sided_binomial_upper_bound(
            false_non_losses,
            completed_triggers,
            confidence_level,
        ),
    )


def _outcome_for_triggering_player(
    audit: CompletedResignationAudit,
    trigger: ResignationObservation | None,
) -> float:
    assert trigger is not None
    return audit.game_outcome if trigger.side_to_move == audit.final_current_player else -audit.game_outcome


def _rate_limited_threshold(
    previous_threshold: float | None,
    target_threshold: float | None,
    parameters: ResignationParams,
) -> float | None:
    if target_threshold is None:
        return None
    starting_threshold = (
        previous_threshold if previous_threshold is not None else parameters.threshold_candidate_minimum
    )
    change = target_threshold - starting_threshold
    limited_threshold = starting_threshold + max(
        -parameters.maximum_threshold_change_per_calibration,
        min(parameters.maximum_threshold_change_per_calibration, change),
    )
    candidates = _threshold_candidates(parameters)
    return min(candidates, key=lambda candidate: abs(candidate - limited_threshold))
