from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import Field
from src.search_stopping.calibration import (
    StopCalibrationState,
    StopDecisionReason,
    StopPolicyPublication,
    load_calibration_state_fail_closed,
    publication_for_generation,
    save_calibration_state,
)
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.policy import SearchStopPolicy, closed_policy
from src.search_stopping.predictor import LoadedStopPredictor, export_stop_predictor, fit_stop_predictor
from src.search_stopping.records import (
    PAIRED_FLOOR_RECORD_DTYPE,
    audit_log_path,
    audit_record_dtype,
    paired_floor_log_path,
    read_records,
)
from src.search_stopping.solver import AuditWindowArrays, solve_noise_floor_anchored_eps
from src.util.frozen_model import FrozenModel
from src.util.log import error as log_error


class StoppingGenerationReport(FrozenModel):
    source_generation: int = Field(ge=0)
    window_generations: tuple[int, ...]
    audit_position_count: int = Field(ge=0)
    paired_floor_count: int = Field(ge=0)
    eps_pi: float | None
    measured_noise_floor: float | None
    eps_clamped: bool
    predictor_applied: bool
    predictor_rejection: str | None
    thresholds: tuple[float, ...]
    attenuated_checkpoints: tuple[bool, ...]
    simulated_mean_spend: float | None
    realized_mean_spend: float | None
    published_apply_learned: bool
    decision_reason: StopDecisionReason
    application_generation: int = Field(ge=0)


class SearchStoppingManager:
    """Per-generation calibration: anchor eps to the paired floor, fit the stop predictor on the
    trailing window, solve the thresholds statelessly, publish fail-closed. No job queue, no
    shards, no GPU claims: audits ride live self-play and land as per-worker record files."""

    def __init__(
        self,
        run_path: Path,
        configuration: SearchStoppingConfiguration,
        configuration_sha256: str,
        first_unstarted_production_generation: int,
        worker_count: int,
    ) -> None:
        if worker_count <= 0:
            raise ValueError('Stop calibration requires at least one self-play worker.')
        self.configuration = configuration
        self.configuration_sha256 = configuration_sha256
        self.worker_count = worker_count
        self.stopping_path = run_path / 'search-stopping'
        self.state_path = self.stopping_path / 'stop-calibration.json'
        # A generation's record files are complete once every worker has activated the next
        # checkpoint, which happens before that generation is finalized — so the parsed arrays
        # can be cached for the trailing window instead of re-read every generation.
        self._audit_cache: dict[int, npt.NDArray[np.void] | None] = {}
        self._floor_cache: dict[int, npt.NDArray[np.float64] | None] = {}
        self.state: StopCalibrationState = load_calibration_state_fail_closed(
            self.state_path,
            configuration,
            configuration_sha256,
            first_unstarted_production_generation,
        )

    def publication_for_generation(self, production_generation: int) -> StopPolicyPublication:
        return publication_for_generation(self.state, production_generation)

    def finalize_generation(
        self,
        source_generation: int,
        first_unstarted_production_generation: int,
        realized_mean_spend: float | None,
    ) -> StoppingGenerationReport:
        try:
            return self._finalize(source_generation, first_unstarted_production_generation, realized_mean_spend)
        except Exception as failure:
            log_error(f'Stop calibration failed for generation {source_generation}: {failure!r}')
            self._publish_closed(
                source_generation, first_unstarted_production_generation, StopDecisionReason.TERMINAL_FAILURE
            )
            return self._closed_report(source_generation, realized_mean_spend)

    def _finalize(
        self,
        source_generation: int,
        first_unstarted_production_generation: int,
        realized_mean_spend: float | None,
    ) -> StoppingGenerationReport:
        configuration = self.configuration
        window = self._window_generations(source_generation)
        records = self._load_audit_window(window)
        floors = self._load_paired_floors(window)
        report_base = dict(
            source_generation=source_generation,
            window_generations=tuple(window),
            audit_position_count=0 if records is None else int(records.shape[0]),
            paired_floor_count=int(floors.size),
            realized_mean_spend=realized_mean_spend,
        )

        def closed(reason: StopDecisionReason, **extra: object) -> StoppingGenerationReport:
            self._publish_closed(source_generation, first_unstarted_production_generation, reason)
            values = {
                'eps_pi': None,
                'measured_noise_floor': None,
                'eps_clamped': False,
                'predictor_applied': False,
                'predictor_rejection': None,
                'thresholds': (),
                'attenuated_checkpoints': (),
                'simulated_mean_spend': None,
                'published_apply_learned': False,
                'decision_reason': reason,
                'application_generation': first_unstarted_production_generation,
            }
            values.update(report_base)
            values.update(extra)
            return StoppingGenerationReport(**values)  # type: ignore[arg-type]

        if (
            first_unstarted_production_generation < configuration.first_production_generation
            or records is None
            or records.shape[0] < configuration.minimum_evidence_trigger_count
            or floors.size == 0
        ):
            return closed(StopDecisionReason.WARMUP)
        if realized_mean_spend is not None and realized_mean_spend > configuration.maximum_realized_mean_spend:
            return closed(StopDecisionReason.SPEND_BREAKER)

        checkpoint_count = len(configuration.checkpoint_multiples)
        features = np.ascontiguousarray(records['features'], dtype=np.float32)
        flat_features = features.reshape(-1, features.shape[-1])
        group_keys = np.repeat(records['game_key'], checkpoint_count).astype(np.uint64)
        eps_probe = solve_noise_floor_anchored_eps(
            self._arrays(records, np.ascontiguousarray(records['stop_probability'], dtype=np.float64)),
            floors,
            configuration,
        )
        labels = ((records['kl_to_final'] >= eps_probe.eps_pi) | (records['value_gap'] >= configuration.eps_v)).astype(
            np.float32
        )
        fit = fit_stop_predictor(flat_features, labels.reshape(-1), group_keys)
        if not fit.applied or fit.network is None:
            return closed(StopDecisionReason.PREDICTOR_REJECTED, predictor_rejection=fit.rejection_reason)
        predictor_path = self.stopping_path / f'stop-predictor-generation-{source_generation:08d}.jit.pt'
        predictor_sha256 = export_stop_predictor(fit.network, predictor_path)
        loaded = LoadedStopPredictor.load(predictor_path, predictor_sha256)
        probabilities = np.array(
            [loaded(tuple(float(value) for value in row)) for row in flat_features],
            dtype=np.float64,
        ).reshape(records.shape[0], checkpoint_count)
        solution = solve_noise_floor_anchored_eps(self._arrays(records, probabilities), floors, configuration)
        thresholds = tuple(item.threshold for item in solution.thresholds.checkpoints)
        attenuated = tuple(item.attenuated for item in solution.thresholds.checkpoints)
        if not solution.thresholds.any_checkpoint_open:
            return closed(
                StopDecisionReason.NO_SAFE_THRESHOLD,
                eps_pi=solution.eps_pi,
                measured_noise_floor=solution.measured_noise_floor,
                eps_clamped=solution.clamped,
                thresholds=thresholds,
                attenuated_checkpoints=attenuated,
                simulated_mean_spend=solution.thresholds.simulated_mean_spend,
            )
        policy = SearchStopPolicy(
            checkpoint_multiples=tuple(configuration.checkpoint_multiples),
            thresholds=thresholds,
            movement_guard_epsilon=configuration.movement_guard_epsilon,
            cap_multiple=configuration.cap_multiple,
            predictor_path=predictor_path,
            predictor_sha256=predictor_sha256,
            apply_learned=True,
        )
        self._publish(
            source_generation,
            first_unstarted_production_generation,
            policy,
            solution.eps_pi,
            predictor_path,
            predictor_sha256,
            solution.clamped,
        )
        values = {
            'eps_pi': solution.eps_pi,
            'measured_noise_floor': solution.measured_noise_floor,
            'eps_clamped': solution.clamped,
            'predictor_applied': True,
            'predictor_rejection': None,
            'thresholds': thresholds,
            'attenuated_checkpoints': attenuated,
            'simulated_mean_spend': solution.thresholds.simulated_mean_spend,
            'published_apply_learned': True,
            'decision_reason': StopDecisionReason.APPLIED,
            'application_generation': first_unstarted_production_generation,
        }
        values.update(report_base)
        return StoppingGenerationReport(**values)  # type: ignore[arg-type]

    def _arrays(self, records: npt.NDArray[np.void], probabilities: npt.NDArray[np.float64]) -> AuditWindowArrays:
        return AuditWindowArrays(
            kl_to_final=np.ascontiguousarray(records['kl_to_final'], dtype=np.float64),
            value_gap=np.ascontiguousarray(records['value_gap'], dtype=np.float64),
            guard_movement=np.ascontiguousarray(records['guard_movement'], dtype=np.float64),
            stop_probability=np.ascontiguousarray(probabilities, dtype=np.float64),
        )

    def _window_generations(self, source_generation: int) -> list[int]:
        oldest = max(0, source_generation - self.configuration.window_generations + 1)
        return list(range(oldest, source_generation + 1))

    def _load_audit_window(self, window: list[int]) -> npt.NDArray[np.void] | None:
        self._evict_outside(window)
        chunks: list[npt.NDArray[np.void]] = []
        current_baseline: int | None = None
        for generation in reversed(window):
            merged = self._generation_audit_records(generation)
            if merged is None:
                continue
            baseline = int(merged['baseline_visits'][0])
            if current_baseline is None:
                current_baseline = baseline
            elif baseline != current_baseline:
                # The window never crosses a baseline-visits schedule step: the reference and
                # every KL in the labels shift systematically at a step.
                break
            chunks.append(merged)
        if not chunks:
            return None
        return np.concatenate(chunks)

    def _load_paired_floors(self, window: list[int]) -> npt.NDArray[np.float64]:
        self._evict_outside(window)
        values = [floors for generation in window if (floors := self._generation_paired_floors(generation)) is not None]
        if not values:
            return np.array([], dtype=np.float64)
        return np.concatenate(values)

    def _generation_audit_records(self, generation: int) -> npt.NDArray[np.void] | None:
        if generation not in self._audit_cache:
            dtype = audit_record_dtype(len(self.configuration.checkpoint_multiples))
            paths = self._record_paths(
                generation,
                [audit_log_path(self.stopping_path, generation, worker_id) for worker_id in range(self.worker_count)],
                f'audit-generation-{generation:08d}-worker-*.np',
            )
            generation_chunks = []
            for path in paths:
                try:
                    generation_chunks.append(read_records(path, dtype))
                except ValueError:
                    log_error(f'Skipping unreadable audit log: {path}')
            self._audit_cache[generation] = np.concatenate(generation_chunks) if generation_chunks else None
        return self._audit_cache[generation]

    def _generation_paired_floors(self, generation: int) -> npt.NDArray[np.float64] | None:
        if generation not in self._floor_cache:
            paths = self._record_paths(
                generation,
                [
                    paired_floor_log_path(self.stopping_path, generation, worker_id)
                    for worker_id in range(self.worker_count)
                ],
                f'paired-floor-generation-{generation:08d}-worker-*.np',
            )
            values = []
            for path in paths:
                try:
                    values.append(read_records(path, PAIRED_FLOOR_RECORD_DTYPE)['kl_symmetric'].astype(np.float64))
                except ValueError:
                    log_error(f'Skipping unreadable paired-floor log: {path}')
            self._floor_cache[generation] = np.concatenate(values) if values else None
        return self._floor_cache[generation]

    def _record_paths(self, generation: int, constructed: list[Path], fallback_pattern: str) -> list[Path]:
        """Worker ids are dense and reused across restarts, so the constructed paths cover every
        file; the glob remains only for a resumed run whose worker count shrank."""
        existing = [path for path in constructed if path.exists()]
        if existing:
            return existing
        return sorted(self.stopping_path.glob(fallback_pattern))

    def _evict_outside(self, window: list[int]) -> None:
        retained = set(window)
        self._audit_cache = {g: v for g, v in self._audit_cache.items() if g in retained}
        self._floor_cache = {g: v for g, v in self._floor_cache.items() if g in retained}

    def _publish(
        self,
        source_generation: int,
        first_unstarted_production_generation: int,
        policy: SearchStopPolicy,
        eps_pi: float,
        predictor_path: Path,
        predictor_sha256: str,
        eps_clamped: bool,
    ) -> None:
        if not math.isfinite(eps_pi):
            raise ValueError('A published eps must be finite.')
        previous = publication_for_generation(self.state, max(0, first_unstarted_production_generation - 1)).policy
        self.state = self.state.model_copy(
            update={
                'finalized_source_generations': self._finalized_with(source_generation),
                'solved_eps_pi': eps_pi,
                'eps_saturated_at_maximum': eps_clamped,
                'predictor_path': predictor_path,
                'predictor_sha256': predictor_sha256,
                'previous_published_policy': previous,
                'published_policy': policy,
                'application_generation': first_unstarted_production_generation,
                'decision_reason': StopDecisionReason.APPLIED,
            }
        )
        save_calibration_state(self.state_path, self.state)

    def _publish_closed(
        self,
        source_generation: int,
        first_unstarted_production_generation: int,
        reason: StopDecisionReason,
    ) -> None:
        previous = publication_for_generation(self.state, max(0, first_unstarted_production_generation - 1)).policy
        self.state = self.state.model_copy(
            update={
                'finalized_source_generations': self._finalized_with(source_generation),
                'previous_published_policy': previous,
                'published_policy': closed_policy(self.configuration),
                'application_generation': first_unstarted_production_generation,
                'decision_reason': reason,
            }
        )
        save_calibration_state(self.state_path, self.state)

    def _finalized_with(self, source_generation: int) -> tuple[int, ...]:
        finalized = self.state.finalized_source_generations
        if source_generation in finalized:
            return finalized
        return (*finalized, source_generation)

    def _closed_report(self, source_generation: int, realized_mean_spend: float | None) -> StoppingGenerationReport:
        return StoppingGenerationReport(
            source_generation=source_generation,
            window_generations=(),
            audit_position_count=0,
            paired_floor_count=0,
            eps_pi=None,
            measured_noise_floor=None,
            eps_clamped=False,
            predictor_applied=False,
            predictor_rejection=None,
            thresholds=(),
            attenuated_checkpoints=(),
            simulated_mean_spend=None,
            realized_mean_spend=realized_mean_spend,
            published_apply_learned=False,
            decision_reason=self.state.decision_reason,
            application_generation=self.state.application_generation,
        )
