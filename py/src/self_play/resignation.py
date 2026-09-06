from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Iterator, Literal, TypeAlias

from pydantic import Field, model_validator
from src.games.contracts import WdlTarget
from src.self_play.completed_game import CompletedSelfPlayGame, SearchObservation, TerminationReason
from src.util.atomic_file import write_text_atomically
from src.util.binomial import one_sided_binomial_upper_bound
from src.util.frozen_model import FrozenModel


class DisabledResignationConfiguration(FrozenModel):
    kind: Literal['disabled'] = 'disabled'


class CalibratedResignationConfiguration(FrozenModel):
    kind: Literal['calibrated'] = 'calibrated'
    first_production_generation: int = Field(ge=0)
    false_nonloss_rate_ceiling: float = Field(gt=0.0, lt=1.0)
    continuation_game_probability: float = Field(gt=0.0, le=1.0)
    triggered_game_window: int = Field(gt=0)
    candidate_threshold_minimum: float = Field(ge=-1.0, lt=0.0)
    candidate_threshold_maximum: float = Field(gt=-1.0, lt=0.0)
    candidate_threshold_step: float = Field(gt=0.0)
    minimum_evidence_trigger_count: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    maximum_relaxation_per_generation: float = Field(gt=0.0)

    @model_validator(mode='after')
    def validate_candidate_grid(self) -> CalibratedResignationConfiguration:
        if self.candidate_threshold_maximum <= self.candidate_threshold_minimum:
            raise ValueError('Maximum resignation threshold must be above the minimum threshold.')
        span_steps = (
            self.candidate_threshold_maximum - self.candidate_threshold_minimum
        ) / self.candidate_threshold_step
        if abs(span_steps - round(span_steps)) > 1e-8:
            raise ValueError('Resignation threshold bounds must be an integral number of configured steps apart.')
        if self.maximum_relaxation_per_generation + 1e-12 < self.candidate_threshold_step:
            raise ValueError('Maximum resignation relaxation must be at least one candidate-grid step.')
        return self

    @property
    def candidate_thresholds(self) -> tuple[float, ...]:
        count = round(
            (self.candidate_threshold_maximum - self.candidate_threshold_minimum) / self.candidate_threshold_step
        )
        return tuple(
            round(self.candidate_threshold_minimum + index * self.candidate_threshold_step, 8)
            for index in range(count + 1)
        )


ResignationConfiguration: TypeAlias = Annotated[
    DisabledResignationConfiguration | CalibratedResignationConfiguration,
    Field(discriminator='kind'),
]


class PublishedResignationPolicy(FrozenModel):
    threshold: float | None = Field(default=None, ge=-1.0, lt=0.0)


class CandidateAuditEvidence(FrozenModel):
    threshold: float = Field(ge=-1.0, lt=0.0)
    trigger_ply: int = Field(ge=0)
    saved_plies: int = Field(ge=0)
    false_nonloss: bool


class TriggeredContinuationGame(FrozenModel):
    game_identity: str
    evidence: tuple[CandidateAuditEvidence, ...] = Field(min_length=1)


class ResignationThresholdStatistics(FrozenModel):
    threshold: float
    trigger_count: int = Field(ge=0)
    false_nonloss_count: int = Field(ge=0)
    false_nonloss_rate: float = Field(ge=0.0, le=1.0)
    false_nonloss_upper_bound: float = Field(ge=0.0, le=1.0)


class ResignationCalibrationState(FrozenModel):
    schema_version: Literal[1] = 1
    configuration_sha256: str
    selected_threshold: float | None = None
    selected_threshold_safe: bool = False
    last_relaxation_generation: int | None = Field(default=None, ge=0)
    triggered_continuation_games: tuple[TriggeredContinuationGame, ...] = ()
    completed_continuation_games: int = Field(default=0, ge=0)
    actual_resignations: int = Field(default=0, ge=0)
    broadest_candidate_triggers: int = Field(default=0, ge=0)
    false_nonlosses_at_selected_threshold: int = Field(default=0, ge=0)
    selected_trigger_ply_total: int = Field(default=0, ge=0)
    selected_saved_ply_total: int = Field(default=0, ge=0)


class ResignationDiagnostics(FrozenModel):
    selected_threshold: float | None
    selected_threshold_safe: bool
    continuation_games: int
    audit_triggers: int
    false_nonlosses: int
    false_nonloss_rate: float
    false_nonloss_upper_bound: float
    actual_resignations: int
    average_trigger_ply: float | None
    average_saved_plies: float | None


def resignation_predicate(observation: SearchObservation, threshold: float) -> bool:
    return observation.root_value <= threshold and observation.highest_visited_child_q <= threshold


class ResignationCalibrationBatch:
    def __init__(
        self,
        configuration: CalibratedResignationConfiguration,
        state: ResignationCalibrationState,
        journal: sqlite3.Connection,
    ) -> None:
        self.configuration = configuration
        self.state = state
        self.journal = journal
        self.has_changes = False

    def observe_completed_game(self, game: CompletedSelfPlayGame) -> None:
        self.observe_game_record(
            archive_key=game.identity.archive_key,
            is_resignation_continuation=game.is_resignation_continuation,
            termination_reason=game.termination_reason,
            final_wdl=game.final_wdl,
            length_plies=len(game.action_ids),
            observations=game.observations,
        )

    def observe_game_record(
        self,
        archive_key: str,
        is_resignation_continuation: bool,
        termination_reason: TerminationReason,
        final_wdl: WdlTarget,
        length_plies: int,
        observations: tuple[SearchObservation, ...],
    ) -> None:
        triggered = (
            _continuation_evidence(
                archive_key,
                termination_reason,
                final_wdl,
                length_plies,
                observations,
                self.configuration,
            )
            if is_resignation_continuation
            else None
        )
        cursor = self.journal.execute(
            """
            INSERT OR IGNORE INTO completed_games (
                game_identity,
                continuation,
                actual_resignation,
                triggered_evidence
            ) VALUES (?, ?, ?, ?)
            """,
            (
                archive_key,
                int(is_resignation_continuation),
                int(termination_reason is TerminationReason.RESIGNATION),
                None if triggered is None else triggered.model_dump_json(),
            ),
        )
        if cursor.rowcount != 1:
            return
        state = self.state
        rolling = state.triggered_continuation_games
        if triggered is not None:
            rolling = (*rolling, triggered)[-self.configuration.triggered_game_window :]
        self.state = ResignationCalibrationState(
            configuration_sha256=state.configuration_sha256,
            selected_threshold=state.selected_threshold,
            selected_threshold_safe=state.selected_threshold_safe,
            last_relaxation_generation=state.last_relaxation_generation,
            triggered_continuation_games=rolling,
            completed_continuation_games=state.completed_continuation_games + int(is_resignation_continuation),
            actual_resignations=state.actual_resignations + int(termination_reason is TerminationReason.RESIGNATION),
            broadest_candidate_triggers=state.broadest_candidate_triggers + int(triggered is not None),
            false_nonlosses_at_selected_threshold=state.false_nonlosses_at_selected_threshold,
            selected_trigger_ply_total=state.selected_trigger_ply_total,
            selected_saved_ply_total=state.selected_saved_ply_total,
        )
        self.has_changes = True


class ResignationCalibrator:
    def __init__(self, path: Path, configuration: CalibratedResignationConfiguration) -> None:
        self.path = path
        self.configuration = configuration
        self._configuration_sha256 = sha256(configuration.model_dump_json().encode('utf-8')).hexdigest()
        self._journal_path = path.with_suffix('.sqlite3')
        self._initialize_journal()
        self.state = self._load()
        self._recover_observation_state()

    def published_policy(self, model_generation: int) -> PublishedResignationPolicy:
        threshold = (
            self.state.selected_threshold
            if model_generation >= self.configuration.first_production_generation and self.state.selected_threshold_safe
            else None
        )
        return PublishedResignationPolicy(threshold=threshold)

    @contextmanager
    def calibration_batch(self) -> Iterator[ResignationCalibrationBatch]:
        journal = sqlite3.connect(self._journal_path)
        batch = ResignationCalibrationBatch(self.configuration, self.state, journal)
        try:
            yield batch
            journal.commit()
        except BaseException:
            journal.rollback()
            raise
        finally:
            journal.close()
        if batch.has_changes:
            self.state = batch.state
            self._save()

    def advance_generation(self, model_generation: int) -> None:
        self._recalibrate(model_generation)
        self._save()

    def diagnostics(self) -> ResignationDiagnostics:
        statistics = self._selected_statistics()
        trigger_count = 0 if statistics is None else statistics.trigger_count
        false_nonlosses = 0 if statistics is None else statistics.false_nonloss_count
        return ResignationDiagnostics(
            selected_threshold=self.state.selected_threshold,
            selected_threshold_safe=self.state.selected_threshold_safe,
            continuation_games=self.state.completed_continuation_games,
            audit_triggers=self.state.broadest_candidate_triggers,
            false_nonlosses=false_nonlosses,
            false_nonloss_rate=0.0 if statistics is None else statistics.false_nonloss_rate,
            false_nonloss_upper_bound=1.0 if statistics is None else statistics.false_nonloss_upper_bound,
            actual_resignations=self.state.actual_resignations,
            average_trigger_ply=(self.state.selected_trigger_ply_total / trigger_count if trigger_count else None),
            average_saved_plies=(self.state.selected_saved_ply_total / trigger_count if trigger_count else None),
        )

    def _recalibrate(self, model_generation: int) -> None:
        statistics = self._statistics()
        safe = tuple(
            item
            for item in statistics
            if item.trigger_count >= self.configuration.minimum_evidence_trigger_count
            and item.false_nonloss_upper_bound <= self.configuration.false_nonloss_rate_ceiling
        )
        target = safe[-1].threshold if safe else None
        selected = self.state.selected_threshold
        last_relaxation = self.state.last_relaxation_generation
        selected_safe = self.state.selected_threshold_safe
        if selected is None and target is not None:
            selected = self.configuration.candidate_threshold_minimum
            last_relaxation = model_generation
            selected_safe = (
                _statistics_at(statistics, selected).false_nonloss_upper_bound
                <= self.configuration.false_nonloss_rate_ceiling
            )
        elif selected is not None and target is not None and target < selected:
            selected = target
            selected_safe = True
        elif selected is not None and target is not None and target > selected and last_relaxation != model_generation:
            selected = min(target, selected + self.configuration.maximum_relaxation_per_generation)
            selected = _nearest_candidate(selected, self.configuration.candidate_thresholds)
            last_relaxation = model_generation
            selected_statistics = _statistics_at(statistics, selected)
            selected_safe = (
                selected_statistics.trigger_count >= self.configuration.minimum_evidence_trigger_count
                and selected_statistics.false_nonloss_upper_bound <= self.configuration.false_nonloss_rate_ceiling
            )
        elif selected is not None and target is None:
            current = _statistics_at(statistics, selected)
            if current.trigger_count >= self.configuration.minimum_evidence_trigger_count:
                selected = self.configuration.candidate_threshold_minimum
                minimum = _statistics_at(statistics, selected)
                selected_safe = (
                    minimum.trigger_count >= self.configuration.minimum_evidence_trigger_count
                    and minimum.false_nonloss_upper_bound <= self.configuration.false_nonloss_rate_ceiling
                )
        selected_statistics = None if selected is None else _statistics_at(statistics, selected)
        state = self.state
        self.state = ResignationCalibrationState(
            configuration_sha256=state.configuration_sha256,
            selected_threshold=selected,
            selected_threshold_safe=selected_safe,
            last_relaxation_generation=last_relaxation,
            triggered_continuation_games=state.triggered_continuation_games,
            completed_continuation_games=state.completed_continuation_games,
            actual_resignations=state.actual_resignations,
            broadest_candidate_triggers=state.broadest_candidate_triggers,
            false_nonlosses_at_selected_threshold=(
                0 if selected_statistics is None else selected_statistics.false_nonloss_count
            ),
            selected_trigger_ply_total=_selected_total(state.triggered_continuation_games, selected, 'trigger_ply'),
            selected_saved_ply_total=_selected_total(state.triggered_continuation_games, selected, 'saved_plies'),
        )

    def _statistics(self) -> tuple[ResignationThresholdStatistics, ...]:
        return tuple(
            _threshold_statistics(
                self.state.triggered_continuation_games,
                threshold,
                self.configuration.confidence_level,
            )
            for threshold in self.configuration.candidate_thresholds
        )

    def _selected_statistics(self) -> ResignationThresholdStatistics | None:
        if self.state.selected_threshold is None:
            return None
        return _statistics_at(self._statistics(), self.state.selected_threshold)

    def _load(self) -> ResignationCalibrationState:
        if not self.path.exists():
            return ResignationCalibrationState(configuration_sha256=self._configuration_sha256)
        state = ResignationCalibrationState.model_validate_json(self.path.read_text(encoding='utf-8'))
        if state.configuration_sha256 != self._configuration_sha256:
            raise ValueError('Resignation calibration state belongs to a different configuration.')
        return state

    def _save(self) -> None:
        write_text_atomically(self.path, self.state.model_dump_json(indent=2) + '\n')

    def _initialize_journal(self) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._journal_path) as journal:
            journal.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_games (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_identity TEXT NOT NULL UNIQUE,
                    continuation INTEGER NOT NULL,
                    actual_resignation INTEGER NOT NULL,
                    triggered_evidence TEXT
                )
                """
            )
            journal.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    configuration_sha256 TEXT NOT NULL
                )
                """
            )
            row = journal.execute('SELECT configuration_sha256 FROM metadata WHERE singleton = 1').fetchone()
            if row is None:
                journal.execute(
                    'INSERT INTO metadata (singleton, configuration_sha256) VALUES (1, ?)',
                    (self._configuration_sha256,),
                )
            elif row[0] != self._configuration_sha256:
                raise ValueError('Resignation calibration journal belongs to a different configuration.')

    def _recover_observation_state(self) -> None:
        with sqlite3.connect(self._journal_path) as journal:
            continuation_games, actual_resignations = journal.execute(
                'SELECT COALESCE(SUM(continuation), 0), COALESCE(SUM(actual_resignation), 0) FROM completed_games'
            ).fetchone()
            rows = journal.execute(
                """
                SELECT triggered_evidence
                FROM completed_games
                WHERE triggered_evidence IS NOT NULL
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (self.configuration.triggered_game_window,),
            ).fetchall()
            (broadest_candidate_triggers,) = journal.execute(
                'SELECT COUNT(*) FROM completed_games WHERE triggered_evidence IS NOT NULL'
            ).fetchone()
        triggered_games = tuple(TriggeredContinuationGame.model_validate_json(payload) for (payload,) in reversed(rows))
        state = self.state
        recovered = ResignationCalibrationState(
            configuration_sha256=state.configuration_sha256,
            selected_threshold=state.selected_threshold,
            selected_threshold_safe=state.selected_threshold_safe,
            last_relaxation_generation=state.last_relaxation_generation,
            triggered_continuation_games=triggered_games,
            completed_continuation_games=int(continuation_games),
            actual_resignations=int(actual_resignations),
            broadest_candidate_triggers=int(broadest_candidate_triggers),
            false_nonlosses_at_selected_threshold=state.false_nonlosses_at_selected_threshold,
            selected_trigger_ply_total=state.selected_trigger_ply_total,
            selected_saved_ply_total=state.selected_saved_ply_total,
        )
        if recovered != self.state:
            self.state = recovered
            self._save()


def _continuation_evidence(
    archive_key: str,
    termination_reason: TerminationReason,
    game_final_wdl: WdlTarget,
    length_plies: int,
    observations: tuple[SearchObservation, ...],
    configuration: CalibratedResignationConfiguration,
) -> TriggeredContinuationGame | None:
    if termination_reason is not TerminationReason.NATURAL:
        return None
    evidence = []
    for threshold in configuration.candidate_thresholds:
        trigger = next((item for item in observations if resignation_predicate(item, threshold)), None)
        if trigger is None:
            continue
        final_wdl = game_final_wdl if (length_plies - trigger.ply) % 2 == 0 else game_final_wdl.reversed()
        evidence.append(
            CandidateAuditEvidence(
                threshold=threshold,
                trigger_ply=trigger.ply,
                saved_plies=length_plies - trigger.ply,
                false_nonloss=final_wdl.loss <= final_wdl.win,
            )
        )
    if not evidence:
        return None
    return TriggeredContinuationGame(game_identity=archive_key, evidence=tuple(evidence))


def _threshold_statistics(
    games: tuple[TriggeredContinuationGame, ...],
    threshold: float,
    confidence_level: float,
) -> ResignationThresholdStatistics:
    evidence = tuple(candidate for game in games for candidate in game.evidence if candidate.threshold == threshold)
    false_nonlosses = sum(item.false_nonloss for item in evidence)
    count = len(evidence)
    return ResignationThresholdStatistics(
        threshold=threshold,
        trigger_count=count,
        false_nonloss_count=false_nonlosses,
        false_nonloss_rate=false_nonlosses / count if count else 0.0,
        false_nonloss_upper_bound=one_sided_binomial_upper_bound(false_nonlosses, count, confidence_level),
    )


def _statistics_at(
    statistics: tuple[ResignationThresholdStatistics, ...], threshold: float
) -> ResignationThresholdStatistics:
    return next(item for item in statistics if item.threshold == threshold)


def _nearest_candidate(value: float, candidates: tuple[float, ...]) -> float:
    return min(candidates, key=lambda candidate: abs(candidate - value))


def _selected_total(
    games: tuple[TriggeredContinuationGame, ...], threshold: float | None, field: Literal['trigger_ply', 'saved_plies']
) -> int:
    if threshold is None:
        return 0
    total = 0
    for game in games:
        candidate = next((item for item in game.evidence if item.threshold == threshold), None)
        if candidate is None:
            continue
        total += candidate.trigger_ply if field == 'trigger_ply' else candidate.saved_plies
    return total
