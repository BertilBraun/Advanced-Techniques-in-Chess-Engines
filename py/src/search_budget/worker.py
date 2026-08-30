from __future__ import annotations

import atexit
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias

from pydantic import Field, model_validator
from src.search_budget.artifacts import (
    LabelShardManifest,
    LabelShardPhase,
    write_immutable_artifact,
    write_persisted_model,
)
from src.search_budget.labeling import (
    DeepSearchRecord,
    DeepSearchShardArtifact,
    LabelPositionSource,
    PolicyCheckpointRecord,
    PredictionRecord,
    PredictionShardArtifact,
)
from src.search_budget.policy import disabled_policy
from src.search_budget.sampling import LabelPositionIdentity
from src.self_play.completed_game import SearchVisitCounts
from src.training.checkpoint import CheckpointReference
from src.util.frozen_model import FrozenModel

if TYPE_CHECKING:
    from AlphaZeroCpp import ChessSelfPlaySearch, GoSelfPlaySearch7, GoSelfPlaySearch9
    from src.games.chess.contract import ChessPosition
    from src.games.go.contract import NativeGoPosition

    ConfiguredNativeSearch: TypeAlias = ChessSelfPlaySearch | GoSelfPlaySearch7 | GoSelfPlaySearch9
    LabelNativePosition: TypeAlias = ChessPosition | NativeGoPosition


class DeviceClaimQueue(Protocol):
    def get(self) -> int: ...


class LabelWorkerRuntime(Protocol):
    def refresh_checkpoint(self, checkpoint: CheckpointReference) -> None: ...

    def predict(self, positions: tuple[LabelPositionSource, ...]) -> tuple[PredictionRecord, ...]: ...

    def deep_search(
        self,
        positions: tuple[LabelPositionSource, ...],
        checkpoint_visits: tuple[tuple[int, ...], ...],
        deep_visit_limit: int,
        maximum_root_capacity: int,
        parallel_searches: int,
    ) -> tuple[DeepSearchRecord, ...]: ...

    def close(self) -> None: ...


LabelWorkerRuntimeFactory = Callable[[int], LabelWorkerRuntime]


@dataclass(frozen=True)
class ConfiguredLabelWorkerRuntimeFactory:
    experiment_configuration_json: str

    def __call__(self, device_id: int) -> LabelWorkerRuntime:
        return ConfiguredLabelWorkerRuntime(self.experiment_configuration_json, device_id)


class ConfiguredLabelWorkerRuntime:
    def __init__(self, experiment_configuration_json: str, device_id: int) -> None:
        from src.experiment.configuration import load_experiment_configuration_json
        from src.games.composition import create_game_implementation

        configuration = load_experiment_configuration_json(experiment_configuration_json)
        self._game = create_game_implementation(configuration)
        self._device_id = device_id
        self._search: ConfiguredNativeSearch | None = None

    def refresh_checkpoint(self, checkpoint: CheckpointReference) -> None:
        if self._search is None:
            self._search = self._create_search(checkpoint)
            return
        self._search.refresh_model(checkpoint.generation, str(checkpoint.inference_model_path))
        self._search.update_search_schedule(
            self._game.native_search_parameters(
                self._game.self_play_parameters_at(checkpoint.generation, disabled_policy())
            )
        )

    def predict(self, positions: tuple[LabelPositionSource, ...]) -> tuple[PredictionRecord, ...]:
        search = self._require_search()
        roots = [search.new_root(self._position(source), maximum_capacity=3) for source in positions]
        requests = [
            search.request(
                root,
                assigned_additional_visits=1,
                policy_checkpoint_visits=[],
                parallel_searches=1,
                add_root_noise=False,
                force_root_playouts=False,
            )
            for root in roots
        ]
        results = search.search(requests).results
        return tuple(
            PredictionRecord(
                identity=source.identity,
                predicted_curve=tuple(result.predicted_budget_curve),
            )
            for source, result in zip(positions, results, strict=True)
        )

    def deep_search(
        self,
        positions: tuple[LabelPositionSource, ...],
        checkpoint_visits: tuple[tuple[int, ...], ...],
        deep_visit_limit: int,
        maximum_root_capacity: int,
        parallel_searches: int,
    ) -> tuple[DeepSearchRecord, ...]:
        from AlphaZeroCpp import SearchCheckpointDetail

        search = self._require_search()
        roots = [
            search.new_root(self._position(source), maximum_capacity=maximum_root_capacity) for source in positions
        ]
        requests = [
            search.request(
                root,
                assigned_additional_visits=deep_visit_limit,
                policy_checkpoint_visits=list(visits),
                parallel_searches=parallel_searches,
                add_root_noise=False,
                force_root_playouts=True,
                checkpoint_detail=SearchCheckpointDetail.POLICIES,
            )
            for root, visits in zip(roots, checkpoint_visits, strict=True)
        ]
        results = search.search(requests).results
        return tuple(
            DeepSearchRecord(
                identity=source.identity,
                checkpoints=tuple(
                    PolicyCheckpointRecord(
                        visits=checkpoint.visits,
                        root_value=checkpoint.root_value,
                        policy_target_visits=SearchVisitCounts.from_native(tuple(checkpoint.policy_target_visits)),
                    )
                    for checkpoint in result.checkpoints
                ),
                final_policy_target_visits=SearchVisitCounts.from_native(tuple(result.policy_target_visits)),
                final_root_value=result.root_value,
                starting_visits=result.starting_visits,
                final_visits=result.final_visits,
            )
            for source, result in zip(positions, results, strict=True)
        )

    def close(self) -> None:
        self._game.close()

    def _create_search(self, checkpoint: CheckpointReference) -> ConfiguredNativeSearch:
        from AlphaZeroCpp import BatchedInferenceParameters, ChessSelfPlaySearch, GoSelfPlaySearch7, GoSelfPlaySearch9
        from src.games.chess.training import ChessImplementation
        from src.games.go.training import GoImplementation
        from src.self_play.configuration import BatchedInferenceParams

        parameters = self._game.self_play_parameters_at(checkpoint.generation, disabled_policy())
        source_inference = self._game.self_play_configuration.inference
        inference = BatchedInferenceParams(
            inference_workers=1,
            inference_batch_size=512,
            outstanding_batches_per_worker=2,
            sdpa_backend=source_inference.sdpa_backend,
            precision=source_inference.precision,
            memory_format=source_inference.memory_format,
            cudnn_benchmark=source_inference.cudnn_benchmark,
        )
        match self._game:
            case ChessImplementation():
                search_type = ChessSelfPlaySearch
            case GoImplementation() as game:
                search_type = GoSelfPlaySearch7 if game.state.board_size == 7 else GoSelfPlaySearch9
        self._game.validate_native_dimensions(search_type.inference_dimensions())
        return search_type(
            self._game.native_inference_configuration(
                self._device_id,
                checkpoint.inference_model_path,
                inference,
            ),
            self._game.native_search_parameters(parameters),
            BatchedInferenceParameters(1, 512, 2),
            checkpoint.generation,
        )

    def _position(self, source: LabelPositionSource) -> LabelNativePosition:
        position = self._game.state.initial_position()
        for action_id in source.action_prefix:
            position = self._game.state.child_position(position, action_id)
        return position

    def _require_search(self) -> ConfiguredNativeSearch:
        if self._search is None:
            raise RuntimeError('Label worker search is unavailable before an immutable checkpoint is loaded.')
        return self._search


class PredictionShardTask(FrozenModel):
    source_generation: int = Field(ge=0)
    shard_index: int = Field(ge=0)
    attempt: int = Field(gt=0)
    checkpoint: CheckpointReference
    positions: tuple[LabelPositionSource, ...] = Field(min_length=1, max_length=512)
    artifact_path: Path
    manifest_path: Path

    @model_validator(mode='after')
    def validate_generation(self) -> PredictionShardTask:
        if self.checkpoint.generation != self.source_generation:
            raise ValueError('Prediction shard checkpoint must match its source generation.')
        if any(position.identity.source_generation != self.source_generation for position in self.positions):
            raise ValueError('Prediction shard positions must match its source generation.')
        return self


class DeepSearchShardTask(FrozenModel):
    source_generation: int = Field(ge=0)
    shard_index: int = Field(ge=0)
    attempt: int = Field(gt=0)
    checkpoint: CheckpointReference
    positions: tuple[LabelPositionSource, ...] = Field(min_length=1, max_length=512)
    checkpoint_visits: tuple[tuple[int, ...], ...]
    deep_visit_limit: int = Field(gt=0)
    parallel_searches: int = Field(default=2, ge=2, le=2)
    add_root_noise: bool = Field(default=False)
    artifact_path: Path
    manifest_path: Path

    @model_validator(mode='after')
    def validate_search_contract(self) -> DeepSearchShardTask:
        if self.checkpoint.generation != self.source_generation:
            raise ValueError('Deep-search shard checkpoint must match its source generation.')
        if any(position.identity.source_generation != self.source_generation for position in self.positions):
            raise ValueError('Deep-search shard positions must match its source generation.')
        if self.add_root_noise:
            raise ValueError('Deep-label searches must disable root noise.')
        if len(self.checkpoint_visits) != len(self.positions):
            raise ValueError('Every deep-label position must carry its policy checkpoints.')
        for visits in self.checkpoint_visits:
            if visits != tuple(sorted(set(visits))) or not visits:
                raise ValueError('Policy checkpoint visits must be unique, sorted, and nonempty.')
            if visits[-1] > self.deep_visit_limit:
                raise ValueError('Policy checkpoint visits cannot exceed the final deep limit.')
        return self

    @property
    def maximum_root_capacity(self) -> int:
        return self.deep_visit_limit + self.parallel_searches + 1


class _WorkerState:
    def __init__(self, device_id: int, runtime: LabelWorkerRuntime) -> None:
        self.device_id = device_id
        self.runtime = runtime
        self.checkpoint_generation = -1
        self.checkpoint_sha256 = ''

    def refresh(self, checkpoint: CheckpointReference) -> None:
        if checkpoint.generation < self.checkpoint_generation:
            raise ValueError('Persistent label workers cannot roll checkpoints backward.')
        if checkpoint.generation == self.checkpoint_generation:
            if checkpoint.inference_model_sha256 != self.checkpoint_sha256:
                raise ValueError('A checkpoint generation cannot change immutable model lineage.')
            return
        checkpoint.validate_inference_model()
        self.runtime.refresh_checkpoint(checkpoint)
        self.checkpoint_generation = checkpoint.generation
        self.checkpoint_sha256 = checkpoint.inference_model_sha256


_WORKER_STATE: _WorkerState | None = None


def initialize_label_worker(device_claims: DeviceClaimQueue, runtime_factory: LabelWorkerRuntimeFactory) -> None:
    global _WORKER_STATE
    if _WORKER_STATE is not None:
        raise RuntimeError('A persistent label worker was initialized more than once.')
    device_id = device_claims.get()
    import torch

    torch.cuda.set_device(device_id)
    _WORKER_STATE = _WorkerState(device_id, runtime_factory(device_id))
    atexit.register(_close_worker)


def execute_prediction_shard(task: PredictionShardTask) -> LabelShardManifest:
    worker = _worker()
    worker.refresh(task.checkpoint)
    started_at = time.perf_counter()
    predictions = worker.runtime.predict(task.positions)
    expected = tuple(position.identity for position in task.positions)
    if tuple(prediction.identity for prediction in predictions) != expected:
        raise ValueError('Prediction worker output does not preserve exact shard position order.')
    artifact = PredictionShardArtifact(
        source_generation=task.source_generation,
        shard_index=task.shard_index,
        checkpoint_sha256=task.checkpoint.inference_model_sha256,
        predictions=predictions,
    )
    content = (artifact.model_dump_json(indent=2) + '\n').encode('utf-8')
    digest = write_immutable_artifact(task.artifact_path, content)
    manifest = _manifest(
        LabelShardPhase.PREDICTION,
        task.source_generation,
        task.shard_index,
        task.attempt,
        worker.device_id,
        expected,
        time.perf_counter() - started_at,
        task.artifact_path,
        digest,
        len(content),
        task.checkpoint.inference_model_sha256,
    )
    write_persisted_model(task.manifest_path, manifest)
    return manifest


def execute_deep_search_shard(task: DeepSearchShardTask) -> LabelShardManifest:
    worker = _worker()
    worker.refresh(task.checkpoint)
    started_at = time.perf_counter()
    records = worker.runtime.deep_search(
        task.positions,
        task.checkpoint_visits,
        task.deep_visit_limit,
        task.maximum_root_capacity,
        task.parallel_searches,
    )
    expected = tuple(position.identity for position in task.positions)
    if tuple(record.identity for record in records) != expected:
        raise ValueError('Deep-search worker output does not preserve exact shard position order.')
    for record, requested in zip(records, task.checkpoint_visits, strict=True):
        recorded = tuple(checkpoint.visits for checkpoint in record.checkpoints)
        if recorded != requested:
            raise ValueError('Deep-search worker did not return every requested policy checkpoint.')
        if record.final_visits != task.deep_visit_limit:
            raise ValueError('Deep-label search did not reach exactly eight times the source baseline.')
    artifact = DeepSearchShardArtifact(
        source_generation=task.source_generation,
        shard_index=task.shard_index,
        checkpoint_sha256=task.checkpoint.inference_model_sha256,
        records=records,
    )
    content = (artifact.model_dump_json(indent=2) + '\n').encode('utf-8')
    digest = write_immutable_artifact(task.artifact_path, content)
    manifest = _manifest(
        LabelShardPhase.DEEP_SEARCH,
        task.source_generation,
        task.shard_index,
        task.attempt,
        worker.device_id,
        expected,
        time.perf_counter() - started_at,
        task.artifact_path,
        digest,
        len(content),
        task.checkpoint.inference_model_sha256,
    )
    write_persisted_model(task.manifest_path, manifest)
    return manifest


def _worker() -> _WorkerState:
    if _WORKER_STATE is None:
        raise RuntimeError('Persistent label worker has not been initialized.')
    return _WORKER_STATE


def _close_worker() -> None:
    global _WORKER_STATE
    if _WORKER_STATE is None:
        return
    _WORKER_STATE.runtime.close()
    _WORKER_STATE = None


def _manifest(
    phase: LabelShardPhase,
    source_generation: int,
    shard_index: int,
    attempt: int,
    device_id: int,
    identities: tuple[LabelPositionIdentity, ...],
    duration_seconds: float,
    artifact_path: Path,
    artifact_sha256: str,
    artifact_size_bytes: int,
    checkpoint_sha256: str,
) -> LabelShardManifest:
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact_sha256:
        raise ValueError('Label shard artifact changed before manifest publication.')
    return LabelShardManifest(
        phase=phase,
        source_generation=source_generation,
        shard_index=shard_index,
        attempt=attempt,
        device_id=device_id,
        position_identities=identities,
        position_count=len(identities),
        duration_seconds=duration_seconds,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size_bytes,
        checkpoint_sha256=checkpoint_sha256,
    )
