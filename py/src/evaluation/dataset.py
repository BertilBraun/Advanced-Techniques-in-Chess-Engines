from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from src.evaluation.configuration import EvaluationDatasetConfiguration
from src.evaluation.contracts import (
    EvaluationDatasetManifest,
    EvaluationSourceGame,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
)
from src.evaluation.engine import EnginePolicy, EnginePolicyProvider, validate_engine_policy
from src.evaluation.inference import decode_packed_inputs
from src.games.contracts import GameStateContract
from src.util.atomic_file import write_text_atomically


PositionT = TypeVar('PositionT')
RETAINED_PLY_INTERVAL = 3
MINIMUM_DATASET_POSITIONS = 480
MAXIMUM_DATASET_POSITIONS = 520
MAXIMUM_SOURCE_GAMES = 1000


@dataclass(frozen=True)
class EvaluationDatasetRow:
    packed_state: bytes
    action_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    top_action_id: int
    source_game_id: int
    ply: int


@dataclass(frozen=True)
class _RetainedDatasetPosition:
    digest: str
    row: EvaluationDatasetRow


@dataclass(frozen=True)
class _GeneratedSourceGame:
    source_game: EvaluationSourceGame
    retained_positions: tuple[_RetainedDatasetPosition, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _position_digest(state: GameStateContract[PositionT], position: PositionT) -> str:
    return hashlib.sha256(state.encode_network_input(position).payload).hexdigest()


def _sample_action(policy: EnginePolicy, temperature: float, generator: random.Random) -> int:
    weights = tuple(entry.probability ** (1.0 / temperature) for entry in policy.entries)
    return generator.choices(
        tuple(entry.action_id for entry in policy.entries),
        weights=weights,
        k=1,
    )[0]


def _dataset_dtype(packed_payload_bytes: int, maximum_policy_entries: int) -> np.dtype:
    return np.dtype(
        [
            ('packed_state', f'V{packed_payload_bytes}'),
            ('policy_count', 'u1'),
            ('action_ids', '<u2', (maximum_policy_entries,)),
            ('probabilities', '<f4', (maximum_policy_entries,)),
            ('top_action_id', '<u2'),
            ('source_game_id', '<u4'),
            ('ply', '<u2'),
        ],
        align=False,
    )


def _layout_digest(dtype: np.dtype) -> str:
    return hashlib.sha256(json.dumps(dtype.descr, separators=(',', ':')).encode('utf-8')).hexdigest()


def _write_dataset(path: Path, rows: tuple[EvaluationDatasetRow, ...], dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    data = np.zeros(len(rows), dtype=dtype)
    for row_index, row in enumerate(rows):
        count = len(row.action_ids)
        data[row_index]['packed_state'] = np.void(row.packed_state)
        data[row_index]['policy_count'] = count
        data[row_index]['action_ids'][:count] = row.action_ids
        data[row_index]['probabilities'][:count] = row.probabilities
        data[row_index]['top_action_id'] = row.top_action_id
        data[row_index]['source_game_id'] = row.source_game_id
        data[row_index]['ply'] = row.ply
    with temporary_path.open('wb') as output:
        data.tofile(output)
        output.flush()
    temporary_path.replace(path)


def dataset_manifest_path(path: Path) -> Path:
    return path.with_name(f'{path.name}.manifest.json')


def _load_existing_dataset(
    path: Path,
    manifest_path: Path,
    configuration: EvaluationDatasetConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
) -> EvaluationDatasetManifest | None:
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.exists() or not manifest_path.exists():
        raise ValueError('Evaluation dataset data and manifest must either both exist or both be absent.')
    manifest = EvaluationDatasetManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    if _file_sha256(path) != manifest.data_sha256:
        raise ValueError('Evaluation dataset hash does not match its manifest.')
    expected = (
        manifest.game == engine.game_name
        and manifest.rules_digest == engine.rules_digest
        and manifest.representation_digest == engine.representation_digest
        and manifest.random_seed == configuration.random_seed
        and manifest.move_sampling_temperature == configuration.move_sampling_temperature
        and manifest.engine_identity == engine.engine_identity
        and manifest.engine_artifact_sha256 == engine.engine_artifact_sha256
        and manifest.label_search_limit == engine.label_search_limit
        and manifest.packed_payload_bytes == state.packed_plane_layout.payload_bytes
    )
    if not expected:
        raise ValueError('Existing evaluation dataset does not match its configured immutable provenance.')
    return manifest


def _generate_source_game(
    source_game_id: int,
    retained_offset: int,
    configuration: EvaluationDatasetConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
    generator: random.Random,
) -> _GeneratedSourceGame:
    position = state.initial_position()
    action_ids: tuple[int, ...] = ()
    retained_positions: list[_RetainedDatasetPosition] = []
    ply = 0
    while state.natural_terminal_wdl(position) is None:
        legal_actions = state.legal_action_ids(position)
        if not legal_actions:
            break
        policy = validate_engine_policy(engine.policy(position, action_ids), legal_actions)
        if ply % RETAINED_PLY_INTERVAL == retained_offset:
            ordered_entries = tuple(sorted(policy.entries, key=lambda entry: (-entry.probability, entry.action_id)))
            retained_positions.append(
                _RetainedDatasetPosition(
                    digest=_position_digest(state, position),
                    row=EvaluationDatasetRow(
                        packed_state=state.encode_network_input(position).payload,
                        action_ids=tuple(entry.action_id for entry in ordered_entries),
                        probabilities=tuple(entry.probability for entry in ordered_entries),
                        top_action_id=policy.selected_action_id,
                        source_game_id=source_game_id,
                        ply=ply,
                    ),
                )
            )
        selected_action = _sample_action(policy, configuration.move_sampling_temperature, generator)
        position = state.child_position(position, selected_action)
        action_ids = (*action_ids, selected_action)
        ply += 1
    return _GeneratedSourceGame(
        source_game=EvaluationSourceGame(
            source_game_id=source_game_id,
            action_ids=action_ids,
            human_readable=engine.render_game(action_ids),
        ),
        retained_positions=tuple(retained_positions),
    )


def _collect_dataset_rows(
    configuration: EvaluationDatasetConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
) -> tuple[tuple[EvaluationDatasetRow, ...], tuple[EvaluationSourceGame, ...], int]:
    generator = random.Random(configuration.random_seed)
    retained_offset = configuration.random_seed % RETAINED_PLY_INTERVAL
    rows: list[EvaluationDatasetRow] = []
    source_games: list[EvaluationSourceGame] = []
    position_digests: set[str] = set()
    while len(rows) < MINIMUM_DATASET_POSITIONS:
        source_game_id = len(source_games)
        if source_game_id >= MAXIMUM_SOURCE_GAMES:
            raise ValueError('Evaluation dataset did not reach its minimum position count.')
        generated = _generate_source_game(
            source_game_id,
            retained_offset,
            configuration,
            state,
            engine,
            generator,
        )
        source_games.append(generated.source_game)
        for retained in generated.retained_positions:
            if retained.digest in position_digests or len(rows) >= MAXIMUM_DATASET_POSITIONS:
                continue
            rows.append(retained.row)
            position_digests.add(retained.digest)
    return tuple(rows), tuple(source_games), retained_offset


def build_evaluation_dataset(
    path: Path,
    configuration: EvaluationDatasetConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
    builder_source_revision: str,
) -> EvaluationDatasetManifest:
    manifest_path = dataset_manifest_path(path)
    existing = _load_existing_dataset(path, manifest_path, configuration, state, engine)
    if existing is not None:
        return existing
    resolved_rows, source_games, retained_offset = _collect_dataset_rows(configuration, state, engine)
    maximum_policy_entries = max(len(row.action_ids) for row in resolved_rows)
    if maximum_policy_entries > 255:
        raise ValueError('Evaluation dataset sparse policies cannot exceed 255 entries.')
    dtype = _dataset_dtype(state.packed_plane_layout.payload_bytes, maximum_policy_entries)
    _write_dataset(path, resolved_rows, dtype)
    manifest = EvaluationDatasetManifest(
        game=engine.game_name,
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        position_count=len(resolved_rows),
        source_games=source_games,
        retained_ply_offset=retained_offset,
        random_seed=configuration.random_seed,
        move_sampling_temperature=configuration.move_sampling_temperature,
        engine_identity=engine.engine_identity,
        engine_artifact_sha256=engine.engine_artifact_sha256,
        label_search_limit=engine.label_search_limit,
        maximum_policy_entries=maximum_policy_entries,
        packed_payload_bytes=state.packed_plane_layout.payload_bytes,
        row_layout_digest=_layout_digest(dtype),
        data_sha256=_file_sha256(path),
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(manifest_path, manifest.model_dump_json(indent=2) + '\n')
    return manifest


def load_evaluation_dataset(
    path: Path,
    manifest: EvaluationDatasetManifest,
) -> npt.NDArray[np.void]:
    dtype = _dataset_dtype(manifest.packed_payload_bytes, manifest.maximum_policy_entries)
    if _layout_digest(dtype) != manifest.row_layout_digest:
        raise ValueError('Evaluation dataset row layout does not match its manifest.')
    expected_size = manifest.position_count * dtype.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError('Evaluation dataset file size does not match its manifest.')
    return np.memmap(path, mode='r', dtype=dtype, shape=(manifest.position_count,))


def evaluate_fixed_dataset(
    job: FixedDatasetEvaluationJob,
    state: GameStateContract[PositionT],
    dataset_path: Path,
    device_type: str,
    batch_size: int = 256,
) -> FixedDatasetEvaluationResult:
    if batch_size <= 0:
        raise ValueError('Evaluation dataset batch size must be positive.')
    started_at = time.monotonic()
    manifest = EvaluationDatasetManifest.model_validate_json(
        dataset_manifest_path(dataset_path).read_text(encoding='utf-8')
    )
    data = load_evaluation_dataset(dataset_path, manifest)
    device = torch.device('cpu') if device_type == 'cpu' else torch.device('cuda', job.device_id)
    model = torch.jit.load(str(job.candidate.inference_model_path), map_location=device)
    model.eval()
    correct = 0
    cross_entropy = 0.0
    with torch.inference_mode():
        for start in range(0, manifest.position_count, batch_size):
            batch = data[start : start + batch_size]
            decoded = decode_packed_inputs(
                state,
                tuple(state.packed_plane_layout.value(bytes(row['packed_state'])) for row in batch),
            )
            policy, _ = model(torch.from_numpy(decoded).to(device))
            policy = policy.float().cpu()
            top_actions = policy.argmax(dim=1).numpy()
            correct += int(np.count_nonzero(top_actions == batch['top_action_id']))
            for row_index, row in enumerate(batch):
                count = int(row['policy_count'])
                action_ids = torch.from_numpy(row['action_ids'][:count].astype(np.int64))
                targets = torch.from_numpy(row['probabilities'][:count].astype(np.float32))
                predicted = policy[row_index, action_ids].clamp_min(1e-12)
                cross_entropy -= float(torch.sum(targets * torch.log(predicted)).item())
    return FixedDatasetEvaluationResult(
        kind='fixed_dataset',
        job=job,
        position_count=manifest.position_count,
        source_game_count=len(manifest.source_games),
        top_action_accuracy=correct / manifest.position_count,
        policy_cross_entropy=cross_entropy / manifest.position_count,
        duration_seconds=time.monotonic() - started_at,
    )
