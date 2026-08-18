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

from src.evaluation.configuration import EngineSelfPlayDatasetSource, EvaluationDatasetConfiguration
from src.evaluation.contracts import (
    AnyEvaluationDatasetManifest,
    EVALUATION_DATASET_MANIFEST_ADAPTER,
    EvaluationDatasetManifest,
    EvaluationSourceGame,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
    KataGoBookDatasetManifest,
    KataGoBookDatasetPositionProvenance,
)
from src.evaluation.engine import (
    EnginePolicy,
    EnginePolicyEntry,
    EnginePolicyProvider,
    EvaluationArtifactMetadata,
    validate_engine_policy,
)
from src.evaluation.inference import decode_packed_inputs
from src.evaluation.katago_book import (
    load_katago_book_export,
    orient_katago_book_positions,
    select_katago_book_positions,
    selection_sha256,
)
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
    legal_action_ids: tuple[int, ...]
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


def _dataset_dtype(
    packed_payload_bytes: int,
    maximum_policy_entries: int,
    maximum_legal_actions: int,
) -> np.dtype:
    return np.dtype(
        [
            ('packed_state', f'V{packed_payload_bytes}'),
            ('policy_count', 'u1'),
            ('action_ids', '<u2', (maximum_policy_entries,)),
            ('probabilities', '<f4', (maximum_policy_entries,)),
            ('legal_count', 'u1'),
            ('legal_action_ids', '<u2', (maximum_legal_actions,)),
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
        data[row_index]['legal_count'] = len(row.legal_action_ids)
        data[row_index]['legal_action_ids'][: len(row.legal_action_ids)] = row.legal_action_ids
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
    if not isinstance(configuration.source, EngineSelfPlayDatasetSource):
        raise ValueError('Engine self-play dataset loader requires an engine-self-play source.')
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
        and manifest.random_seed == configuration.source.random_seed
        and manifest.move_sampling_temperature == configuration.source.move_sampling_temperature
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
    if not isinstance(configuration.source, EngineSelfPlayDatasetSource):
        raise ValueError('Engine self-play dataset builder requires an engine-self-play source.')
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
                        legal_action_ids=tuple(sorted(legal_actions)),
                        top_action_id=policy.selected_action_id,
                        source_game_id=source_game_id,
                        ply=ply,
                    ),
                )
            )
        selected_action = _sample_action(policy, configuration.source.move_sampling_temperature, generator)
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
    if not isinstance(configuration.source, EngineSelfPlayDatasetSource):
        raise ValueError('Engine self-play dataset builder requires an engine-self-play source.')
    generator = random.Random(configuration.source.random_seed)
    retained_offset = configuration.source.random_seed % RETAINED_PLY_INTERVAL
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
    if not isinstance(configuration.source, EngineSelfPlayDatasetSource):
        raise ValueError('Engine self-play dataset builder requires an engine-self-play source.')
    manifest_path = dataset_manifest_path(path)
    existing = _load_existing_dataset(path, manifest_path, configuration, state, engine)
    if existing is not None:
        return existing
    resolved_rows, source_games, retained_offset = _collect_dataset_rows(configuration, state, engine)
    maximum_policy_entries = max(len(row.action_ids) for row in resolved_rows)
    maximum_legal_actions = max(len(row.legal_action_ids) for row in resolved_rows)
    if maximum_policy_entries > 255:
        raise ValueError('Evaluation dataset sparse policies cannot exceed 255 entries.')
    dtype = _dataset_dtype(
        state.packed_plane_layout.payload_bytes,
        maximum_policy_entries,
        maximum_legal_actions,
    )
    _write_dataset(path, resolved_rows, dtype)
    manifest = EvaluationDatasetManifest(
        game=engine.game_name,
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        position_count=len(resolved_rows),
        source_games=source_games,
        retained_ply_offset=retained_offset,
        random_seed=configuration.source.random_seed,
        move_sampling_temperature=configuration.source.move_sampling_temperature,
        engine_identity=engine.engine_identity,
        engine_artifact_sha256=engine.engine_artifact_sha256,
        label_search_limit=engine.label_search_limit,
        maximum_policy_entries=maximum_policy_entries,
        maximum_legal_actions=maximum_legal_actions,
        packed_payload_bytes=state.packed_plane_layout.payload_bytes,
        row_layout_digest=_layout_digest(dtype),
        data_sha256=_file_sha256(path),
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(manifest_path, manifest.model_dump_json(indent=2) + '\n')
    return manifest


def build_katago_book_evaluation_dataset(
    path: Path,
    export_path: Path,
    configuration: EvaluationDatasetConfiguration,
    state: GameStateContract[PositionT],
    engine: EvaluationArtifactMetadata[PositionT],
    builder_source_revision: str,
) -> KataGoBookDatasetManifest:
    if configuration.source.kind != 'katago_book':
        raise ValueError('KataGo book dataset builder requires a book source.')
    selection_digest = selection_sha256(configuration.source.selection)
    manifest_path = dataset_manifest_path(path)
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise ValueError('Evaluation dataset data and manifest must either both exist or both be absent.')
        manifest = KataGoBookDatasetManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        expected = (
            _file_sha256(path) == manifest.data_sha256
            and manifest.rules_digest == engine.rules_digest
            and manifest.representation_digest == engine.representation_digest
            and manifest.book_export_sha256 == configuration.source.selection.export_sha256
            and manifest.book_selection_sha256 == selection_digest
            and manifest.position_count == configuration.source.position_count
        )
        if not expected:
            raise ValueError('Existing book evaluation dataset does not match its configured immutable provenance.')
        return manifest

    export = load_katago_book_export(export_path, configuration.source.selection.export_sha256)
    selected = orient_katago_book_positions(
        select_katago_book_positions(
            export,
            configuration.source.selection,
            configuration.source.position_count,
        )
    )
    rows: list[EvaluationDatasetRow] = []
    source_games: list[EvaluationSourceGame] = []
    book_positions: list[KataGoBookDatasetPositionProvenance] = []
    position_digests: set[str] = set()
    for source_game_id, oriented_position in enumerate(selected):
        book_position = oriented_position.position
        position = state.initial_position()
        for action_id in book_position.action_ids:
            if action_id not in state.legal_action_ids(position):
                raise ValueError(f'KataGo book path contains illegal action {action_id}.')
            position = state.child_position(position, action_id)
        if state.natural_terminal_wdl(position) is not None:
            raise ValueError('KataGo book dataset path ends at a natural terminal position.')
        legal_actions = state.legal_action_ids(position)
        if book_position.preferred_action_id not in legal_actions:
            raise ValueError('KataGo book preferred action is illegal under the configured game rules.')
        digest = _position_digest(state, position)
        if digest in position_digests:
            raise ValueError('KataGo book selection produced duplicate native dataset positions.')
        position_digests.add(digest)
        policy = validate_engine_policy(
            EnginePolicy(
                entries=tuple(
                    EnginePolicyEntry(action_id=entry.action_id, probability=entry.probability)
                    for entry in book_position.policy
                ),
                selected_action_id=book_position.preferred_action_id,
            ),
            legal_actions,
        )
        ordered_entries = tuple(sorted(policy.entries, key=lambda entry: (-entry.probability, entry.action_id)))
        rows.append(
            EvaluationDatasetRow(
                packed_state=state.encode_network_input(position).payload,
                action_ids=tuple(entry.action_id for entry in ordered_entries),
                probabilities=tuple(entry.probability for entry in ordered_entries),
                legal_action_ids=tuple(sorted(legal_actions)),
                top_action_id=book_position.preferred_action_id,
                source_game_id=source_game_id,
                ply=len(book_position.action_ids),
            )
        )
        source_games.append(
            EvaluationSourceGame(
                source_game_id=source_game_id,
                action_ids=book_position.action_ids,
                human_readable=engine.render_game(book_position.action_ids),
            )
        )
        book_positions.append(
            KataGoBookDatasetPositionProvenance(
                source_game_id=source_game_id,
                book_node_id=book_position.node_id,
                applied_symmetry=oriented_position.applied_symmetry,
            )
        )
    maximum_policy_entries = max(len(row.action_ids) for row in rows)
    maximum_legal_actions = max(len(row.legal_action_ids) for row in rows)
    if maximum_policy_entries > 255:
        raise ValueError('Evaluation dataset sparse policies cannot exceed 255 entries.')
    dtype = _dataset_dtype(
        state.packed_plane_layout.payload_bytes,
        maximum_policy_entries,
        maximum_legal_actions,
    )
    resolved_rows = tuple(rows)
    _write_dataset(path, resolved_rows, dtype)
    manifest = KataGoBookDatasetManifest(
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        position_count=len(resolved_rows),
        source_games=tuple(source_games),
        book_positions=tuple(book_positions),
        maximum_policy_entries=maximum_policy_entries,
        maximum_legal_actions=maximum_legal_actions,
        packed_payload_bytes=state.packed_plane_layout.payload_bytes,
        row_layout_digest=_layout_digest(dtype),
        data_sha256=_file_sha256(path),
        book_export_sha256=configuration.source.selection.export_sha256,
        book_selection_sha256=selection_digest,
        book_source_root_url=export.source_root_url,
        book_source_updated_on=export.source_updated_on,
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(manifest_path, manifest.model_dump_json(indent=2) + '\n')
    return manifest


def load_evaluation_dataset(
    path: Path,
    manifest: AnyEvaluationDatasetManifest,
) -> npt.NDArray[np.void]:
    dtype = _dataset_dtype(
        manifest.packed_payload_bytes,
        manifest.maximum_policy_entries,
        manifest.maximum_legal_actions,
    )
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
    manifest = EVALUATION_DATASET_MANIFEST_ADAPTER.validate_json(
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
            policy_logits, _ = model(torch.from_numpy(decoded).to(device))
            policy_logits = policy_logits.float().cpu()
            for row_index, row in enumerate(batch):
                count = int(row['policy_count'])
                action_ids = torch.from_numpy(row['action_ids'][:count].astype(np.int64))
                targets = torch.from_numpy(row['probabilities'][:count].astype(np.float32))
                legal_count = int(row['legal_count'])
                if legal_count <= 0:
                    raise ValueError('Evaluation dataset row has no legal actions.')
                legal_action_ids = torch.from_numpy(row['legal_action_ids'][:legal_count].astype(np.int64))
                legal_logits = policy_logits[row_index, legal_action_ids]
                legal_log_probabilities = torch.log_softmax(legal_logits, dim=0)
                target_positions = torch.searchsorted(legal_action_ids, action_ids)
                if bool(torch.any(target_positions >= legal_count)):
                    raise ValueError('Evaluation policy target contains an action outside its legal set.')
                if not torch.equal(legal_action_ids[target_positions], action_ids):
                    raise ValueError('Evaluation policy target contains an action outside its legal set.')
                top_action = int(legal_action_ids[int(legal_logits.argmax())])
                correct += int(top_action == int(row['top_action_id']))
                cross_entropy -= float(torch.sum(targets * legal_log_probabilities[target_positions]).item())
    return FixedDatasetEvaluationResult(
        kind='fixed_dataset',
        job=job,
        position_count=manifest.position_count,
        source_game_count=len(manifest.source_games),
        top_action_accuracy=correct / manifest.position_count,
        policy_cross_entropy=cross_entropy / manifest.position_count,
        duration_seconds=time.monotonic() - started_at,
    )
