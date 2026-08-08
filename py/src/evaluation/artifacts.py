from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import exp, log
from pathlib import Path
import random
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from src.evaluation.configuration import EvaluationDatasetConfiguration, OpeningSuiteConfiguration
from src.evaluation.contracts import EvaluationDatasetManifest, OpeningLine, OpeningSuiteManifest
from src.evaluation.engine import EnginePolicy, EnginePolicyProvider
from src.games.contracts import GameStateContract
from src.util.atomic_file import write_text_atomically


PositionT = TypeVar('PositionT')
RETAINED_PLY_INTERVAL = 3
MINIMUM_DATASET_POSITIONS = 480
MAXIMUM_DATASET_POSITIONS = 520
OPENING_EXPANSION_PLIES = 4
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
class _OpeningCandidate(Generic[PositionT]):
    position: PositionT
    action_ids: tuple[int, ...]
    log_probability: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _position_digest(state: GameStateContract[PositionT], position: PositionT) -> str:
    return hashlib.sha256(state.encode_network_input(position).payload).hexdigest()


def _normalized_policy(policy: EnginePolicy, legal_actions: tuple[int, ...]) -> EnginePolicy:
    legal = set(legal_actions)
    if any(entry.action_id not in legal for entry in policy.entries):
        raise ValueError('Engine policy contains an illegal action.')
    return policy


def build_opening_suite(
    path: Path,
    configuration: OpeningSuiteConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
    builder_source_revision: str,
) -> OpeningSuiteManifest:
    if path.exists():
        return OpeningSuiteManifest.model_validate_json(path.read_text(encoding='utf-8'))
    frontier = (_OpeningCandidate(state.initial_position(), (), 0.0),)
    for _ in range(OPENING_EXPANSION_PLIES):
        candidates_by_position: dict[str, _OpeningCandidate[PositionT]] = {}
        for candidate in frontier:
            legal_actions = state.legal_action_ids(candidate.position)
            policy = _normalized_policy(engine.policy(candidate.position, candidate.action_ids), legal_actions)
            expanded_entries = sorted(
                policy.entries,
                key=lambda entry: (-entry.probability, entry.action_id),
            )[: configuration.expanded_actions_per_position]
            for entry in expanded_entries:
                child = state.child_position(candidate.position, entry.action_id)
                if state.natural_terminal_wdl(child) is not None:
                    continue
                action_ids = (*candidate.action_ids, entry.action_id)
                expanded = _OpeningCandidate(
                    position=child,
                    action_ids=action_ids,
                    log_probability=candidate.log_probability + log(entry.probability),
                )
                digest = _position_digest(state, child)
                previous = candidates_by_position.get(digest)
                if previous is None or (expanded.log_probability, tuple(-value for value in action_ids)) > (
                    previous.log_probability,
                    tuple(-value for value in previous.action_ids),
                ):
                    candidates_by_position[digest] = expanded
        frontier = tuple(
            sorted(
                candidates_by_position.values(),
                key=lambda candidate: (-candidate.log_probability, candidate.action_ids),
            )[: configuration.beam_width]
        )
        if not frontier:
            raise ValueError('Engine-guided opening expansion produced no nonterminal positions.')
    selected = frontier[: configuration.opening_count]
    if len(selected) != configuration.opening_count:
        raise ValueError('Engine-guided opening expansion did not produce enough distinct positions.')
    openings = tuple(
        OpeningLine(
            opening_id=f'opening-{index:03d}',
            action_ids=candidate.action_ids,
            path_probability=exp(candidate.log_probability),
            final_position_digest=_position_digest(state, candidate.position),
            human_readable=engine.render_game(candidate.action_ids),
        )
        for index, candidate in enumerate(selected)
    )
    manifest = OpeningSuiteManifest(
        game=engine.game_name,
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        random_seed=configuration.random_seed,
        engine_identity=engine.engine_identity,
        engine_artifact_sha256=engine.engine_artifact_sha256,
        label_search_limit=engine.label_search_limit,
        expanded_actions_per_position=configuration.expanded_actions_per_position,
        beam_width=configuration.beam_width,
        openings=openings,
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(path, manifest.model_dump_json(indent=2) + '\n')
    return manifest


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


def build_evaluation_dataset(
    path: Path,
    configuration: EvaluationDatasetConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
    builder_source_revision: str,
) -> EvaluationDatasetManifest:
    manifest_path = dataset_manifest_path(path)
    if path.exists() and manifest_path.exists():
        manifest = EvaluationDatasetManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        if file_sha256(path) != manifest.data_sha256:
            raise ValueError('Evaluation dataset hash does not match its manifest.')
        return manifest
    if path.exists() or manifest_path.exists():
        raise ValueError('Evaluation dataset data and manifest must either both exist or both be absent.')
    generator = random.Random(configuration.random_seed)
    retained_offset = configuration.random_seed % RETAINED_PLY_INTERVAL
    rows: list[EvaluationDatasetRow] = []
    position_digests: set[str] = set()
    source_game_count = 0
    while len(rows) < MINIMUM_DATASET_POSITIONS:
        if source_game_count >= MAXIMUM_SOURCE_GAMES:
            raise ValueError('Evaluation dataset did not reach its minimum position count.')
        position = state.initial_position()
        action_ids: tuple[int, ...] = ()
        ply = 0
        while True:
            legal_actions = state.legal_action_ids(position)
            if not legal_actions:
                break
            policy = _normalized_policy(engine.policy(position, action_ids), legal_actions)
            digest = _position_digest(state, position)
            if (
                ply % RETAINED_PLY_INTERVAL == retained_offset
                and digest not in position_digests
                and len(rows) < MAXIMUM_DATASET_POSITIONS
            ):
                ordered_entries = tuple(sorted(policy.entries, key=lambda entry: (-entry.probability, entry.action_id)))
                rows.append(
                    EvaluationDatasetRow(
                        packed_state=state.encode_network_input(position).payload,
                        action_ids=tuple(entry.action_id for entry in ordered_entries),
                        probabilities=tuple(entry.probability for entry in ordered_entries),
                        top_action_id=policy.top_action_id,
                        source_game_id=source_game_count,
                        ply=ply,
                    )
                )
                position_digests.add(digest)
            selected_action = _sample_action(
                policy,
                configuration.move_sampling_temperature,
                generator,
            )
            position = state.child_position(position, selected_action)
            action_ids = (*action_ids, selected_action)
            ply += 1
        source_game_count += 1
    maximum_policy_entries = max(len(row.action_ids) for row in rows)
    if maximum_policy_entries > 255:
        raise ValueError('Evaluation dataset sparse policies cannot exceed 255 entries.')
    dtype = _dataset_dtype(state.packed_plane_layout.payload_bytes, maximum_policy_entries)
    resolved_rows = tuple(rows)
    _write_dataset(path, resolved_rows, dtype)
    manifest = EvaluationDatasetManifest(
        game=engine.game_name,
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        position_count=len(resolved_rows),
        source_game_count=source_game_count,
        retained_ply_offset=retained_offset,
        random_seed=configuration.random_seed,
        move_sampling_temperature=configuration.move_sampling_temperature,
        engine_identity=engine.engine_identity,
        engine_artifact_sha256=engine.engine_artifact_sha256,
        label_search_limit=engine.label_search_limit,
        maximum_policy_entries=maximum_policy_entries,
        packed_payload_bytes=state.packed_plane_layout.payload_bytes,
        row_layout_digest=_layout_digest(dtype),
        data_sha256=file_sha256(path),
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
