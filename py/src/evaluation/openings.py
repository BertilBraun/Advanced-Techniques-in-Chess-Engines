from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import exp, log
from pathlib import Path
from typing import Generic, TypeVar

from src.evaluation.configuration import ChessBookOpeningSource, EngineBeamOpeningSource, OpeningSuiteConfiguration
from src.evaluation.contracts import (
    ChessBookOpeningLine,
    ChessBookOpeningSuiteManifest,
    KataGoBookOpeningLine,
    KataGoBookOpeningSuiteManifest,
    OpeningLine,
    OpeningSuiteManifest,
)
from src.evaluation.engine import EnginePolicyProvider, EvaluationArtifactMetadata, validate_engine_policy
from src.evaluation.katago_book import (
    load_katago_book_export,
    orient_katago_book_positions,
    select_katago_book_positions,
    selection_sha256,
)
from src.games.contracts import GameStateContract
from src.games.chess.contract import ChessPosition, ChessStateContract
from src.util.atomic_file import write_text_atomically
from src.util.hashing import file_sha256


PositionT = TypeVar('PositionT')
OPENING_EXPANSION_PLIES = 4


@dataclass(frozen=True)
class OpeningCandidate(Generic[PositionT]):
    position: PositionT
    action_ids: tuple[int, ...]
    log_probability: float


def _position_digest(state: GameStateContract[PositionT], position: PositionT) -> str:
    return hashlib.sha256(state.encode_network_input(position).payload).hexdigest()


def _load_existing_opening_suite(
    path: Path,
    configuration: OpeningSuiteConfiguration,
    engine: EnginePolicyProvider[PositionT],
) -> OpeningSuiteManifest | None:
    if not path.exists():
        return None
    manifest = OpeningSuiteManifest.model_validate_json(path.read_text(encoding='utf-8'))
    expected = (
        manifest.game == engine.game_name
        and manifest.rules_digest == engine.rules_digest
        and manifest.representation_digest == engine.representation_digest
        and manifest.random_seed == configuration.source.random_seed
        and manifest.engine_identity == engine.engine_identity
        and manifest.engine_artifact_sha256 == engine.engine_artifact_sha256
        and manifest.label_search_limit == engine.label_search_limit
        and manifest.expanded_actions_per_position == configuration.source.expanded_actions_per_position
        and manifest.beam_width == configuration.source.beam_width
        and len(manifest.openings) == configuration.opening_count
    )
    if not expected:
        raise ValueError('Existing opening suite does not match its configured immutable provenance.')
    return manifest


def _expand_frontier(
    frontier: tuple[OpeningCandidate[PositionT], ...],
    configuration: OpeningSuiteConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
) -> tuple[OpeningCandidate[PositionT], ...]:
    candidates_by_position: dict[str, OpeningCandidate[PositionT]] = {}
    for candidate in frontier:
        legal_actions = state.legal_action_ids(candidate.position)
        policy = validate_engine_policy(engine.policy(candidate.position, candidate.action_ids), legal_actions)
        expanded_entries = sorted(
            policy.entries,
            key=lambda entry: (-entry.probability, entry.action_id),
        )[: configuration.source.expanded_actions_per_position]
        for entry in expanded_entries:
            child = state.child_position(candidate.position, entry.action_id)
            if state.natural_terminal_wdl(child) is not None:
                continue
            action_ids = (*candidate.action_ids, entry.action_id)
            expanded = OpeningCandidate(
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
    return tuple(
        sorted(
            candidates_by_position.values(),
            key=lambda candidate: (-candidate.log_probability, candidate.action_ids),
        )[: configuration.source.beam_width]
    )


def _select_openings(
    frontier: tuple[OpeningCandidate[PositionT], ...],
    configuration: OpeningSuiteConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
) -> tuple[OpeningLine, ...]:
    selected = frontier[: configuration.opening_count]
    if len(selected) != configuration.opening_count:
        raise ValueError('Engine-guided opening expansion did not produce enough distinct positions.')
    return tuple(
        OpeningLine(
            opening_id=f'opening-{index:03d}',
            action_ids=candidate.action_ids,
            path_probability=exp(candidate.log_probability),
            final_position_digest=_position_digest(state, candidate.position),
            human_readable=engine.render_game(candidate.action_ids),
        )
        for index, candidate in enumerate(selected)
    )


def build_opening_suite(
    path: Path,
    configuration: OpeningSuiteConfiguration,
    state: GameStateContract[PositionT],
    engine: EnginePolicyProvider[PositionT],
    builder_source_revision: str,
) -> OpeningSuiteManifest:
    if not isinstance(configuration.source, EngineBeamOpeningSource):
        raise ValueError('Engine-guided opening builder requires an engine-beam source.')
    existing = _load_existing_opening_suite(path, configuration, engine)
    if existing is not None:
        return existing
    frontier = (OpeningCandidate(state.initial_position(), (), 0.0),)
    for _ in range(OPENING_EXPANSION_PLIES):
        frontier = _expand_frontier(frontier, configuration, state, engine)
        if not frontier:
            raise ValueError('Engine-guided opening expansion produced no nonterminal positions.')
    openings = _select_openings(frontier, configuration, state, engine)
    manifest = OpeningSuiteManifest(
        game=engine.game_name,
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        random_seed=configuration.source.random_seed,
        engine_identity=engine.engine_identity,
        engine_artifact_sha256=engine.engine_artifact_sha256,
        label_search_limit=engine.label_search_limit,
        expanded_actions_per_position=configuration.source.expanded_actions_per_position,
        beam_width=configuration.source.beam_width,
        openings=openings,
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(path, manifest.model_dump_json(indent=2) + '\n')
    return manifest


@dataclass(frozen=True)
class ChessOpeningSuiteRow:
    opening_id: str
    moves_uci: tuple[str, ...]
    final_fen: str


def load_chess_opening_suite(path: Path) -> tuple[ChessOpeningSuiteRow, ...]:
    rows: list[ChessOpeningSuiteRow] = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line or line.startswith('#'):
            continue
        columns = line.split('\t')
        if len(columns) != 3:
            raise ValueError(f'Chess opening suite line {line_number} must contain three tab-separated columns.')
        opening_id, moves, final_fen = columns
        moves_uci = tuple(moves.split())
        if not opening_id or not moves_uci or not final_fen:
            raise ValueError(f'Chess opening suite line {line_number} contains an empty field.')
        rows.append(ChessOpeningSuiteRow(opening_id=opening_id, moves_uci=moves_uci, final_fen=final_fen))
    if len({row.opening_id for row in rows}) != len(rows):
        raise ValueError('Chess opening suite IDs must be unique.')
    return tuple(rows)


def _replay_chess_book_line(state: ChessStateContract, row: ChessOpeningSuiteRow) -> ChessBookOpeningLine:
    position: ChessPosition = state.initial_position()
    action_ids: list[int] = []
    for move_uci in row.moves_uci:
        action_id = position.action_id_from_uci(move_uci)
        if action_id not in state.legal_action_ids(position):
            raise ValueError(f'Chess book opening {row.opening_id!r} contains illegal move {move_uci!r}.')
        action_ids.append(action_id)
        position = state.child_position(position, action_id)
    if position.fen != row.final_fen:
        raise ValueError(f'Chess book opening {row.opening_id!r} does not reproduce its expected FEN.')
    if state.natural_terminal_wdl(position) is not None:
        raise ValueError(f'Chess book opening {row.opening_id!r} ends in a terminal position.')
    return ChessBookOpeningLine(
        opening_id=row.opening_id,
        action_ids=tuple(action_ids),
        final_position_digest=_position_digest(state, position),
        human_readable=' '.join(row.moves_uci),
    )


def build_chess_book_opening_suite(
    path: Path,
    selection_path: Path,
    configuration: OpeningSuiteConfiguration,
    state: ChessStateContract,
    metadata: EvaluationArtifactMetadata[ChessPosition],
    builder_source_revision: str,
) -> ChessBookOpeningSuiteManifest:
    source = configuration.source
    if not isinstance(source, ChessBookOpeningSource):
        raise ValueError('Chess book opening builder requires a chess-book source.')
    if not selection_path.is_file() or file_sha256(selection_path) != source.selection_sha256:
        raise ValueError('Chess book selection is missing or does not match its configured SHA-256.')
    if path.exists():
        manifest = ChessBookOpeningSuiteManifest.model_validate_json(path.read_text(encoding='utf-8'))
        expected = (
            manifest.rules_digest == metadata.rules_digest
            and manifest.representation_digest == metadata.representation_digest
            and manifest.selection_sha256 == source.selection_sha256
            and len(manifest.openings) == configuration.opening_count
        )
        if not expected:
            raise ValueError('Existing chess book opening suite does not match its configured immutable provenance.')
        return manifest
    rows = load_chess_opening_suite(selection_path)
    if len(rows) != configuration.opening_count:
        raise ValueError('Chess book selection count does not match the configured opening count.')
    manifest = ChessBookOpeningSuiteManifest(
        rules_digest=metadata.rules_digest,
        representation_digest=metadata.representation_digest,
        selection_sha256=source.selection_sha256,
        openings=tuple(_replay_chess_book_line(state, row) for row in rows),
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(path, manifest.model_dump_json(indent=2) + '\n')
    return manifest


def build_katago_book_opening_suite(
    path: Path,
    export_path: Path,
    configuration: OpeningSuiteConfiguration,
    state: GameStateContract[PositionT],
    engine: EvaluationArtifactMetadata[PositionT],
    builder_source_revision: str,
) -> KataGoBookOpeningSuiteManifest:
    if configuration.source.kind != 'katago_book':
        raise ValueError('KataGo book opening builder requires a book source.')
    selection_digest = selection_sha256(configuration.source.selection)
    if path.exists():
        manifest = KataGoBookOpeningSuiteManifest.model_validate_json(path.read_text(encoding='utf-8'))
        expected = (
            manifest.rules_digest == engine.rules_digest
            and manifest.representation_digest == engine.representation_digest
            and manifest.book_export_sha256 == configuration.source.selection.export_sha256
            and manifest.book_selection_sha256 == selection_digest
            and len(manifest.openings) == configuration.opening_count
        )
        if not expected:
            raise ValueError('Existing book opening suite does not match its configured immutable provenance.')
        return manifest

    export = load_katago_book_export(export_path, configuration.source.selection.export_sha256)
    selected = orient_katago_book_positions(
        select_katago_book_positions(export, configuration.source.selection, configuration.opening_count)
    )
    openings: list[KataGoBookOpeningLine] = []
    position_digests: set[str] = set()
    for position_index, oriented_position in enumerate(selected):
        book_position = oriented_position.position
        position = state.initial_position()
        for action_id in book_position.action_ids:
            if action_id not in state.legal_action_ids(position):
                raise ValueError(f'KataGo book path contains illegal action {action_id}.')
            position = state.child_position(position, action_id)
        if state.natural_terminal_wdl(position) is not None:
            raise ValueError('KataGo book opening ends at a natural terminal position.')
        digest = _position_digest(state, position)
        if digest in position_digests:
            raise ValueError('KataGo book selection produced duplicate native positions.')
        position_digests.add(digest)
        openings.append(
            KataGoBookOpeningLine(
                opening_id=f'opening-{position_index:03d}',
                action_ids=book_position.action_ids,
                path_probability=book_position.path_probability,
                final_position_digest=digest,
                human_readable=engine.render_game(book_position.action_ids),
                book_node_id=book_position.node_id,
                applied_symmetry=oriented_position.applied_symmetry,
                black_win_probability=book_position.black_win_probability,
                black_score=book_position.black_score,
                win_probability_uncertainty=book_position.win_probability_uncertainty,
                score_uncertainty=book_position.score_uncertainty,
                book_visits=book_position.visits,
            )
        )
    manifest = KataGoBookOpeningSuiteManifest(
        rules_digest=engine.rules_digest,
        representation_digest=engine.representation_digest,
        book_export_sha256=configuration.source.selection.export_sha256,
        book_selection_sha256=selection_digest,
        book_source_root_url=export.source_root_url,
        book_source_updated_on=export.source_updated_on,
        openings=tuple(openings),
        builder_source_revision=builder_source_revision,
    )
    write_text_atomically(path, manifest.model_dump_json(indent=2) + '\n')
    return manifest
