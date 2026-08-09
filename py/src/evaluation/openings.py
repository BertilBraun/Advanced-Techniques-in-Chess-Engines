from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import exp, log
from pathlib import Path
from typing import Generic, TypeVar

from src.evaluation.configuration import OpeningSuiteConfiguration
from src.evaluation.contracts import OpeningLine, OpeningSuiteManifest
from src.evaluation.engine import EnginePolicyProvider, validate_engine_policy
from src.games.contracts import GameStateContract
from src.util.atomic_file import write_text_atomically


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
        and manifest.random_seed == configuration.random_seed
        and manifest.engine_identity == engine.engine_identity
        and manifest.engine_artifact_sha256 == engine.engine_artifact_sha256
        and manifest.label_search_limit == engine.label_search_limit
        and manifest.expanded_actions_per_position == configuration.expanded_actions_per_position
        and manifest.beam_width == configuration.beam_width
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
        )[: configuration.expanded_actions_per_position]
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
        )[: configuration.beam_width]
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
