from __future__ import annotations

import ast
import hashlib
import heapq
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from pydantic import Field
from src.evaluation.configuration import KataGoBookSelectionConfiguration
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel

OFFICIAL_9X9_TT_ROOT_URL = 'https://katagobooks.org/book9x9tt/root/root.html'
OFFICIAL_9X9_TT_UPDATED_ON = date(2026, 2, 26)
_CONSTANT_PATTERN = re.compile(r'const\s+(?P<name>\w+)\s*=\s*(?P<value>.*?);', re.DOTALL)


class KataGoBookPageProvenance(FrozenModel):
    url: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)


class KataGoBookPolicyEntry(FrozenModel):
    action_id: int = Field(ge=0)
    probability: float = Field(gt=0.0, le=1.0)


class KataGoBookPosition(FrozenModel):
    node_id: str = Field(min_length=1)
    root_variation_id: str = Field(min_length=1)
    action_ids: tuple[int, ...] = Field(min_length=1)
    path_probability: float = Field(ge=0.0, le=1.0)
    black_win_probability: float = Field(ge=0.0, le=1.0)
    black_score: float
    win_probability_uncertainty: float = Field(ge=0.0, le=0.5)
    score_uncertainty: float = Field(ge=0.0)
    visits: int = Field(gt=0)
    preferred_action_id: int = Field(ge=0)
    policy: tuple[KataGoBookPolicyEntry, ...] = Field(min_length=1)


class KataGoBookExport(FrozenModel):
    schema_version: Literal[2] = 2
    source_root_url: str = Field(min_length=1)
    source_updated_on: date
    board_size: Literal[9] = 9
    rules: Literal['tromp-taylor'] = 'tromp-taylor'
    komi_half_points: Literal[14] = 14
    maximum_crawl_depth: int = Field(gt=0)
    maximum_crawl_pages: int = Field(gt=0)
    pages: tuple[KataGoBookPageProvenance, ...] = Field(min_length=1)
    positions: tuple[KataGoBookPosition, ...] = Field(min_length=1)


@dataclass(frozen=True)
class OrientedKataGoBookPosition:
    position: KataGoBookPosition
    applied_symmetry: int


@dataclass(frozen=True)
class _BookMove:
    action_id: int
    child_url: str | None
    child_symmetry: int | None
    policy_prior: float
    black_win_probability: float
    black_score: float
    win_probability_uncertainty: float
    score_uncertainty: float
    visits: int
    preference_index: int


@dataclass(frozen=True)
class _BookPage:
    url: str
    sha256: str
    moves: tuple[_BookMove, ...]

    def normalized_policy(self) -> tuple[KataGoBookPolicyEntry, ...]:
        positive_moves = tuple(move for move in self.moves if move.policy_prior > 0.0)
        total_probability = sum(move.policy_prior for move in positive_moves)
        if total_probability <= 0.0:
            raise ValueError('KataGo book page contains no positive policy priors.')
        return tuple(
            KataGoBookPolicyEntry(
                action_id=move.action_id,
                probability=move.policy_prior / total_probability,
            )
            for move in positive_moves
        )


@dataclass(frozen=True)
class _CrawlEntry:
    url: str
    symmetry: int
    action_ids: tuple[int, ...]
    path_probability: float
    root_variation_id: str | None
    black_win_probability: float
    black_score: float
    win_probability_uncertainty: float
    score_uncertainty: float
    visits: int


def _crawl_priority(
    entry: _CrawlEntry,
    fetched_pages_by_root: dict[str, int],
    fetched_pages_by_bucket: dict[tuple[str, int], int],
    identity: int,
) -> tuple[int, int, tuple[float, float, float, float, int, float], tuple[int, ...], int, _CrawlEntry]:
    if entry.root_variation_id is None:
        return (0, 0, (0.0, 0.0, 0.0, 0.0, 0, 0.0), (), identity, entry)
    depth = len(entry.action_ids)
    quality = (
        abs(entry.black_win_probability - 0.5),
        abs(entry.black_score),
        entry.win_probability_uncertainty,
        entry.score_uncertainty,
        -entry.visits,
        -entry.path_probability,
    )
    return (
        fetched_pages_by_root[entry.root_variation_id],
        fetched_pages_by_bucket[(entry.root_variation_id, depth)],
        quality,
        entry.action_ids,
        identity,
        entry,
    )


def _literal(constants: dict[str, str], name: str) -> object:
    value = constants.get(name)
    if value is None:
        raise ValueError(f'KataGo book page omitted {name}.')
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f'KataGo book page has invalid {name}.') from error


def _transform_action(action_id: int, symmetry: int, board_size: int = 9) -> int:
    if action_id == board_size * board_size:
        return action_id
    x = action_id % board_size
    y = action_id // board_size
    if symmetry & 1:
        y = board_size - 1 - y
    if symmetry & 2:
        x = board_size - 1 - x
    if symmetry & 4:
        x, y = y, x
    return x + y * board_size


def _compose_symmetry(first: int, second: int) -> int:
    if first & 4:
        second = (second & 4) | ((second & 2) >> 1) | ((second & 1) << 1)
    return first ^ second


def _parse_book_page(url: str, content: bytes, current_symmetry: int) -> _BookPage:
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError('KataGo book page is not UTF-8.') from error
    constants = {match.group('name'): match.group('value') for match in _CONSTANT_PATTERN.finditer(text)}
    links = _literal(constants, 'links')
    link_symmetries = _literal(constants, 'linkSyms')
    moves = _literal(constants, 'moves')
    if not isinstance(links, dict) or not isinstance(link_symmetries, dict) or not isinstance(moves, list):
        raise ValueError('KataGo book page has invalid link or move data.')

    parsed_moves: list[_BookMove] = []
    for preference_index, raw_move in enumerate(moves):
        if not isinstance(raw_move, dict):
            raise ValueError('KataGo book move entry must be a mapping.')
        coordinates = raw_move.get('xy')
        if coordinates is None:
            coordinate_pairs = ((9, 9),) if raw_move.get('move') == 'pass' else ()
        elif isinstance(coordinates, list):
            coordinate_pairs = tuple(tuple(pair) for pair in coordinates)
        else:
            raise ValueError('KataGo book move coordinates must be a list.')
        for coordinate_pair in coordinate_pairs:
            if len(coordinate_pair) != 2 or not all(isinstance(value, int) for value in coordinate_pair):
                raise ValueError('KataGo book move coordinate is invalid.')
            x, y = coordinate_pair
            canonical_action = 81 if (x, y) == (9, 9) else x + y * 9
            child_link = links.get(canonical_action)
            child_symmetry = link_symmetries.get(canonical_action)
            if not isinstance(child_link, str) or not isinstance(child_symmetry, int):
                child_link = None
                child_symmetry = None
            parsed_moves.append(
                _BookMove(
                    action_id=_transform_action(canonical_action, current_symmetry),
                    child_url=None if child_link is None else urljoin(url, child_link),
                    child_symmetry=(
                        None if child_symmetry is None else _compose_symmetry(child_symmetry, current_symmetry)
                    ),
                    policy_prior=max(float(raw_move['p']), 0.0),
                    black_win_probability=0.5 * (1.0 - float(raw_move['wl'])),
                    black_score=-float(raw_move['ssM']),
                    win_probability_uncertainty=0.5 * float(raw_move['wlRad']),
                    score_uncertainty=float(raw_move['sRad']),
                    visits=round(float(raw_move['v'])),
                    preference_index=preference_index,
                )
            )
    return _BookPage(
        url=url,
        sha256=hashlib.sha256(content).hexdigest(),
        moves=tuple(parsed_moves),
    )


def _fetch_page_content(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'katagobooks.org':
        raise ValueError('KataGo book crawl is restricted to official HTTPS pages.')
    request = Request(url, headers={'User-Agent': 'AlphaZero-evaluation-artifact-builder/1.0'})
    with urlopen(request, timeout=30) as response:
        return response.read()


def crawl_official_9x9_book(
    maximum_depth: int,
    maximum_pages: int,
    progress: Callable[[int, int], None] | None = None,
    page_reader: Callable[[str], bytes] | None = None,
) -> KataGoBookExport:
    if maximum_depth <= 0 or maximum_pages <= 0:
        raise ValueError('KataGo book crawl bounds must be positive.')
    initial_entry = _CrawlEntry(OFFICIAL_9X9_TT_ROOT_URL, 0, (), 1.0, None, 0.5, 0.0, 0.0, 0.0, 1)
    queue = [_crawl_priority(initial_entry, {}, {}, 0)]
    next_queue_identity = 1
    fetched_pages_by_root: dict[str, int] = defaultdict(int)
    fetched_pages_by_bucket: dict[tuple[str, int], int] = defaultdict(int)
    fetched: dict[str, _BookPage] = {}
    page_content_by_url: dict[str, bytes] = {}
    positions_by_node: dict[str, KataGoBookPosition] = {}
    discovered_nodes: set[str] = set()
    resolved_preferred_nodes: set[str] = set()
    while queue and len(fetched) < maximum_pages:
        queued_root_pages, queued_bucket_pages, _, _, _, entry = heapq.heappop(queue)
        if entry.root_variation_id is not None:
            current_root_pages = fetched_pages_by_root[entry.root_variation_id]
            bucket = (entry.root_variation_id, len(entry.action_ids))
            current_bucket_pages = fetched_pages_by_bucket[bucket]
            if queued_root_pages != current_root_pages or queued_bucket_pages != current_bucket_pages:
                heapq.heappush(
                    queue,
                    _crawl_priority(
                        entry,
                        fetched_pages_by_root,
                        fetched_pages_by_bucket,
                        next_queue_identity,
                    ),
                )
                next_queue_identity += 1
                continue
        if entry.url in fetched:
            continue
        content = page_content_by_url.get(entry.url)
        if content is None:
            content = _fetch_page_content(entry.url) if page_reader is None else page_reader(entry.url)
            page_content_by_url[entry.url] = content
        page = _parse_book_page(entry.url, content, entry.symmetry)
        fetched[entry.url] = page
        if entry.root_variation_id is not None:
            fetched_pages_by_root[entry.root_variation_id] += 1
            fetched_pages_by_bucket[(entry.root_variation_id, len(entry.action_ids))] += 1
        if progress is not None and len(fetched) % 100 == 0:
            progress(len(fetched), len(discovered_nodes))
        node_id = urlparse(entry.url).path
        preferred_moves = tuple(move for move in page.moves if move.preference_index == 0)
        if entry.root_variation_id is not None and preferred_moves:
            positions_by_node[node_id] = KataGoBookPosition(
                node_id=node_id,
                root_variation_id=entry.root_variation_id,
                action_ids=entry.action_ids,
                path_probability=entry.path_probability,
                black_win_probability=entry.black_win_probability,
                black_score=entry.black_score,
                win_probability_uncertainty=entry.win_probability_uncertainty,
                score_uncertainty=entry.score_uncertainty,
                visits=entry.visits,
                preferred_action_id=min(move.action_id for move in preferred_moves),
                policy=page.normalized_policy(),
            )
            resolved_preferred_nodes.add(node_id)
        if len(entry.action_ids) >= maximum_depth:
            continue
        for move in page.moves:
            if move.child_url is None or move.child_symmetry is None:
                continue
            action_ids = (*entry.action_ids, move.action_id)
            child_path = urlparse(move.child_url).path
            node_id = child_path
            root_variation_id = entry.root_variation_id or child_path
            path_probability = entry.path_probability * move.policy_prior
            discovered_nodes.add(node_id)
            child_entry = _CrawlEntry(
                move.child_url,
                move.child_symmetry,
                action_ids,
                path_probability,
                root_variation_id,
                move.black_win_probability,
                move.black_score,
                move.win_probability_uncertainty,
                move.score_uncertainty,
                move.visits,
            )
            heapq.heappush(
                queue,
                _crawl_priority(
                    child_entry,
                    fetched_pages_by_root,
                    fetched_pages_by_bucket,
                    next_queue_identity,
                ),
            )
            next_queue_identity += 1
    pages = tuple(
        KataGoBookPageProvenance(url=page.url, sha256=page.sha256)
        for page in sorted(fetched.values(), key=lambda page: page.url)
    )
    return KataGoBookExport(
        source_root_url=OFFICIAL_9X9_TT_ROOT_URL,
        source_updated_on=OFFICIAL_9X9_TT_UPDATED_ON,
        maximum_crawl_depth=maximum_depth,
        maximum_crawl_pages=maximum_pages,
        pages=pages,
        positions=tuple(
            sorted(
                (position for node_id, position in positions_by_node.items() if node_id in resolved_preferred_nodes),
                key=lambda position: position.action_ids,
            )
        ),
    )


def write_katago_book_export(path: Path, export: KataGoBookExport) -> None:
    write_text_atomically(path, export.model_dump_json(indent=2) + '\n')


def load_katago_book_export(path: Path, expected_sha256: str) -> KataGoBookExport:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('KataGo book export hash does not match configuration.')
    return KataGoBookExport.model_validate_json(content)


def selection_sha256(configuration: KataGoBookSelectionConfiguration) -> str:
    canonical_selection = configuration.model_dump_json(exclude={'export_path'})
    return hashlib.sha256(canonical_selection.encode('utf-8')).hexdigest()


def select_katago_book_positions(
    export: KataGoBookExport,
    configuration: KataGoBookSelectionConfiguration,
    count: int,
) -> tuple[KataGoBookPosition, ...]:
    if count <= 0:
        raise ValueError('KataGo book selection count must be positive.')
    eligible = tuple(
        position
        for position in export.positions
        if configuration.minimum_ply <= len(position.action_ids) <= configuration.maximum_ply
        and abs(position.black_score) <= configuration.maximum_absolute_black_score
        and abs(position.black_win_probability - 0.5) <= configuration.maximum_black_win_probability_deviation
        and position.win_probability_uncertainty <= configuration.maximum_win_probability_uncertainty
        and position.score_uncertainty <= configuration.maximum_score_uncertainty
        and position.visits >= configuration.minimum_visits
    )
    buckets: dict[tuple[str, int], list[KataGoBookPosition]] = defaultdict(list)
    for position in eligible:
        buckets[(position.root_variation_id, len(position.action_ids))].append(position)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda position: (
                abs(position.black_win_probability - 0.5),
                abs(position.black_score),
                position.win_probability_uncertainty,
                position.score_uncertainty,
                -position.visits,
                -position.path_probability,
                position.action_ids,
            )
        )
    selected: list[KataGoBookPosition] = []
    ordered_keys = sorted(buckets)
    while len(selected) < count and ordered_keys:
        next_keys: list[tuple[str, int]] = []
        for key in ordered_keys:
            bucket = buckets[key]
            selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
            if len(selected) == count:
                break
        ordered_keys = next_keys
    if len(selected) != count:
        raise ValueError(f'KataGo book selection found {len(selected)} eligible positions; {count} required.')
    return tuple(selected)


def orient_katago_book_positions(
    positions: tuple[KataGoBookPosition, ...],
) -> tuple[OrientedKataGoBookPosition, ...]:
    return tuple(
        OrientedKataGoBookPosition(
            position.model_copy(
                update={
                    'action_ids': tuple(_transform_action(action_id, index % 8) for action_id in position.action_ids),
                    'preferred_action_id': _transform_action(position.preferred_action_id, index % 8),
                    'policy': tuple(
                        entry.model_copy(update={'action_id': _transform_action(entry.action_id, index % 8)})
                        for entry in position.policy
                    ),
                }
            ),
            index % 8,
        )
        for index, position in enumerate(positions)
    )


def canonical_json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
