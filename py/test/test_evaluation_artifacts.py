from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import log
from pathlib import Path
import random

import pytest
import torch

import src.evaluation.dataset as dataset_module
import src.evaluation.katago_book as katago_book_module
from src.evaluation.dataset import (
    build_evaluation_dataset,
    build_katago_book_evaluation_dataset,
    dataset_manifest_path,
    load_evaluation_dataset,
)
from src.evaluation.openings import build_katago_book_opening_suite, build_opening_suite
from src.evaluation.configuration import (
    EngineBeamOpeningSource,
    EngineSelfPlayDatasetSource,
    EvaluationDatasetConfiguration,
    FixedDatasetEvaluationDefinition,
    KataGoBookDatasetSource,
    KataGoBookOpeningSource,
    KataGoBookSelectionConfiguration,
    OpeningSuiteConfiguration,
)
from src.evaluation.contracts import EvaluationDatasetManifest, FixedDatasetEvaluationJob, OpeningSuiteManifest
from src.evaluation.dataset import evaluate_fixed_dataset
from src.evaluation.engine import EnginePolicy, EnginePolicyEntry
from src.evaluation.katago_book import (
    KataGoBookExport,
    KataGoBookPageProvenance,
    KataGoBookPosition,
    canonical_json_sha256,
    select_katago_book_positions,
    write_katago_book_export,
)
from src.experiment.configuration import load_experiment_configuration
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import PackedPlaneLayout, PackedPlanePayload, RepresentationDimensions
from src.replay.contracts import ReplaySample
from src.self_play.completed_game import TerminationReason
from src.training.checkpoint import CheckpointReference


def test_checked_in_go_baseline_reference_artifacts_match_configuration() -> None:
    experiment = load_experiment_configuration(Path('configs/baselines/vast-go-7x7-2gpu-4h.yaml'))
    dataset_path = Path(experiment.evaluation.dataset.path).relative_to('py')
    openings_path = Path(experiment.evaluation.openings.path).relative_to('py')
    dataset_manifest = EvaluationDatasetManifest.model_validate_json(
        dataset_manifest_path(dataset_path).read_text(encoding='utf-8')
    )
    opening_manifest = OpeningSuiteManifest.model_validate_json(openings_path.read_text(encoding='utf-8'))
    engine = experiment.evaluation.engine
    assert engine.kind == 'katago'

    assert 480 <= dataset_manifest.position_count <= 520
    assert dataset_manifest.label_search_limit == engine.label_max_visits
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == dataset_manifest.data_sha256
    assert len(opening_manifest.openings) == experiment.evaluation.openings.opening_count == 200
    assert opening_manifest.label_search_limit == engine.label_max_visits


def test_checked_in_go_9x9_book_export_matches_configuration() -> None:
    experiment = load_experiment_configuration(Path('configs/go-9x9-experiment-template.yaml'))
    opening_source = experiment.evaluation.openings.source
    dataset_source = experiment.evaluation.dataset.source
    assert opening_source.kind == 'katago_book'
    assert dataset_source.kind == 'katago_book'
    export_path = Path(opening_source.selection.export_path).relative_to('py')
    export = KataGoBookExport.model_validate_json(export_path.read_text(encoding='utf-8'))

    assert canonical_json_sha256(export_path) == opening_source.selection.export_sha256
    assert opening_source.selection.export_sha256 == dataset_source.selection.export_sha256
    assert export.source_root_url == 'https://katagobooks.org/book9x9tt/root/root.html'
    assert export.source_updated_on.isoformat() == '2026-02-26'
    assert len(export.pages) == 1000
    assert len(export.positions) == 791
    openings = select_katago_book_positions(export, opening_source.selection, 200)
    assert len(openings) == 200
    assert {len(position.action_ids) for position in openings} == set(range(4, 13))
    assert len({position.root_variation_id for position in openings}) == 8
    assert max(abs(position.black_win_probability - 0.5) for position in openings) <= 0.0177
    assert max(abs(position.black_score) for position in openings) <= 0.52
    assert len(select_katago_book_positions(export, dataset_source.selection, 500)) == 500


def test_katago_book_crawl_parses_bounded_official_page_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    root = b"""const links = {0: 'child.html'};
const linkSyms = {0: 0};
const moves = [{'xy': [[0, 0]], 'p': 0.4, 'wl': 0.0, 'ssM': 0.0, 'wlRad': 0.1, 'sRad': 0.2, 'v': 20000}];"""
    child = b"""const links = {1: 'grandchild.html'};
const linkSyms = {1: 0};
const moves = [{'xy': [[1, 0]], 'p': 0.5, 'wl': 0.2, 'ssM': -0.3, 'wlRad': 0.1, 'sRad': 0.2, 'v': 18000}];"""
    content_by_url = {
        katago_book_module.OFFICIAL_9X9_TT_ROOT_URL: root,
        'https://katagobooks.org/book9x9tt/root/child.html': child,
    }

    def fetch_fixture(url: str) -> bytes:
        return content_by_url[url]

    monkeypatch.setattr(katago_book_module, '_fetch_page_content', fetch_fixture)
    export = katago_book_module.crawl_official_9x9_book(maximum_depth=2, maximum_pages=2)

    assert len(export.pages) == 2
    assert len(export.positions) == 1
    assert export.positions[0].action_ids == (0,)
    assert export.positions[0].preferred_action_id == 1
    assert export.positions[0].black_win_probability == 0.5


def test_katago_book_crawl_keeps_transposed_path_and_preferred_move_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = b"""const links = {0: 'child.html', 8: 'child.html'};
const linkSyms = {0: 0, 8: 2};
const moves = [
{'xy': [[0, 0]], 'p': 0.1, 'wl': 0.0, 'ssM': 0.0, 'wlRad': 0.1, 'sRad': 0.2, 'v': 20000},
{'xy': [[8, 0]], 'p': 0.9, 'wl': 0.8, 'ssM': -8.0, 'wlRad': 0.1, 'sRad': 0.2, 'v': 20000}
];"""
    child = b"""const links = {1: 'grandchild.html'};
const linkSyms = {1: 0};
const moves = [{'xy': [[1, 0]], 'p': 0.5, 'wl': 0.2, 'ssM': -0.3, 'wlRad': 0.1, 'sRad': 0.2, 'v': 18000}];"""
    content_by_url = {
        katago_book_module.OFFICIAL_9X9_TT_ROOT_URL: root,
        'https://katagobooks.org/book9x9tt/root/child.html': child,
    }

    monkeypatch.setattr(katago_book_module, '_fetch_page_content', content_by_url.__getitem__)
    export = katago_book_module.crawl_official_9x9_book(maximum_depth=2, maximum_pages=2)

    assert len(export.positions) == 1
    assert export.positions[0].action_ids == (0,)
    assert export.positions[0].preferred_action_id == 1


@dataclass(frozen=True)
class FakePosition:
    actions: tuple[int, ...]


class FakeState(GameStateContract[FakePosition]):
    @property
    def name(self) -> str:
        return 'go'

    @property
    def action_size(self) -> int:
        return 4

    @property
    def representation(self) -> RepresentationDimensions:
        layout = PackedPlaneLayout(8, 1, 0)
        return RepresentationDimensions(1, 8, 8, (0,), (), layout)

    def initial_position(self) -> FakePosition:
        return FakePosition(())

    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        return () if len(position.actions) == 60 else (0, 1, 2, 3)

    def child_position(self, position: FakePosition, action_id: int) -> FakePosition:
        if action_id not in self.legal_action_ids(position):
            raise ValueError('Illegal fake action.')
        return FakePosition((*position.actions, action_id))

    def current_player(self, position: FakePosition) -> Player:
        return Player.FIRST if len(position.actions) % 2 == 0 else Player.SECOND

    def natural_terminal_wdl(self, position: FakePosition) -> WdlTarget | None:
        return WdlTarget(win=0.0, draw=1.0, loss=0.0) if len(position.actions) == 60 else None

    def adjudicated_wdl(self, position: FakePosition, reason: TerminationReason) -> WdlTarget:
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def encode_network_input(self, position: FakePosition) -> PackedPlanePayload:
        digest = hashlib.sha256(bytes(position.actions)).digest()[:8]
        return self.packed_plane_layout.value(digest)

    @property
    def augmentation_count(self) -> int:
        return 1

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        return action_id

    def transform_encoded_state(
        self,
        encoded_state: PackedPlanePayload,
        augmentation_index: int,
    ) -> PackedPlanePayload:
        return encoded_state

    def transform_replay_targets(self, sample: ReplaySample, augmentation_index: int) -> ReplaySample:
        return sample


class FakeEngine:
    game_name = 'go'
    rules_digest = '1' * 64
    representation_digest = '2' * 64
    engine_identity = 'fake-engine-v1'
    engine_artifact_sha256 = ('3' * 64,)
    label_search_limit = 32

    def policy(self, position: FakePosition, action_ids: tuple[int, ...]) -> EnginePolicy:
        assert position.actions == action_ids
        return EnginePolicy(tuple(EnginePolicyEntry(action_id, 0.25) for action_id in range(4)), 0)

    def render_game(self, action_ids: tuple[int, ...]) -> str:
        return ' '.join(str(action_id) for action_id in action_ids)

    def close(self) -> None:
        pass


class TerminalWithLegalMovesState(FakeState):
    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        return (0, 1, 2, 3)

    def natural_terminal_wdl(self, position: FakePosition) -> WdlTarget | None:
        return WdlTarget(win=0.0, draw=1.0, loss=0.0) if position.actions else None


class TerminalGuardEngine(FakeEngine):
    def policy(self, position: FakePosition, action_ids: tuple[int, ...]) -> EnginePolicy:
        if position.actions:
            raise AssertionError('Dataset builder queried the engine after natural termination.')
        return super().policy(position, action_ids)


class FixedPolicyModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.tensor((0.4, 0.3, 0.2, 0.1), device=inputs.device).expand(inputs.shape[0], 4)
        value = torch.tensor((0.2, 0.6, 0.2), device=inputs.device).expand(inputs.shape[0], 3)
        return policy, value


def _base_four_actions(value: int, length: int = 5) -> tuple[int, ...]:
    actions = [0] * length
    for index in range(length - 1, -1, -1):
        actions[index] = value % 4
        value //= 4
    return tuple(actions)


def _book_export(position_count: int = 500) -> KataGoBookExport:
    return KataGoBookExport(
        source_root_url='https://katagobooks.org/book9x9tt/root/root.html',
        source_updated_on='2026-02-26',
        maximum_crawl_depth=5,
        maximum_crawl_pages=position_count,
        pages=(KataGoBookPageProvenance(url='https://katagobooks.org/page.html', sha256='a' * 64),),
        positions=tuple(
            KataGoBookPosition(
                node_id=f'node-{index}',
                root_variation_id=f'root-{index % 10}',
                action_ids=_base_four_actions(index),
                path_probability=0.1,
                black_win_probability=0.5,
                black_score=0.0,
                win_probability_uncertainty=0.01,
                score_uncertainty=0.1,
                visits=10000,
                preferred_action_id=1,
            )
            for index in range(position_count)
        ),
    )


def _book_selection(export_path: Path, export_sha256: str) -> KataGoBookSelectionConfiguration:
    return KataGoBookSelectionConfiguration(
        export_path=str(export_path),
        export_sha256=export_sha256,
        minimum_ply=5,
        maximum_ply=5,
        maximum_absolute_black_score=1.0,
        maximum_black_win_probability_deviation=0.1,
        maximum_win_probability_uncertainty=0.1,
        maximum_score_uncertainty=1.0,
        minimum_visits=1000,
    )


def test_opening_builder_expands_four_plies_and_reuses_manifest(tmp_path: Path) -> None:
    path = tmp_path / 'openings.json'
    configuration = OpeningSuiteConfiguration(
        path=str(path),
        opening_count=50,
        source=EngineBeamOpeningSource(
            kind='engine_beam',
            random_seed=7,
            expanded_actions_per_position=4,
            beam_width=128,
        ),
    )

    manifest = build_opening_suite(path, configuration, FakeState(), FakeEngine(), 'revision')
    loaded = build_opening_suite(path, configuration, FakeState(), FakeEngine(), 'revision')

    assert manifest == loaded
    assert len(manifest.openings) == 50
    assert all(len(opening.action_ids) == 4 for opening in manifest.openings)
    assert len({opening.final_position_digest for opening in manifest.openings}) == 50


def test_book_opening_and_dataset_builders_replay_paths_and_reuse_artifacts(tmp_path: Path) -> None:
    export_path = tmp_path / 'book.json'
    write_katago_book_export(export_path, _book_export())
    export_sha256 = canonical_json_sha256(export_path)
    selection = _book_selection(export_path, export_sha256)
    openings_path = tmp_path / 'book-openings.json'
    openings_configuration = OpeningSuiteConfiguration(
        path=str(openings_path),
        opening_count=50,
        source=KataGoBookOpeningSource(kind='katago_book', selection=selection),
    )
    dataset_path = tmp_path / 'book-dataset.bin'
    dataset_configuration = EvaluationDatasetConfiguration(
        path=str(dataset_path),
        source=KataGoBookDatasetSource(kind='katago_book', position_count=480, selection=selection),
    )

    openings = build_katago_book_opening_suite(
        openings_path, export_path, openings_configuration, FakeState(), FakeEngine(), 'revision'
    )
    dataset = build_katago_book_evaluation_dataset(
        dataset_path, export_path, dataset_configuration, FakeState(), FakeEngine(), 'revision'
    )
    data = load_evaluation_dataset(dataset_path, dataset)

    assert len(openings.openings) == 50
    assert all(len(opening.action_ids) == 5 for opening in openings.openings)
    assert dataset.position_count == 480
    assert all(int(row['top_action_id']) == 1 for row in data)
    assert (
        build_katago_book_opening_suite(
            openings_path, export_path, openings_configuration, FakeState(), FakeEngine(), 'revision'
        )
        == openings
    )
    assert (
        build_katago_book_evaluation_dataset(
            dataset_path, export_path, dataset_configuration, FakeState(), FakeEngine(), 'revision'
        )
        == dataset
    )

    changed_selection = selection.model_copy(update={'maximum_absolute_black_score': 9.0})
    changed_openings_configuration = openings_configuration.model_copy(
        update={'source': KataGoBookOpeningSource(kind='katago_book', selection=changed_selection)}
    )
    with pytest.raises(ValueError, match='immutable provenance'):
        build_katago_book_opening_suite(
            openings_path,
            export_path,
            changed_openings_configuration,
            FakeState(),
            FakeEngine(),
            'revision',
        )


def test_dataset_builder_retains_every_third_position_in_requested_range(tmp_path: Path) -> None:
    path = tmp_path / 'evaluation.bin'
    configuration = EvaluationDatasetConfiguration(
        path=str(path),
        source=EngineSelfPlayDatasetSource(kind='engine_self_play', random_seed=7, move_sampling_temperature=1.0),
    )

    manifest = build_evaluation_dataset(path, configuration, FakeState(), FakeEngine(), 'revision')
    data = load_evaluation_dataset(path, manifest)

    assert 480 <= manifest.position_count <= 520
    assert tuple(game.source_game_id for game in manifest.source_games) == tuple(range(len(manifest.source_games)))
    assert all(game.action_ids and game.human_readable for game in manifest.source_games)
    assert manifest.retained_ply_interval == 3
    assert all(int(row['ply']) % 3 == manifest.retained_ply_offset for row in data)
    assert all(int(row['policy_count']) == 4 for row in data)
    assert build_evaluation_dataset(path, configuration, FakeState(), FakeEngine(), 'revision') == manifest


def test_dataset_builder_stops_at_natural_terminal_with_remaining_legal_actions() -> None:
    generated = dataset_module._generate_source_game(
        source_game_id=0,
        retained_offset=0,
        configuration=EvaluationDatasetConfiguration(
            path='terminal.bin',
            source=EngineSelfPlayDatasetSource(kind='engine_self_play', random_seed=0, move_sampling_temperature=1.0),
        ),
        state=TerminalWithLegalMovesState(),
        engine=TerminalGuardEngine(),
        generator=random.Random(0),
    )

    assert generated.source_game.action_ids == (3,)
    assert len(generated.retained_positions) == 1


def test_fixed_dataset_evaluates_raw_policy_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / 'evaluation.bin'
    manifest = build_evaluation_dataset(
        dataset_path,
        EvaluationDatasetConfiguration(
            path=str(dataset_path),
            source=EngineSelfPlayDatasetSource(kind='engine_self_play', random_seed=7, move_sampling_temperature=1.0),
        ),
        FakeState(),
        FakeEngine(),
        'revision',
    )
    inference_path = tmp_path / 'inference.pt'
    traced = torch.jit.trace(FixedPolicyModel(), torch.zeros((1, 1, 8, 8)))
    traced.save(str(inference_path))
    checkpoint = CheckpointReference(
        generation=1,
        manifest_path=tmp_path / 'checkpoint.json',
        model_path=tmp_path / 'model.pt',
        optimizer_path=tmp_path / 'optimizer.pt',
        inference_model_path=inference_path,
        inference_model_sha256='0' * 64,
    )
    job = FixedDatasetEvaluationJob(
        kind='fixed_dataset',
        job_id='fixed-dataset',
        definition=FixedDatasetEvaluationDefinition(kind='fixed_dataset', definition_id='fixed-dataset'),
        boundary_seconds=1200,
        candidate=checkpoint,
        device_id=0,
        deadline_seconds=60,
        random_seed=7,
        result_path=tmp_path / 'result.json',
    )

    result = evaluate_fixed_dataset(job, FakeState(), dataset_path, 'cpu')

    assert result.position_count == manifest.position_count
    assert result.top_action_accuracy == 1.0
    assert result.policy_cross_entropy == pytest.approx(-sum(log(value) for value in (0.4, 0.3, 0.2, 0.1)) / 4)
