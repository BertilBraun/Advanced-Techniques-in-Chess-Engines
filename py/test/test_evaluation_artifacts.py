from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from src.evaluation.artifacts import (
    build_evaluation_dataset,
    build_opening_suite,
    load_evaluation_dataset,
)
from src.evaluation.configuration import EvaluationDatasetConfiguration, OpeningSuiteConfiguration
from src.evaluation.engine import EnginePolicy, EnginePolicyEntry
from src.games.contracts import GameStateContract, Player, RepresentationDimensions, WdlTarget
from src.packed_planes import PackedPlaneLayout, PackedPlanePayload
from src.replay.contracts import ReplaySample
from src.self_play.completed_game import TerminationReason


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
        layout = PackedPlaneLayout(1, 1, 0)
        return RepresentationDimensions(1, 1, 1, (0,), (), layout)

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
        return EnginePolicy(tuple(EnginePolicyEntry(action_id, 0.25) for action_id in range(4)))

    def render_game(self, action_ids: tuple[int, ...]) -> str:
        return ' '.join(str(action_id) for action_id in action_ids)

    def close(self) -> None:
        pass


def test_opening_builder_expands_four_plies_and_reuses_manifest(tmp_path: Path) -> None:
    path = tmp_path / 'openings.json'
    configuration = OpeningSuiteConfiguration(
        path=str(path),
        random_seed=7,
        opening_count=50,
        expanded_actions_per_position=4,
        beam_width=128,
    )

    manifest = build_opening_suite(path, configuration, FakeState(), FakeEngine(), 'revision')
    loaded = build_opening_suite(path, configuration, FakeState(), FakeEngine(), 'revision')

    assert manifest == loaded
    assert len(manifest.openings) == 50
    assert all(len(opening.action_ids) == 4 for opening in manifest.openings)
    assert len({opening.final_position_digest for opening in manifest.openings}) == 50


def test_dataset_builder_retains_every_third_position_in_requested_range(tmp_path: Path) -> None:
    path = tmp_path / 'evaluation.bin'
    configuration = EvaluationDatasetConfiguration(
        path=str(path),
        random_seed=7,
        move_sampling_temperature=1.0,
    )

    manifest = build_evaluation_dataset(path, configuration, FakeState(), FakeEngine(), 'revision')
    data = load_evaluation_dataset(path, manifest)

    assert 480 <= manifest.position_count <= 520
    assert manifest.retained_ply_interval == 3
    assert all(int(row['ply']) % 3 == manifest.retained_ply_offset for row in data)
    assert all(int(row['policy_count']) == 4 for row in data)
    assert build_evaluation_dataset(path, configuration, FakeState(), FakeEngine(), 'revision') == manifest
