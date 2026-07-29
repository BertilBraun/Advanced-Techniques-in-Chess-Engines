from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.az.config.cli import main
from src.az.config.dependency_lock import parse_pinned_dependency_lock
from src.az.config.manifest import (
    BuildDeclaration,
    DependencyDeclaration,
    DependencyRecord,
    HardwareDeclaration,
    RunManifest,
    SourceState,
    build_manifest,
    file_sha256,
    inspect_source_state,
)
from src.az.config.models import (
    MixedSearchBudget,
    ProgressiveSearchBudget,
    ResolvedRunConfiguration,
    SearchConfiguration,
    VisitMarginAdaptiveRule,
)
from src.az.config.resolution import AuthoringRunConfiguration, resolve_configuration
from src.az.config.seeds import (
    GameSeedCoordinates,
    SearchSeedCoordinates,
    SeedPurpose,
    derive_seed,
)
from src.az.config.serialization import (
    canonical_json,
    load_authoring_configuration,
    load_resolved_configuration,
    model_sha256,
    resolve_file,
)
from src.az.games.api import GameIdentifier, create_game_registry


CONFIGURATION_DIRECTORY = Path('configs/v2')
AUTHORING_PATHS = tuple(sorted(CONFIGURATION_DIRECTORY.glob('*.authoring.json')))


def fixed_authoring() -> AuthoringRunConfiguration:
    return load_authoring_configuration(CONFIGURATION_DIRECTORY / 'go-7x7-fixed.authoring.json')


@pytest.mark.parametrize('path', AUTHORING_PATHS)
def test_representative_authoring_and_resolved_configurations_match(path: Path) -> None:
    resolved = resolve_file(path)
    checked_in_resolved = load_resolved_configuration(
        path.with_name(path.name.replace('.authoring.json', '.resolved.json'))
    )

    assert resolved == checked_in_resolved
    assert resolved.game.board_size == 7
    assert resolved.game.action_count == 50
    assert resolved.game.input_plane_count == 17


def test_resolution_materializes_every_root_category() -> None:
    authoring = fixed_authoring()
    assert set(authoring.experiment.model_fields_set) < set(type(authoring.experiment).model_fields)

    serialized = resolve_configuration(authoring).model_dump(mode='json', exclude_unset=False)

    assert set(serialized) == set(ResolvedRunConfiguration.model_fields)
    assert serialized['game']['scoring_rule'] == 'area'
    assert serialized['game']['ko_rule'] == 'positional_superko'
    assert serialized['game']['suicide_rule'] == 'illegal'
    assert serialized['game']['pass_exempt_from_superko'] is True
    assert serialized['game']['score_comparison'] == 'doubled_integer_points'
    assert serialized['search']['backup_discount'] == 1.0
    assert serialized['training']['objective']['kind'] == 'go_policy_value'
    assert serialized['telemetry']['required_metrics']


def test_resolved_configuration_is_frozen() -> None:
    configuration = resolve_configuration(fixed_authoring())

    with pytest.raises(ValidationError, match='frozen'):
        configuration.game.komi_half_points = 13


def test_forbids_unknown_fields_at_nested_boundaries() -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['search']['budget']['legacy_playout_cap'] = 0.25

    with pytest.raises(ValidationError, match='legacy_playout_cap'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


@pytest.mark.parametrize(
    ('path_name', 'budget_type', 'stopping_type'),
    (
        ('go-7x7-progressive.authoring.json', ProgressiveSearchBudget, type(None)),
        ('go-7x7-mixed.authoring.json', MixedSearchBudget, type(None)),
        ('go-7x7-adaptive.authoring.json', type(None), VisitMarginAdaptiveRule),
    ),
)
def test_behavioral_discriminators_select_exact_variants(
    path_name: str,
    budget_type: type[ProgressiveSearchBudget] | type[MixedSearchBudget] | type[None],
    stopping_type: type[VisitMarginAdaptiveRule] | type[None],
) -> None:
    configuration = resolve_file(CONFIGURATION_DIRECTORY / path_name)

    if budget_type is not type(None):
        assert type(configuration.search.budget) is budget_type
    if stopping_type is not type(None):
        assert type(configuration.search.stopping) is stopping_type


def test_unknown_budget_discriminator_fails_clearly() -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['search']['budget']['kind'] = 'gumbel'

    with pytest.raises(ValidationError, match='union_tag_invalid'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


@pytest.mark.parametrize(
    'fpu',
    (
        {'kind': 'parent_value'},
        {'kind': 'reduced_parent_value', 'reduction': 0.2},
        {'kind': 'visited_child_mean', 'no_visited_child_value': 0.0},
    ),
)
def test_all_fpu_discriminators_validate(fpu: dict[str, str | float]) -> None:
    candidate = resolve_configuration(fixed_authoring()).search.model_dump(mode='json')
    candidate['fpu'] = fpu

    search = SearchConfiguration.model_validate_json(json.dumps(candidate))

    assert search.fpu.kind == fpu['kind']


def test_reduced_fpu_requires_reduction() -> None:
    candidate = resolve_configuration(fixed_authoring()).search.model_dump(mode='json')
    candidate['fpu'] = {'kind': 'reduced_parent_value'}

    with pytest.raises(ValidationError, match='reduction'):
        SearchConfiguration.model_validate_json(json.dumps(candidate))


@pytest.mark.parametrize(
    ('mutation', 'message'),
    (
        (('board_size', 8), 'board_size'),
        (('scoring_rule', 'territory'), 'scoring_rule'),
        (('ko_rule', 'simple'), 'ko_rule'),
        (('suicide_rule', 'allowed'), 'suicide_rule'),
        (('pass_exempt_from_superko', False), 'pass_exempt_from_superko'),
        (('pass_action', 'zero'), 'pass_action'),
        (('capped_game_value_target_weight', 1), 'capped_game_value_target_weight'),
    ),
)
def test_go_rule_invariants_are_fixed(
    mutation: tuple[str, int | str],
    message: str,
) -> None:
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    field_name, value = mutation
    candidate['game'][field_name] = value

    with pytest.raises(ValidationError, match=message):
        ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))


def test_komi_is_serialized_as_integer_half_points() -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['game']['komi_half_points'] = 7.5

    with pytest.raises(ValidationError, match='komi_half_points'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


@pytest.mark.parametrize(
    ('field_name', 'value'),
    (
        ('komi_half_points', -(2**31) - 1),
        ('komi_half_points', 2**31),
        ('safety_ply_cap', 2**31),
        ('history_length', 1_025),
    ),
)
def test_go_native_integer_boundaries_are_enforced(
    field_name: str,
    value: int,
) -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['game'][field_name] = value

    with pytest.raises(ValidationError, match=field_name):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


@pytest.mark.parametrize('komi_half_points', (-(2**31), 2**31 - 1))
def test_go_native_komi_boundaries_are_accepted(komi_half_points: int) -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['game']['komi_half_points'] = komi_half_points

    configuration = AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))

    assert configuration.game.komi_half_points == komi_half_points


def test_go_native_positive_maxima_are_accepted() -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['game']['safety_ply_cap'] = 2**31 - 1
    candidate['game']['history_length'] = 1_024

    configuration = AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))

    assert configuration.game.safety_ply_cap == 2**31 - 1
    assert configuration.game.history_length == 1_024


def test_go_evaluation_komi_uses_native_integer_boundaries() -> None:
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    candidate['evaluation']['suite']['komi_half_points'] = 2**31

    with pytest.raises(ValidationError, match='komi_half_points'):
        ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))


def test_progressive_stages_require_zero_and_strict_order() -> None:
    candidate = load_authoring_configuration(CONFIGURATION_DIRECTORY / 'go-7x7-progressive.authoring.json').model_dump(
        mode='json'
    )
    candidate['search']['budget']['stages'][1]['start_elapsed_seconds'] = 0

    with pytest.raises(ValidationError, match='increase strictly'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


def test_mixed_search_requires_distinct_caps() -> None:
    candidate = load_authoring_configuration(CONFIGURATION_DIRECTORY / 'go-7x7-mixed.authoring.json').model_dump(
        mode='json'
    )
    candidate['search']['budget']['cheap_simulations'] = 128

    with pytest.raises(ValidationError, match='cheap cap'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


def test_adaptive_minimum_must_be_below_finite_cap() -> None:
    candidate = load_authoring_configuration(CONFIGURATION_DIRECTORY / 'go-7x7-adaptive.authoring.json').model_dump(
        mode='json'
    )
    candidate['search']['stopping']['minimum_simulations'] = 128

    with pytest.raises(ValidationError, match='Adaptive stopping minimum'):
        resolve_configuration(AuthoringRunConfiguration.model_validate_json(json.dumps(candidate)))


def test_adaptive_minimum_must_be_below_every_progressive_cap() -> None:
    candidate = load_authoring_configuration(CONFIGURATION_DIRECTORY / 'go-7x7-progressive.authoring.json').model_dump(
        mode='json'
    )
    candidate['search']['stopping'] = {
        'kind': 'visit_margin',
        'minimum_simulations': 16,
        'check_interval_simulations': 8,
        'required_top_visit_fraction': 0.8,
        'required_top_two_margin': 0.5,
        'calibration_id': 'test-calibration',
    }

    with pytest.raises(ValidationError, match='every applicable budget cap'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


def test_mixed_and_adaptive_composition_is_explicitly_rejected() -> None:
    candidate = load_authoring_configuration(CONFIGURATION_DIRECTORY / 'go-7x7-mixed.authoring.json').model_dump(
        mode='json'
    )
    candidate['search']['stopping'] = {
        'kind': 'visit_margin',
        'minimum_simulations': 8,
        'check_interval_simulations': 4,
        'required_top_visit_fraction': 0.8,
        'required_top_two_margin': 0.5,
        'calibration_id': 'test-calibration',
    }

    with pytest.raises(ValidationError, match='cannot be combined with mixed'):
        AuthoringRunConfiguration.model_validate_json(json.dumps(candidate))


def test_global_batch_matches_trainer_ranks() -> None:
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    candidate['training']['local_batch_size'] = 1_024

    with pytest.raises(ValidationError, match='Global batch size'):
        ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))


def test_canonical_hash_is_stable_across_round_trip_and_key_order() -> None:
    configuration = resolve_configuration(fixed_authoring())
    serialized = configuration.model_dump_json()
    reordered = json.dumps(json.loads(serialized), sort_keys=True, indent=4)
    round_tripped = ResolvedRunConfiguration.model_validate_json(reordered)

    assert model_sha256(configuration) == model_sha256(round_tripped)
    assert canonical_json(configuration) == canonical_json(round_tripped)


def test_configuration_paths_have_platform_independent_serialization() -> None:
    serialized = canonical_json(resolve_configuration(fixed_authoring()))

    assert '"output_directory":"runs/go-7x7-fixed"' in serialized
    assert '"shard_directory":"runs/go-7x7-fixed/replay"' in serialized
    assert '\\' not in serialized


def test_progressive_model_schedule_uses_discriminator_and_ordering() -> None:
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    small_architecture = candidate['model']['schedule']['architecture'].copy()
    small_architecture['channels'] = 64
    small_architecture['residual_blocks'] = 5
    large_architecture = candidate['model']['schedule']['architecture'].copy()
    candidate['model']['schedule'] = {
        'kind': 'progressive',
        'stages': [
            {'start_elapsed_seconds': 0, 'architecture': small_architecture},
            {'start_elapsed_seconds': 10_800, 'architecture': large_architecture},
        ],
    }

    progressive = ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))
    assert progressive.model.schedule.kind == 'progressive'

    candidate['model']['schedule']['stages'][1]['start_elapsed_seconds'] = 0
    with pytest.raises(ValidationError, match='increase strictly'):
        ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))


def test_topology_devices_must_fit_declared_hardware() -> None:
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    candidate['topology']['evaluation']['device_ids'] = [2]

    with pytest.raises(ValidationError, match='expected GPU count'):
        ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))


def test_every_authoring_category_override_flows_to_resolved_configuration() -> None:
    candidate = fixed_authoring().model_dump(mode='json')
    candidate['experiment']['duration_seconds'] = 22_000
    candidate['hardware']['expected_gpu_model'] = 'override-gpu'
    candidate['topology']['native_threads_per_worker'] = 7
    candidate['game']['komi_half_points'] = 13
    candidate['model']['schedule']['architecture']['channels'] = 96
    candidate['search']['algorithm']['exploration_constant'] = 1.25
    candidate['search']['fpu'] = {'kind': 'parent_value'}
    candidate['search']['root_exploration'] = {'kind': 'disabled'}
    candidate['search']['temperature'] = {'kind': 'constant', 'temperature': 0.5}
    candidate['search']['tree_reuse'] = {'kind': 'disabled'}
    candidate['search']['inference']['maximum_batch_size'] = 32
    candidate['search']['backup_discount'] = 0.9
    candidate['self_play']['concurrent_games_per_worker'] = 7
    candidate['replay']['capacity_positions'] = 123_456
    candidate['replay']['shard_directory'] = 'custom/replay'
    candidate['replay']['credits']['target_reuse'] = 3
    candidate['training']['optimizer']['learning_rate'] = 0.001
    candidate['training']['learning_rate_schedule']['multiplier'] = 0.5
    candidate['training']['objective']['value_loss_weight'] = 1.5
    candidate['training']['maximum_optimizer_steps'] = 123_000
    candidate['evaluation']['search']['budget']['simulations'] = 96
    candidate['evaluation']['paired_games_per_checkpoint'] = 20
    candidate['evaluation']['bootstrap_samples'] = 2_000
    candidate['evaluation']['komi_half_points'] = 13
    candidate['telemetry']['write_every_seconds'] = 3
    candidate['retention']['recent_checkpoint_count'] = 2

    resolved = resolve_configuration(AuthoringRunConfiguration.model_validate_json(json.dumps(candidate)))

    assert resolved.experiment.duration_seconds == 22_000
    assert resolved.hardware.expected_gpu_model == 'override-gpu'
    assert resolved.topology.native_threads_per_worker == 7
    assert resolved.game.komi_half_points == 13
    assert resolved.model.schedule.architecture.channels == 96
    assert resolved.search.algorithm.exploration_constant == pytest.approx(1.25)
    assert resolved.search.fpu.kind == 'parent_value'
    assert resolved.search.root_exploration.kind == 'disabled'
    assert resolved.search.temperature.kind == 'constant'
    assert resolved.search.tree_reuse.kind == 'disabled'
    assert resolved.search.inference.maximum_batch_size == 32
    assert resolved.search.backup_discount == pytest.approx(0.9)
    assert resolved.self_play.concurrent_games_per_worker == 7
    assert resolved.replay.capacity_positions == 123_456
    assert str(resolved.replay.shard_directory) == 'custom/replay'
    assert resolved.replay.credits.target_reuse == pytest.approx(3)
    assert resolved.training.optimizer.learning_rate == pytest.approx(0.001)
    assert resolved.training.learning_rate_schedule.multiplier == pytest.approx(0.5)
    assert resolved.training.objective.value_loss_weight == pytest.approx(1.5)
    assert resolved.training.maximum_optimizer_steps == 123_000
    assert resolved.evaluation.search.budget.simulations == 96
    assert resolved.evaluation.paired_games_per_checkpoint == 20
    assert resolved.evaluation.bootstrap_samples == 2_000
    assert resolved.telemetry.write_every_seconds == 3
    assert resolved.retention.recent_checkpoint_count == 2


def test_progressive_model_stages_must_start_within_run_duration() -> None:
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    architecture = candidate['model']['schedule']['architecture']
    candidate['model']['schedule'] = {
        'kind': 'progressive',
        'stages': [
            {'start_elapsed_seconds': 0, 'architecture': architecture},
            {'start_elapsed_seconds': 21_600, 'architecture': architecture},
        ],
    }

    with pytest.raises(ValidationError, match='model stage must start before'):
        ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))


def test_seed_derivation_is_stable_and_separates_purposes() -> None:
    game_coordinates = GameSeedCoordinates(
        purpose=SeedPurpose.GAME,
        process_index=0,
        worker_index=2,
        game_index=11,
    )
    search_coordinates = SearchSeedCoordinates(
        purpose=SeedPurpose.SEARCH,
        process_index=0,
        worker_index=2,
        game_index=11,
        ply=0,
    )

    assert derive_seed(1234, game_coordinates) == 5836480873312707840
    assert derive_seed(1234, game_coordinates) == derive_seed(1234, game_coordinates)
    assert derive_seed(1234, game_coordinates) != derive_seed(1234, search_coordinates)
    assert derive_seed(1235, game_coordinates) != derive_seed(1234, game_coordinates)


def test_seed_coordinates_require_exact_purpose_specific_fields() -> None:
    with pytest.raises(ValidationError, match='game_index'):
        GameSeedCoordinates(
            purpose=SeedPurpose.GAME,
            process_index=0,
            worker_index=0,
        )
    with pytest.raises(ValidationError, match='ply'):
        GameSeedCoordinates.model_validate(
            {
                'purpose': SeedPurpose.GAME,
                'process_index': 0,
                'worker_index': 0,
                'game_index': 0,
                'ply': 0,
            }
        )


def test_registry_imports_go_module_only_when_resolved() -> None:
    sys.modules.pop('src.az.games.go.module', None)
    registry = create_game_registry()
    assert 'src.az.games.go.module' not in sys.modules

    registration = registry.resolve(GameIdentifier.GO)

    assert registration.identifier is GameIdentifier.GO
    assert registration.payload_schema_name == 'go-training-payload'
    assert 'src.az.games.go.module' in sys.modules


def _initialize_git_repository(path: Path) -> None:
    subprocess.run(('git', 'init', '-q'), cwd=path, check=True)
    subprocess.run(('git', 'config', 'user.email', 'test@example.com'), cwd=path, check=True)
    subprocess.run(('git', 'config', 'user.name', 'Test'), cwd=path, check=True)
    (path / 'tracked.txt').write_text('clean\n', encoding='utf-8')
    (path / 'requirements.lock').write_text('pydantic==2.13.2\n', encoding='utf-8')
    subprocess.run(('git', 'add', 'tracked.txt', 'requirements.lock'), cwd=path, check=True)
    subprocess.run(('git', 'commit', '-qm', 'fixture'), cwd=path, check=True)


def make_test_build_declaration() -> BuildDeclaration:
    return BuildDeclaration(
        build_id='test-build',
        build_type='debug',
        compiler='test-compiler',
        python_version='3.10',
        platform='test-platform',
    )


def make_test_dependency_declaration(directory: Path) -> DependencyDeclaration:
    lock_file = directory / 'requirements.lock'
    if not lock_file.is_file():
        lock_file.write_text('pydantic==2.13.2\n', encoding='utf-8')
    return DependencyDeclaration(
        lock_file=lock_file,
        lock_file_sha256=file_sha256(lock_file),
        packages=(DependencyRecord(name='pydantic', version='2.13.2'),),
    )


def matching_hardware_declaration() -> HardwareDeclaration:
    return HardwareDeclaration(
        gpu_model='NVIDIA GeForce RTX 4090',
        gpu_count=2,
        logical_cpu_count=16,
        ram_gib=64,
        free_disk_gib=100,
    )


def test_manifest_records_clean_and_dirty_source_provenance(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    clean = inspect_source_state(tmp_path)
    assert clean.clean
    assert clean.dirty_patch_sha256 is None

    (tmp_path / 'tracked.txt').write_text('dirty\n', encoding='utf-8')
    dirty = inspect_source_state(tmp_path)

    assert not dirty.clean
    assert dirty.revision == clean.revision
    assert dirty.dirty_patch_sha256 is not None
    assert len(dirty.dirty_patch_sha256) == 64


def test_manifest_rejects_dirty_source_when_policy_requires_clean(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    (tmp_path / 'tracked.txt').write_text('dirty\n', encoding='utf-8')

    with pytest.raises(ValueError, match='requires a clean source'):
        build_manifest(
            configuration=resolve_configuration(fixed_authoring()),
            repository_root=tmp_path,
            build=make_test_build_declaration(),
            dependencies=make_test_dependency_declaration(tmp_path),
            hardware=matching_hardware_declaration(),
        )


def test_manifest_records_dirty_patch_when_policy_allows_it(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    (tmp_path / 'tracked.txt').write_text('dirty\n', encoding='utf-8')
    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    candidate['experiment']['manifest_policy']['require_clean_source'] = False
    configuration = ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))

    manifest = build_manifest(
        configuration=configuration,
        repository_root=tmp_path,
        build=make_test_build_declaration(),
        dependencies=make_test_dependency_declaration(tmp_path),
        hardware=matching_hardware_declaration(),
    )

    assert not manifest.source.clean
    assert manifest.source.dirty_patch_sha256 is not None


@pytest.mark.parametrize(
    ('hardware', 'message'),
    (
        (
            HardwareDeclaration(
                gpu_model='wrong-gpu',
                gpu_count=2,
                logical_cpu_count=16,
                ram_gib=64,
                free_disk_gib=100,
            ),
            'Expected GPU model',
        ),
        (
            HardwareDeclaration(
                gpu_model='NVIDIA GeForce RTX 4090',
                gpu_count=1,
                logical_cpu_count=16,
                ram_gib=64,
                free_disk_gib=100,
            ),
            'Expected 2 GPUs',
        ),
        (
            HardwareDeclaration(
                gpu_model='NVIDIA GeForce RTX 4090',
                gpu_count=2,
                logical_cpu_count=15,
                ram_gib=64,
                free_disk_gib=100,
            ),
            'CPU count',
        ),
        (
            HardwareDeclaration(
                gpu_model='NVIDIA GeForce RTX 4090',
                gpu_count=2,
                logical_cpu_count=16,
                ram_gib=63,
                free_disk_gib=100,
            ),
            'RAM',
        ),
        (
            HardwareDeclaration(
                gpu_model='NVIDIA GeForce RTX 4090',
                gpu_count=2,
                logical_cpu_count=16,
                ram_gib=64,
                free_disk_gib=99,
            ),
            'free disk',
        ),
    ),
)
def test_manifest_rejects_hardware_mismatch(
    tmp_path: Path,
    hardware: HardwareDeclaration,
    message: str,
) -> None:
    _initialize_git_repository(tmp_path)

    with pytest.raises(ValueError, match=message):
        build_manifest(
            configuration=resolve_configuration(fixed_authoring()),
            repository_root=tmp_path,
            build=make_test_build_declaration(),
            dependencies=make_test_dependency_declaration(tmp_path),
            hardware=hardware,
        )


def test_manifest_hash_and_revision_types_reject_malformed_values() -> None:
    with pytest.raises(ValidationError, match='revision'):
        SourceState(revision='not-a-revision', clean=True, dirty_patch_sha256=None)
    with pytest.raises(ValidationError, match='lock_file_sha256'):
        DependencyDeclaration(
            lock_file=Path('requirements.lock'),
            lock_file_sha256='not-a-sha',
            packages=(),
        )


def test_source_state_rejects_inconsistent_clean_flag() -> None:
    with pytest.raises(ValidationError, match='clean source state'):
        SourceState(revision='a' * 40, clean=True, dirty_patch_sha256='b' * 64)
    with pytest.raises(ValidationError, match='dirty source state'):
        SourceState(revision='a' * 40, clean=False, dirty_patch_sha256=None)


def test_manifest_rejects_tampered_integrity_fields(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    manifest = build_manifest(
        configuration=resolve_configuration(fixed_authoring()),
        repository_root=tmp_path,
        build=make_test_build_declaration(),
        dependencies=make_test_dependency_declaration(tmp_path),
        hardware=matching_hardware_declaration(),
        created_at_utc=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    baseline = manifest.model_dump(mode='json')

    tampered_hash = baseline.copy()
    tampered_hash['configuration_sha256'] = '0' * 64
    with pytest.raises(ValidationError, match='does not match its configuration'):
        RunManifest.model_validate_json(json.dumps(tampered_hash))

    tampered_mode = baseline.copy()
    tampered_mode['determinism_mode'] = 'strict_single_thread'
    with pytest.raises(ValidationError, match='determinism mode'):
        RunManifest.model_validate_json(json.dumps(tampered_mode))

    tampered_version = baseline.copy()
    tampered_version['seed_derivation_version'] = 'az-seed-v1'
    with pytest.raises(ValidationError, match='seed_derivation_version'):
        RunManifest.model_validate_json(json.dumps(tampered_version))

    naive_time = baseline.copy()
    naive_time['created_at_utc'] = '2026-07-30T00:00:00'
    with pytest.raises(ValidationError, match='must use UTC'):
        RunManifest.model_validate_json(json.dumps(naive_time))

    non_utc_time = baseline.copy()
    non_utc_time['created_at_utc'] = '2026-07-30T02:00:00+02:00'
    with pytest.raises(ValidationError, match='must use UTC'):
        RunManifest.model_validate_json(json.dumps(non_utc_time))

    dirty_source = baseline.copy()
    dirty_source['source'] = {
        'revision': manifest.source.revision,
        'clean': False,
        'dirty_patch_sha256': 'a' * 64,
    }
    with pytest.raises(ValidationError, match='requires a clean source'):
        RunManifest.model_validate_json(json.dumps(dirty_source))

    tampered_dependencies = baseline.copy()
    tampered_dependencies['dependencies'] = baseline['dependencies'].copy()
    tampered_dependencies['dependencies']['packages'] = []
    with pytest.raises(ValidationError, match='required by the manifest policy'):
        RunManifest.model_validate_json(json.dumps(tampered_dependencies))

    with pytest.raises(ValidationError, match='does not match its configuration'):
        RunManifest(
            manifest_version=manifest.manifest_version,
            created_at_utc=manifest.created_at_utc,
            configuration=manifest.configuration,
            configuration_sha256='0' * 64,
            source=manifest.source,
            build=manifest.build,
            dependencies=manifest.dependencies,
            hardware=manifest.hardware,
            determinism_mode=manifest.determinism_mode,
            seed_derivation_version=manifest.seed_derivation_version,
        )
    with pytest.raises(ValidationError, match='must use UTC'):
        RunManifest(
            manifest_version=manifest.manifest_version,
            created_at_utc=datetime(
                2026,
                7,
                30,
                tzinfo=timezone(timedelta(hours=2)),
            ),
            configuration=manifest.configuration,
            configuration_sha256=manifest.configuration_sha256,
            source=manifest.source,
            build=manifest.build,
            dependencies=manifest.dependencies,
            hardware=manifest.hardware,
            determinism_mode=manifest.determinism_mode,
            seed_derivation_version=manifest.seed_derivation_version,
        )


@pytest.mark.parametrize(
    ('field_name', 'value', 'message'),
    (
        ('gpu_model', 'wrong-gpu', 'Expected GPU model'),
        ('gpu_count', 1, 'Expected 2 GPUs'),
        ('logical_cpu_count', 15, 'CPU count'),
        ('ram_gib', 63, 'RAM'),
        ('free_disk_gib', 99, 'free disk'),
    ),
)
def test_manifest_json_rejects_tampered_hardware(
    tmp_path: Path,
    field_name: str,
    value: str | int,
    message: str,
) -> None:
    _initialize_git_repository(tmp_path)
    manifest = build_manifest(
        configuration=resolve_configuration(fixed_authoring()),
        repository_root=tmp_path,
        build=make_test_build_declaration(),
        dependencies=make_test_dependency_declaration(tmp_path),
        hardware=matching_hardware_declaration(),
    )
    tampered = manifest.model_dump(mode='json')
    tampered['hardware'][field_name] = value

    with pytest.raises(ValidationError, match=message):
        RunManifest.model_validate_json(json.dumps(tampered))


def test_manifest_requires_existing_matching_dependency_lock(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    missing = DependencyDeclaration(
        lock_file=tmp_path / 'missing.lock',
        lock_file_sha256='a' * 64,
        packages=(DependencyRecord(name='pydantic', version='2.13.2'),),
    )
    with pytest.raises(ValueError, match='does not exist'):
        build_manifest(
            configuration=resolve_configuration(fixed_authoring()),
            repository_root=tmp_path,
            build=make_test_build_declaration(),
            dependencies=missing,
            hardware=matching_hardware_declaration(),
        )

    mismatched = make_test_dependency_declaration(tmp_path).model_copy(update={'lock_file_sha256': 'a' * 64})
    with pytest.raises(ValueError, match='SHA-256 does not match'):
        build_manifest(
            configuration=resolve_configuration(fixed_authoring()),
            repository_root=tmp_path,
            build=make_test_build_declaration(),
            dependencies=mismatched,
            hardware=matching_hardware_declaration(),
        )


def test_dependency_records_follow_manifest_policy(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    lock_file = tmp_path / 'requirements.lock'
    with pytest.raises(ValidationError, match='must be unique'):
        DependencyDeclaration(
            lock_file=lock_file,
            lock_file_sha256=file_sha256(lock_file),
            packages=(
                DependencyRecord(name='pydantic', version='2.13.2'),
                DependencyRecord(name='Pydantic', version='2.13.1'),
            ),
        )

    no_packages = DependencyDeclaration(
        lock_file=lock_file,
        lock_file_sha256=file_sha256(lock_file),
        packages=(),
    )
    with pytest.raises(ValueError, match='required by the manifest policy'):
        build_manifest(
            configuration=resolve_configuration(fixed_authoring()),
            repository_root=tmp_path,
            build=make_test_build_declaration(),
            dependencies=no_packages,
            hardware=matching_hardware_declaration(),
        )

    candidate = resolve_configuration(fixed_authoring()).model_dump(mode='json')
    candidate['experiment']['manifest_policy']['record_dependency_versions'] = False
    no_versions_configuration = ResolvedRunConfiguration.model_validate_json(json.dumps(candidate))
    with pytest.raises(ValueError, match='must be empty'):
        build_manifest(
            configuration=no_versions_configuration,
            repository_root=tmp_path,
            build=make_test_build_declaration(),
            dependencies=make_test_dependency_declaration(tmp_path),
            hardware=matching_hardware_declaration(),
        )
    manifest = build_manifest(
        configuration=no_versions_configuration,
        repository_root=tmp_path,
        build=make_test_build_declaration(),
        dependencies=no_packages,
        hardware=matching_hardware_declaration(),
    )
    assert manifest.dependencies.packages == ()


def test_dependency_lock_parser_records_complete_sorted_inventory(tmp_path: Path) -> None:
    lock_file = tmp_path / 'requirements.lock'
    lock_file.write_text(
        '\n'.join(
            (
                '# generated lock',
                'Torch==2.7.1 \\',
                '    --hash=sha256:' + 'a' * 64,
                'numpy==2.2.6',
                'pydantic==2.13.2',
            )
        ),
        encoding='utf-8',
    )

    records = parse_pinned_dependency_lock(lock_file)

    assert tuple((record.name, record.version) for record in records) == (
        ('numpy', '2.2.6'),
        ('pydantic', '2.13.2'),
        ('Torch', '2.7.1'),
    )


@pytest.mark.parametrize(
    ('contents', 'message'),
    (
        ('numpy>=2.2.6\n', 'exact name==version pin'),
        ('numpy\n', 'exact name==version pin'),
        ('numpy @ https://example.invalid/numpy.whl\n', 'exact name==version pin'),
        ('-r other.lock\n', 'Recursive requirement'),
        ('--index-url https://example.invalid/simple\n', 'Unsupported dependency lock option'),
        ('numpy==2.2.6\nNumPy==2.2.5\n', 'duplicate normalized package'),
    ),
)
def test_dependency_lock_parser_rejects_unsupported_or_unpinned_entries(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    lock_file = tmp_path / 'requirements.lock'
    lock_file.write_text(contents, encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        parse_pinned_dependency_lock(lock_file)


def test_manifest_records_configuration_environment_and_seed_contract(tmp_path: Path) -> None:
    _initialize_git_repository(tmp_path)
    configuration = resolve_configuration(fixed_authoring())
    manifest = build_manifest(
        configuration=configuration,
        repository_root=tmp_path,
        build=make_test_build_declaration(),
        dependencies=make_test_dependency_declaration(tmp_path),
        hardware=matching_hardware_declaration(),
        created_at_utc=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )

    assert manifest.configuration_sha256 == model_sha256(configuration)
    assert manifest.source.clean
    assert manifest.determinism_mode.value == 'seeded_concurrent'
    assert manifest.seed_derivation_version == 'az-seed-v2'
    assert type(manifest.model_validate_json(manifest.model_dump_json())) is type(manifest)


def test_cli_validates_resolves_and_prints_without_starting_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authoring_path = CONFIGURATION_DIRECTORY / 'go-7x7-fixed.authoring.json'
    resolved_path = tmp_path / 'resolved.json'

    assert main(('validate', str(authoring_path))) == 0
    validation_output = capsys.readouterr().out.strip()
    assert len(validation_output) == 64

    assert main(('resolve', str(authoring_path), '--output', str(resolved_path))) == 0
    capsys.readouterr()
    assert resolved_path.is_file()

    assert main(('print-config', str(resolved_path), '--resolved-input')) == 0
    printed = capsys.readouterr().out
    assert ResolvedRunConfiguration.model_validate_json(printed) == load_resolved_configuration(resolved_path)


def test_cli_prints_manifest_without_starting_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_path = tmp_path / 'repository'
    repository_path.mkdir()
    _initialize_git_repository(repository_path)
    resolved_path = tmp_path / 'resolved.json'
    resolved_path.write_text(
        resolve_configuration(fixed_authoring()).model_dump_json(indent=2),
        encoding='utf-8',
    )
    lock_path = tmp_path / 'requirements.lock'
    lock_path.write_text('pydantic==2.13.2\nnumpy==2.2.6\n', encoding='utf-8')

    exit_code = main(
        (
            'manifest',
            str(resolved_path),
            '--repository-root',
            str(repository_path),
            '--dependency-lock',
            str(lock_path),
            '--gpu-model',
            'NVIDIA GeForce RTX 4090',
            '--gpu-count',
            '2',
            '--logical-cpu-count',
            '16',
            '--ram-gib',
            '64',
            '--free-disk-gib',
            '100',
        )
    )
    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert printed['configuration_sha256'] == model_sha256(resolve_configuration(fixed_authoring()))
    assert printed['source']['clean'] is True
    assert printed['seed_derivation_version'] == 'az-seed-v2'
    assert printed['dependencies']['packages'] == [
        {'name': 'numpy', 'version': '2.2.6'},
        {'name': 'pydantic', 'version': '2.13.2'},
    ]
