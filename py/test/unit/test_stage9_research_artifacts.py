from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import UUID

import pytest

from src.az.calibration.calibrate import (
    _calibrate_visit_margin,
    calibrate_from_committed_trace_artifacts,
    validate_adaptive_compatibility,
)
from src.az.calibration.models import (
    CalibrationSourceTrace,
    InitialTraceModelIdentity,
    MaximumMeanDisagreementRule,
    SearchTraceCollectionPayload,
    SearchTraceObservation,
    SearchTraceSampleLineage,
    SearchTraceSnapshot,
    VisitMarginCandidate,
    load_trace_collection_artifact,
    publish_trace_collection_artifact,
)
from src.az.config.seeds import (
    EvaluationActionSeedCoordinates,
    EvaluationGameSeedCoordinates,
    EvaluationSearchSeedCoordinates,
    SEED_DERIVATION_VERSION,
    SeedPurpose,
    SearchTraceSampleSeedCoordinates,
    derive_seed,
)
from src.az.config.artifacts import CalibrationArtifactReference
from src.az.config.search import VisitMarginAdaptiveRule
from src.az.config.serialization import load_resolved_configuration
from src.az.evaluation.models import (
    CandidateCheckpointIdentity,
    EvaluationCostCategory,
    EvaluationGameResult,
    EvaluationPairResult,
    EvaluationSeedLineage,
    GoColor,
    RandomOpponentIdentity,
)
from src.az.evaluation.checkpoints import EvaluationModelArtifactRepository
from src.az.evaluation.scheduling import ElapsedCheckpointScheduler
from src.az.evaluation.statistics import (
    LearningCurvePoint,
    learning_curve_statistics,
    score_to_elo,
    summarize_match,
)
from src.az.evaluation.storage import EvaluationResultRepository
from src.az.replay.envelope import GameTermination
from src.az.reporting.build import EvaluationCheckpointEvidence, RunReportEvidence, build_report
from src.az.reporting.matrix import (
    SearchComputeAblationMatrix,
    SearchComputeArmDefinition,
    expand_matrix,
)
from src.az.reporting.models import RunIdentity
from src.az.reporting.render import render_csv, render_machine_json, render_markdown
from src.az.training.checkpoints import (
    CheckpointArtifact,
    CheckpointArtifactFormat,
    CheckpointArtifactKind,
    CheckpointPurpose,
    LoadedModelCheckpoint,
    ModelCheckpointManifest,
)


HASH_A = 'a' * 64
HASH_B = 'b' * 64
EVALUATION_ID = UUID('10000000-0000-0000-0000-000000000001')
CANDIDATE = CandidateCheckpointIdentity(
    checkpoint_id=UUID('20000000-0000-0000-0000-000000000002'),
    model_artifact_sha256=HASH_A,
    model_version=3,
)


def _snapshot(simulations: int, first_visits: int, value: float) -> SearchTraceSnapshot:
    second_visits = simulations - first_visits
    return SearchTraceSnapshot(
        simulations=simulations,
        root_policy=(first_visits / simulations, second_visits / simulations),
        root_visits=(first_visits, second_visits),
        root_value=value,
    )


def _observation(identity: int, full_cap: int = 8) -> SearchTraceObservation:
    return SearchTraceObservation(
        source_position_id=UUID(int=identity),
        prefixes=(
            _snapshot(2, 2, 0.1),
            _snapshot(4, 4, 0.2),
            _snapshot(6, 5, 0.3),
        ),
        full=_snapshot(full_cap, full_cap - 2, 0.4),
    )


def test_calibration_is_deterministic_authenticated_and_cap_compatible() -> None:
    candidate = VisitMarginCandidate(
        minimum_simulations=2,
        check_interval_simulations=2,
        required_top_visit_fraction=0.75,
        required_top_two_margin=0.5,
    )
    observations = (_observation(1), _observation(2))
    sources = tuple(
        CalibrationSourceTrace(
            source_position_id=observation.source_position_id,
            trace_payload_sha256=HASH_A,
            trace_file_sha256=HASH_B,
        )
        for observation in observations
    )
    source_model = InitialTraceModelIdentity(
        kind='initial_model',
        model_initialization_seed=7,
        model_configuration_sha256=HASH_A,
    )
    rule = MaximumMeanDisagreementRule(
        kind='maximum_mean_disagreement',
        maximum_policy_total_variation=1,
        maximum_value_absolute_error=2,
    )
    first = _calibrate_visit_margin(
        artifact_id=UUID(int=9),
        source_run_id=UUID(int=10),
        source_configuration_sha256=HASH_A,
        source_model=source_model,
        game_configuration_sha256=HASH_B,
        observations=observations,
        sources=sources,
        candidates=(candidate,),
        acceptance_rule=rule,
    )
    second = _calibrate_visit_margin(
        artifact_id=UUID(int=9),
        source_run_id=UUID(int=10),
        source_configuration_sha256=HASH_A,
        source_model=source_model,
        game_configuration_sha256=HASH_B,
        observations=observations,
        sources=sources,
        candidates=(candidate,),
        acceptance_rule=rule,
    )

    assert first == second
    profile = first.payload.profiles[0]
    assert profile.observation_count == 2
    assert profile.selected.policy_total_variation_p95 >= profile.selected.policy_total_variation_median
    assert sum(item.observation_count for item in profile.selected_simulation_distribution) == 2
    validate_adaptive_compatibility(first, 8, 2, 2, 0.75, 0.5)
    with pytest.raises(ValueError, match='no unique profile'):
        validate_adaptive_compatibility(first, 16, 2, 2, 0.75, 0.5)


@pytest.mark.parametrize(
    'observations, message',
    (
        (
            (
                _observation(1),
                SearchTraceObservation(
                    source_position_id=UUID(int=2),
                    prefixes=(_snapshot(2, 2, 0.1), _snapshot(6, 5, 0.3)),
                    full=_snapshot(8, 6, 0.4),
                ),
            ),
            'checkpoint schedule',
        ),
        ((_observation(1), _observation(1)), 'identities must be unique'),
    ),
)
def test_calibration_rejects_inhomogeneous_or_duplicate_observations(
    observations: tuple[SearchTraceObservation, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _calibrate_visit_margin(
            artifact_id=UUID(int=9),
            source_run_id=UUID(int=10),
            source_configuration_sha256=HASH_A,
            source_model=InitialTraceModelIdentity(
                kind='initial_model',
                model_initialization_seed=7,
                model_configuration_sha256=HASH_A,
            ),
            game_configuration_sha256=HASH_B,
            observations=observations,
            sources=tuple(
                CalibrationSourceTrace(
                    source_position_id=observation.source_position_id,
                    trace_payload_sha256=HASH_A,
                    trace_file_sha256=HASH_B,
                )
                for observation in observations
            ),
            candidates=(
                VisitMarginCandidate(
                    minimum_simulations=2,
                    check_interval_simulations=2,
                    required_top_visit_fraction=0.75,
                    required_top_two_margin=0.5,
                ),
            ),
            acceptance_rule=MaximumMeanDisagreementRule(
                kind='maximum_mean_disagreement',
                maximum_policy_total_variation=1,
                maximum_value_absolute_error=2,
            ),
        )


def _go_snapshot(simulations: int, first_visits: int, value: float) -> SearchTraceSnapshot:
    visits = (first_visits, simulations - first_visits, *(0 for _ in range(8)))
    return SearchTraceSnapshot(
        simulations=simulations,
        root_policy=tuple(visits_value / simulations for visits_value in visits),
        root_visits=visits,
        root_value=value,
    )


def _published_trace(
    directory: Path,
    identity: int,
    *,
    source_model: InitialTraceModelIdentity,
    source_run_id: UUID = UUID(int=10),
    game_configuration_sha256: str = HASH_B,
) -> Path:
    sample_id = UUID(int=identity)
    root_seed = 91
    trace_seed = derive_seed(
        root_seed,
        SearchTraceSampleSeedCoordinates(
            purpose=SeedPurpose.SEARCH_TRACE_SAMPLE,
            process_index=0,
            worker_index=0,
            game_index=identity,
            ply=0,
        ),
    )
    payload = SearchTraceCollectionPayload(
        artifact_id=UUID(int=identity + 1000),
        source_run_id=source_run_id,
        source_configuration_sha256=HASH_A,
        source_model=source_model,
        game_id=UUID(int=identity + 2000),
        replay_sample_id=sample_id,
        lifecycle='completed_game_awaiting_replay_commit',
        game_configuration_sha256=game_configuration_sha256,
        native_state_hash=identity,
        encoding_planes=5,
        encoding_board_size=3,
        canonical_encoding=(0,) * 45,
        legal_actions=tuple(range(10)),
        observation=SearchTraceObservation(
            source_position_id=sample_id,
            prefixes=(
                _go_snapshot(2, 2, 0.1),
                _go_snapshot(4, 4, 0.2),
                _go_snapshot(6, 5, 0.3),
            ),
            full=_go_snapshot(8, 6, 0.4),
        ),
        seed_lineage=SearchTraceSampleLineage(
            derivation_version=SEED_DERIVATION_VERSION,
            root_seed=root_seed,
            process_index=0,
            worker_index=0,
            game_index=identity,
            ply=0,
            trace_sample_seed=trace_seed,
        ),
    )
    return publish_trace_collection_artifact(directory, payload)


def test_committed_trace_calibration_records_ordered_authenticated_sources(tmp_path: Path) -> None:
    source_model = InitialTraceModelIdentity(
        kind='initial_model',
        model_initialization_seed=7,
        model_configuration_sha256=HASH_A,
    )
    second = load_trace_collection_artifact(_published_trace(tmp_path.resolve(), 2, source_model=source_model))
    first = load_trace_collection_artifact(_published_trace(tmp_path.resolve(), 1, source_model=source_model))
    artifact = calibrate_from_committed_trace_artifacts(
        artifact_id=UUID(int=99),
        loaded_artifacts=(second, first),
        committed_replay_sample_ids=frozenset((UUID(int=1), UUID(int=2))),
        candidates=(
            VisitMarginCandidate(
                minimum_simulations=2,
                check_interval_simulations=2,
                required_top_visit_fraction=0.75,
                required_top_two_margin=0.5,
            ),
        ),
        acceptance_rule=MaximumMeanDisagreementRule(
            kind='maximum_mean_disagreement',
            maximum_policy_total_variation=1,
            maximum_value_absolute_error=2,
        ),
    )

    sources = artifact.payload.profiles[0].sources
    assert tuple(source.source_position_id for source in sources) == (UUID(int=1), UUID(int=2))
    assert tuple(source.trace_file_sha256 for source in sources) == (
        first.file_sha256,
        second.file_sha256,
    )
    assert artifact.payload.source_model == source_model


def test_committed_trace_calibration_rejects_orphans_and_mixed_sources(tmp_path: Path) -> None:
    source_model = InitialTraceModelIdentity(
        kind='initial_model',
        model_initialization_seed=7,
        model_configuration_sha256=HASH_A,
    )
    first = load_trace_collection_artifact(_published_trace(tmp_path.resolve(), 1, source_model=source_model))
    mixed = load_trace_collection_artifact(
        _published_trace(
            tmp_path.resolve(),
            2,
            source_model=source_model,
            source_run_id=UUID(int=77),
        )
    )
    candidate = VisitMarginCandidate(
        minimum_simulations=2,
        check_interval_simulations=2,
        required_top_visit_fraction=0.75,
        required_top_two_margin=0.5,
    )
    rule = MaximumMeanDisagreementRule(
        kind='maximum_mean_disagreement',
        maximum_policy_total_variation=1,
        maximum_value_absolute_error=2,
    )
    with pytest.raises(ValueError, match='orphaned'):
        calibrate_from_committed_trace_artifacts(
            artifact_id=UUID(int=99),
            loaded_artifacts=(first,),
            committed_replay_sample_ids=frozenset(),
            candidates=(candidate,),
            acceptance_rule=rule,
        )
    with pytest.raises(ValueError, match='must share run'):
        calibrate_from_committed_trace_artifacts(
            artifact_id=UUID(int=99),
            loaded_artifacts=(first, mixed),
            committed_replay_sample_ids=frozenset((UUID(int=1), UUID(int=2))),
            candidates=(candidate,),
            acceptance_rule=rule,
        )


@pytest.mark.parametrize(
    ('field_name', 'value', 'message'),
    (
        ('canonical_encoding', (2,) + (0,) * 44, 'must be binary'),
        ('legal_actions', (-1, 0), 'inside the full action space'),
        ('legal_actions', (0, 10), 'inside the full action space'),
    ),
)
def test_trace_position_evidence_rejects_invalid_encoding_or_actions(
    tmp_path: Path,
    field_name: str,
    value: tuple[int, ...],
    message: str,
) -> None:
    source_model = InitialTraceModelIdentity(
        kind='initial_model',
        model_initialization_seed=7,
        model_configuration_sha256=HASH_A,
    )
    loaded = load_trace_collection_artifact(_published_trace(tmp_path.resolve(), 1, source_model=source_model))
    payload = loaded.artifact.payload.model_dump()
    payload[field_name] = value

    with pytest.raises(ValueError, match=message):
        SearchTraceCollectionPayload.model_validate(payload)


def _lineage(
    pair_index: int,
    game_in_pair: int,
    plies: int,
    root_seed: int = 91,
    evaluation_index: int = 2,
) -> EvaluationSeedLineage:
    return EvaluationSeedLineage(
        derivation_version=SEED_DERIVATION_VERSION,
        root_seed=root_seed,
        evaluation_index=evaluation_index,
        pair_index=pair_index,
        game_in_pair=game_in_pair,
        game_seed=derive_seed(
            root_seed,
            EvaluationGameSeedCoordinates(
                purpose=SeedPurpose.EVALUATION_GAME,
                evaluation_index=evaluation_index,
                pair_index=pair_index,
                game_in_pair=game_in_pair,
            ),
        ),
        search_seeds=tuple(
            derive_seed(
                root_seed,
                EvaluationSearchSeedCoordinates(
                    purpose=SeedPurpose.EVALUATION_SEARCH,
                    evaluation_index=evaluation_index,
                    pair_index=pair_index,
                    game_in_pair=game_in_pair,
                    ply=ply,
                ),
            )
            for ply in range(plies)
        ),
        action_seeds=tuple(
            derive_seed(
                root_seed,
                EvaluationActionSeedCoordinates(
                    purpose=SeedPurpose.EVALUATION_ACTION,
                    evaluation_index=evaluation_index,
                    pair_index=pair_index,
                    game_in_pair=game_in_pair,
                    ply=ply,
                ),
            )
            for ply in range(plies)
        ),
    )


def _game(pair_index: int, game_in_pair: int, score: float) -> EvaluationGameResult:
    candidate_color = GoColor.BLACK if game_in_pair == 0 else GoColor.WHITE
    winner = (
        None
        if score == 0.5
        else candidate_color
        if score == 1
        else (GoColor.WHITE if candidate_color is GoColor.BLACK else GoColor.BLACK)
    )
    return EvaluationGameResult(
        evaluation_id=EVALUATION_ID,
        game_id=UUID(int=100 + pair_index * 2 + game_in_pair),
        pair_index=pair_index,
        game_in_pair=game_in_pair,
        requested_elapsed_seconds=3600,
        published_checkpoint_elapsed_seconds=3612.5,
        common_search_sha256=HASH_A,
        candidate=CANDIDATE,
        opponent=RandomOpponentIdentity(kind='random'),
        candidate_color=candidate_color,
        board_size=7,
        komi_half_points=15,
        scoring_rule='area',
        ko_rule='positional_superko',
        suicide_rule='illegal',
        seed_lineage=_lineage(pair_index, game_in_pair, 0),
        winner=winner,
        candidate_score=score,
        termination=GameTermination.TWO_CONSECUTIVE_PASSES,
        plies=0,
        candidate_configured_simulations=8,
        candidate_actual_simulations=8,
        opponent_configured_simulations=0,
        opponent_actual_simulations=0,
        evaluation_wall_seconds=1.5,
        cost_category=EvaluationCostCategory.EVALUATION,
    )


def _pair(pair_index: int, scores: tuple[float, float]) -> EvaluationPairResult:
    return EvaluationPairResult(
        evaluation_id=EVALUATION_ID,
        pair_index=pair_index,
        games=(_game(pair_index, 0, scores[0]), _game(pair_index, 1, scores[1])),
    )


def test_statistics_are_paired_deterministic_and_formula_exact() -> None:
    pairs = (_pair(0, (1, 0)), _pair(1, (1, 0.5)))
    first = summarize_match(pairs, bootstrap_samples=200, confidence_level=0.9, bootstrap_seed=17)
    second = summarize_match(pairs, bootstrap_samples=200, confidence_level=0.9, bootstrap_seed=17)

    assert first == second
    assert (first.wins, first.draws, first.losses) == (2, 1, 1)
    assert first.mean_score == pytest.approx(0.625)
    assert first.elo == pytest.approx(score_to_elo(0.625, 4))
    assert first.elo_confidence_interval.lower <= first.elo <= first.elo_confidence_interval.upper

    curve = learning_curve_statistics(
        (
            LearningCurvePoint(elapsed_hours=1, score=0.5, elo=0),
            LearningCurvePoint(elapsed_hours=3, score=0.75, elo=100),
        )
    )
    assert curve.score_auc_score_hours == pytest.approx(1.25)
    assert curve.elo_auc_elo_hours == pytest.approx(100)
    assert curve.final_score_per_hour == pytest.approx(0.25)


def test_statistics_reject_duplicate_or_mixed_pairs_and_over_budget_games() -> None:
    pair = _pair(0, (1, 0.5))
    with pytest.raises(ValueError, match='duplicate'):
        summarize_match((pair, pair), bootstrap_samples=20, confidence_level=0.9, bootstrap_seed=7)

    other_evaluation_id = UUID(int=444)
    mixed = EvaluationPairResult(
        evaluation_id=other_evaluation_id,
        pair_index=1,
        games=tuple(
            game.model_copy(
                update={
                    'evaluation_id': other_evaluation_id,
                    'pair_index': 1,
                    'game_id': UUID(int=500 + game.game_in_pair),
                    'seed_lineage': _lineage(1, game.game_in_pair, 0),
                }
            )
            for game in pair.games
        ),
    )
    with pytest.raises(ValueError, match='homogeneous'):
        summarize_match((pair, mixed), bootstrap_samples=20, confidence_level=0.9, bootstrap_seed=7)

    over_budget = _game(0, 0, 1).model_dump()
    over_budget['candidate_actual_simulations'] = 9
    with pytest.raises(ValueError, match='cannot exceed'):
        EvaluationGameResult.model_validate(over_budget)

    second = pair.games[1].model_copy(update={'seed_lineage': _lineage(0, 1, 0, root_seed=92)})
    with pytest.raises(ValueError, match='root seed'):
        EvaluationPairResult(
            evaluation_id=pair.evaluation_id,
            pair_index=pair.pair_index,
            games=(pair.games[0], second),
        )


def test_scheduler_boundaries_and_result_repository_resume(tmp_path: Path) -> None:
    scheduler = ElapsedCheckpointScheduler(1_000_000_000, (10, 20))
    assert scheduler.due(10_999_999_999, frozenset()) == ()
    due = scheduler.due(11_000_000_000, frozenset())
    assert due[0].requested_elapsed_seconds == 10
    assert due[0].detected_elapsed_seconds == 10

    repository = EvaluationResultRepository(tmp_path.resolve())
    game = _game(0, 0, 1)
    partial = repository.path(EVALUATION_ID, 0, 0).with_suffix('.partial')
    partial.write_bytes(b'torn')
    assert repository.publish(game) == game
    assert repository.publish(game) == game
    assert repository.load(EVALUATION_ID, 0, 0) == game
    collision = game.model_copy(update={'published_checkpoint_elapsed_seconds': 999.0})
    with pytest.raises(ValueError, match='different result'):
        repository.publish(collision)


def _loaded_model_checkpoint(identity: int, model_version: int, contents: bytes) -> LoadedModelCheckpoint:
    artifact = CheckpointArtifact(
        kind=CheckpointArtifactKind.MODEL,
        format=CheckpointArtifactFormat.TORCH_STATE_DICT_V1,
        filename='model.pt',
        byte_count=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )
    return LoadedModelCheckpoint(
        manifest=ModelCheckpointManifest(
            run_id=UUID(int=41),
            resolved_configuration_sha256=HASH_A,
            checkpoint_id=UUID(int=identity),
            created_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            purpose=CheckpointPurpose.SCHEDULED,
            model_version=model_version,
            model=artifact,
        ),
        model_artifact=contents,
    )


def test_evaluation_model_claim_retains_historical_checkpoint_after_advancement(tmp_path: Path) -> None:
    repository = EvaluationModelArtifactRepository(tmp_path.resolve())
    first = _loaded_model_checkpoint(51, 1, b'first-model')
    second = _loaded_model_checkpoint(52, 2, b'second-model')

    first_identity = repository.claim(first)
    repository.claim(second)

    assert repository.load(first_identity) == b'first-model'


def test_explicit_ablation_arms_change_only_declared_search_compute_factors() -> None:
    fixed = load_resolved_configuration(Path('configs/go/go-7x7-fixed.resolved.json'))
    progressive = load_resolved_configuration(Path('configs/go/go-7x7-progressive.resolved.json'))
    mixed = load_resolved_configuration(Path('configs/go/go-7x7-mixed.resolved.json'))
    definitions = tuple(
        SearchComputeArmDefinition(
            arm_id=configuration.experiment.arm_id,
            name=configuration.experiment.name,
            hypothesis=configuration.experiment.hypothesis,
            output_directory=configuration.experiment.output_directory,
            budget=configuration.search.budget,
            stopping=configuration.search.stopping,
        )
        for configuration in (fixed, progressive, mixed)
    ) + (
        SearchComputeArmDefinition(
            arm_id='adaptive',
            name='Adaptive search test arm',
            hypothesis='A real calibration reference is supplied after calibration.',
            output_directory=fixed.experiment.output_directory / 'adaptive',
            budget=fixed.search.budget,
            stopping=VisitMarginAdaptiveRule(
                kind='visit_margin',
                minimum_simulations=16,
                check_interval_simulations=16,
                required_top_visit_fraction=0.75,
                required_top_two_margin=0.5,
                calibration=CalibrationArtifactReference(
                    artifact_root='reference_artifacts',
                    artifact_id=UUID(int=1),
                    path=PurePosixPath('calibration/test-only.json'),
                    sha256='1' * 64,
                ),
            ),
        ),
    )
    matrix = SearchComputeAblationMatrix(
        kind='search_compute',
        matrix_id=UUID(int=801),
        common_configuration=fixed,
        arms=definitions,
        root_seeds=(11, 12),
    )

    expanded = expand_matrix(matrix)

    assert len(expanded) == 8
    assert len({arm.common_controls_sha256 for arm in expanded}) == 1
    assert {arm.configuration.experiment.arm_id for arm in expanded} == {
        definition.arm_id for definition in definitions
    }
    assert all(arm.configuration.experiment.root_seed == arm.seed for arm in expanded)
    assert all(arm.configuration.hardware == fixed.hardware for arm in expanded)
    assert all(arm.configuration.evaluation == fixed.evaluation for arm in expanded)

    with pytest.raises(ValueError, match='output directories'):
        SearchComputeAblationMatrix(
            kind='search_compute',
            matrix_id=UUID(int=802),
            common_configuration=fixed,
            arms=(
                definitions[0],
                definitions[1].model_copy(update={'output_directory': definitions[0].output_directory}),
            ),
            root_seeds=(11, 12),
        )
    with pytest.raises(ValueError, match='between zero'):
        SearchComputeAblationMatrix(
            kind='search_compute',
            matrix_id=UUID(int=803),
            common_configuration=fixed,
            arms=definitions[:2],
            root_seeds=(-1, 12),
        )


def test_report_artifact_and_human_summaries_are_deterministic_and_honest() -> None:
    pairs = (_pair(0, (1, 0.5)),)
    match = summarize_match(pairs, bootstrap_samples=20, confidence_level=0.9, bootstrap_seed=7)
    checkpoint = EvaluationCheckpointEvidence(
        elapsed_hours=2,
        pairs=pairs,
        bootstrap_samples=20,
        confidence_level=0.9,
        bootstrap_seed=7,
    )
    run = RunReportEvidence(
        identity=RunIdentity(
            run_id=UUID(int=901),
            arm_id=UUID(int=902),
            seed=11,
            resolved_configuration_sha256=HASH_A,
            source_revision='revision',
            hardware_identity='two-test-gpus',
        ),
        committed_replay_envelopes=(),
        evaluation_checkpoints=(checkpoint,),
        checkpoint_timing=(),
        optimizer_steps=None,
        replay_reuse=None,
        gpu_utilization_percent=None,
        source_artifact_sha256s=(HASH_B,),
    )
    first = build_report(
        report_id=UUID(int=903),
        title='Stage 9 deterministic fixture',
        matrix_id=UUID(int=904),
        common_controls_sha256=HASH_A,
        runs=(run,),
    )
    second = build_report(
        report_id=UUID(int=903),
        title='Stage 9 deterministic fixture',
        matrix_id=UUID(int=904),
        common_controls_sha256=HASH_A,
        runs=(run,),
    )

    assert first == second
    assert first.payload.runs[0].final_match == match
    assert render_machine_json(first.payload) == render_machine_json(second.payload)
    assert render_markdown(first.payload) == render_markdown(second.payload)
    assert render_csv(first.payload) == render_csv(second.payload)
    assert 'GPU utilization was not recorded' in render_markdown(first.payload)
    assert 'Evaluation simulations' in render_markdown(first.payload)
    duplicate_run_id = replace(
        run,
        identity=run.identity.model_copy(
            update={
                'arm_id': UUID(int=906),
                'seed': 12,
            }
        ),
    )
    with pytest.raises(ValueError, match='run IDs'):
        build_report(
            report_id=UUID(int=905),
            title='Duplicate run fixture',
            matrix_id=UUID(int=904),
            common_controls_sha256=HASH_A,
            runs=(run, duplicate_run_id),
        )
    with pytest.raises(ValueError, match='unique strictly increasing'):
        replace(
            run,
            evaluation_checkpoints=(
                checkpoint,
                replace(checkpoint, elapsed_hours=1),
            ),
        )
