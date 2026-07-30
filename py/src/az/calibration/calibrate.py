from __future__ import annotations

from uuid import UUID

from src.az.calibration.models import (
    CalibratedCandidate,
    CalibrationSourceTrace,
    LoadedSearchTraceArtifact,
    MaximumMeanDisagreementRule,
    SearchCalibrationArtifact,
    SearchCalibrationPayload,
    SearchCalibrationProfile,
    SearchTraceCollectionArtifact,
    SearchTraceObservation,
    SearchTraceSnapshot,
    TraceModelIdentity,
    VisitMarginCandidate,
    SimulationCountFrequency,
    calibration_payload_sha256,
    load_trace_collection_artifact,
)
from src.az.config.seeds import SearchTraceSampleSeedCoordinates, SeedPurpose, derive_seed


def _total_variation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return 0.5 * sum(abs(left_value - right_value) for left_value, right_value in zip(left, right, strict=True))


def _candidate_snapshot(
    observation: SearchTraceObservation,
    candidate: VisitMarginCandidate,
) -> SearchTraceSnapshot:
    for prefix in observation.prefixes:
        if prefix.simulations < candidate.minimum_simulations:
            continue
        if (prefix.simulations - candidate.minimum_simulations) % candidate.check_interval_simulations != 0:
            continue
        ordered_visits = sorted(prefix.root_visits, reverse=True)
        top = ordered_visits[0]
        second = ordered_visits[1]
        if (
            top / prefix.simulations >= candidate.required_top_visit_fraction
            and (top - second) / prefix.simulations >= candidate.required_top_two_margin
        ):
            return prefix
    return observation.full


def _measure_candidate(
    observations: tuple[SearchTraceObservation, ...],
    candidate: VisitMarginCandidate,
) -> CalibratedCandidate:
    selected = tuple(_candidate_snapshot(observation, candidate) for observation in observations)
    count = len(observations)
    policy_disagreements = tuple(
        _total_variation(snapshot.root_policy, observation.full.root_policy)
        for snapshot, observation in zip(selected, observations, strict=True)
    )
    value_disagreements = tuple(
        abs(snapshot.root_value - observation.full.root_value)
        for snapshot, observation in zip(selected, observations, strict=True)
    )
    return CalibratedCandidate(
        candidate=candidate,
        mean_policy_total_variation=sum(policy_disagreements) / count,
        mean_value_absolute_error=sum(value_disagreements) / count,
        early_stop_fraction=sum(
            snapshot.simulations < observation.full.simulations
            for snapshot, observation in zip(selected, observations, strict=True)
        )
        / count,
        mean_simulations=sum(snapshot.simulations for snapshot in selected) / count,
        policy_total_variation_median=_median(policy_disagreements),
        policy_total_variation_p95=_quantile(policy_disagreements, 0.95),
        policy_total_variation_maximum=max(policy_disagreements),
        value_absolute_error_median=_median(value_disagreements),
        value_absolute_error_p95=_quantile(value_disagreements, 0.95),
        value_absolute_error_maximum=max(value_disagreements),
        stop_simulations_median=_median(tuple(float(snapshot.simulations) for snapshot in selected)),
        stop_simulations_p95=_quantile(tuple(float(snapshot.simulations) for snapshot in selected), 0.95),
        stop_simulations_maximum=max(snapshot.simulations for snapshot in selected),
    )


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _calibrate_profile(
    observations: tuple[SearchTraceObservation, ...],
    sources: tuple[CalibrationSourceTrace, ...],
    candidates: tuple[VisitMarginCandidate, ...],
    acceptance_rule: MaximumMeanDisagreementRule,
) -> SearchCalibrationProfile:
    checkpoint_schedules = {
        tuple(prefix.simulations for prefix in observation.prefixes) for observation in observations
    }
    action_counts = {
        len(snapshot.root_policy)
        for observation in observations
        for snapshot in (*observation.prefixes, observation.full)
    }
    if len(checkpoint_schedules) != 1 or len(action_counts) != 1:
        raise ValueError('A calibration profile requires one complete checkpoint schedule and action space.')
    measured = tuple(_measure_candidate(observations, candidate) for candidate in candidates)
    acceptable = tuple(
        result
        for result in measured
        if result.mean_policy_total_variation <= acceptance_rule.maximum_policy_total_variation
        and result.mean_value_absolute_error <= acceptance_rule.maximum_value_absolute_error
    )
    if not acceptable:
        raise ValueError('No adaptive candidate satisfies the calibration acceptance rule.')
    selected = min(
        acceptable,
        key=lambda result: (
            result.mean_simulations,
            -result.early_stop_fraction,
            result.mean_policy_total_variation,
            result.mean_value_absolute_error,
            result.candidate.minimum_simulations,
            result.candidate.check_interval_simulations,
            result.candidate.required_top_visit_fraction,
            result.candidate.required_top_two_margin,
        ),
    )
    chosen_snapshots = tuple(_candidate_snapshot(observation, selected.candidate) for observation in observations)
    counts = tuple(
        SimulationCountFrequency(
            simulations=simulations,
            observation_count=sum(snapshot.simulations == simulations for snapshot in chosen_snapshots),
        )
        for simulations in sorted({snapshot.simulations for snapshot in chosen_snapshots})
    )
    return SearchCalibrationProfile(
        full_simulation_cap=observations[0].full.simulations,
        trace_checkpoints=tuple(prefix.simulations for prefix in observations[0].prefixes),
        selected=selected,
        candidate_results=measured,
        selected_simulation_distribution=counts,
        observation_count=len(observations),
        sources=sources,
    )


def _calibrate_visit_margin(
    *,
    artifact_id: UUID,
    source_run_id: UUID,
    source_configuration_sha256: str,
    source_model: TraceModelIdentity,
    game_configuration_sha256: str,
    observations: tuple[SearchTraceObservation, ...],
    sources: tuple[CalibrationSourceTrace, ...],
    candidates: tuple[VisitMarginCandidate, ...],
    acceptance_rule: MaximumMeanDisagreementRule,
) -> SearchCalibrationArtifact:
    if not observations:
        raise ValueError('Calibration requires trace observations.')
    position_ids = tuple(observation.source_position_id for observation in observations)
    if len(set(position_ids)) != len(position_ids):
        raise ValueError('Calibration source position identities must be unique.')
    if tuple(source.source_position_id for source in sources) != position_ids:
        raise ValueError('Calibration source trace evidence must identify observations in exact order.')
    full_caps = tuple(sorted({observation.full.simulations for observation in observations}))
    profiles = tuple(
        _calibrate_profile(
            tuple(observation for observation in observations if observation.full.simulations == cap),
            tuple(
                source
                for observation, source in zip(observations, sources, strict=True)
                if observation.full.simulations == cap
            ),
            candidates,
            acceptance_rule,
        )
        for cap in full_caps
    )
    payload = SearchCalibrationPayload(
        artifact_id=artifact_id,
        source_run_id=source_run_id,
        source_configuration_sha256=source_configuration_sha256,
        source_model=source_model,
        game_configuration_sha256=game_configuration_sha256,
        acceptance_rule=acceptance_rule,
        candidates=candidates,
        profiles=profiles,
    )
    return SearchCalibrationArtifact(payload=payload, payload_sha256=calibration_payload_sha256(payload))


def calibrate_from_committed_trace_artifacts(
    *,
    artifact_id: UUID,
    loaded_artifacts: tuple[LoadedSearchTraceArtifact, ...],
    committed_replay_sample_ids: frozenset[UUID],
    candidates: tuple[VisitMarginCandidate, ...],
    acceptance_rule: MaximumMeanDisagreementRule,
) -> SearchCalibrationArtifact:
    if not loaded_artifacts:
        raise ValueError('Calibration requires authenticated trace artifacts.')
    authenticated = tuple(load_trace_collection_artifact(loaded.path) for loaded in loaded_artifacts)
    if authenticated != loaded_artifacts:
        raise ValueError('Loaded trace artifact evidence no longer matches its source file.')
    ordered = tuple(
        sorted(
            authenticated,
            key=lambda loaded: loaded.artifact.payload.observation.source_position_id.hex,
        )
    )
    artifacts = tuple(loaded.artifact for loaded in ordered)
    observations = committed_trace_observations(artifacts, committed_replay_sample_ids)
    source_identities = {
        (
            artifact.payload.source_run_id,
            artifact.payload.source_configuration_sha256,
            artifact.payload.source_model,
            artifact.payload.game_configuration_sha256,
        )
        for artifact in artifacts
    }
    if len(source_identities) != 1:
        raise ValueError('Calibration trace artifacts must share run, configuration, model, and game configuration.')
    source_run_id, source_configuration_sha256, source_model, game_configuration_sha256 = next(iter(source_identities))
    sources = tuple(
        CalibrationSourceTrace(
            source_position_id=artifact.payload.observation.source_position_id,
            trace_payload_sha256=artifact.payload_sha256,
            trace_file_sha256=loaded.file_sha256,
        )
        for loaded, artifact in zip(ordered, artifacts, strict=True)
    )
    return _calibrate_visit_margin(
        artifact_id=artifact_id,
        source_run_id=source_run_id,
        source_configuration_sha256=source_configuration_sha256,
        source_model=source_model,
        game_configuration_sha256=game_configuration_sha256,
        observations=observations,
        sources=sources,
        candidates=candidates,
        acceptance_rule=acceptance_rule,
    )


def validate_adaptive_compatibility(
    artifact: SearchCalibrationArtifact,
    simulation_cap: int,
    minimum_simulations: int,
    check_interval_simulations: int,
    required_top_visit_fraction: float,
    required_top_two_margin: float,
) -> None:
    matching = tuple(profile for profile in artifact.payload.profiles if profile.full_simulation_cap == simulation_cap)
    if len(matching) != 1:
        raise ValueError('Calibration artifact has no unique profile for the configured simulation cap.')
    candidate = matching[0].selected.candidate
    expected = (
        simulation_cap,
        minimum_simulations,
        check_interval_simulations,
        required_top_visit_fraction,
        required_top_two_margin,
    )
    actual = (
        matching[0].full_simulation_cap,
        candidate.minimum_simulations,
        candidate.check_interval_simulations,
        candidate.required_top_visit_fraction,
        candidate.required_top_two_margin,
    )
    if actual != expected:
        raise ValueError('Adaptive search configuration is incompatible with its calibration artifact.')


def validate_trace_collection_lineage(artifact: SearchTraceCollectionArtifact) -> None:
    lineage = artifact.payload.seed_lineage
    expected = derive_seed(
        lineage.root_seed,
        SearchTraceSampleSeedCoordinates(
            purpose=SeedPurpose.SEARCH_TRACE_SAMPLE,
            process_index=lineage.process_index,
            worker_index=lineage.worker_index,
            game_index=lineage.game_index,
            ply=lineage.ply,
        ),
    )
    if lineage.trace_sample_seed != expected:
        raise ValueError('Search trace sample seed does not match its deterministic coordinates.')


def committed_trace_observations(
    artifacts: tuple[SearchTraceCollectionArtifact, ...],
    committed_replay_sample_ids: frozenset[UUID],
) -> tuple[SearchTraceObservation, ...]:
    if not artifacts:
        raise ValueError('Trace calibration assembly requires artifacts.')
    observations: list[SearchTraceObservation] = []
    for artifact in artifacts:
        validate_trace_collection_lineage(artifact)
        if artifact.payload.replay_sample_id not in committed_replay_sample_ids:
            raise ValueError('Uncommitted or orphaned search trace cannot enter calibration.')
        observations.append(artifact.payload.observation)
    identities = tuple(observation.source_position_id for observation in observations)
    if len(set(identities)) != len(identities):
        raise ValueError('Committed trace source position identities must be unique.')
    return tuple(observations)
