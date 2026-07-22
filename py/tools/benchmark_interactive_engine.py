from __future__ import annotations

import argparse
import gc
import hashlib
import os
import platform
import subprocess
import sys
from enum import Enum
from pathlib import Path
from time import time_ns

import chess
import torch
from pydantic import BaseModel, ConfigDict

from src.eval.InteractiveEngine import (
    AnalysisMode,
    AnalysisResult,
    InferenceMetrics,
    InferenceTarget,
    InteractiveEngine,
)


class RunStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


class WorkloadKind(str, Enum):
    TIMED = "timed"
    FIXED_SEARCHES = "fixed_searches"


class BenchmarkConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    inference_target: InferenceTarget
    device_id: int
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int
    workload: WorkloadKind
    budget: int


class QualityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference_move_agreement: bool
    reference_move_rank: int | None
    reference_move_visit_share: float
    root_value_delta: float
    principal_variation_prefix: int


class CandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    move_uci: str
    prior: float
    visits: int
    visit_share: float
    mean_value: float | None


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: RunStatus
    configuration: BenchmarkConfiguration
    skip_reason: str | None
    chosen_move_uci: str | None
    root_value: float | None
    candidates: tuple[CandidateRecord, ...]
    searches: int
    searches_per_second: float
    maximum_depth: int
    elapsed_milliseconds: int
    deadline_overshoot_milliseconds: int | None
    root_visits_before: int
    root_visits_after: int
    principal_variation: tuple[str, ...]
    quality: QualityMetrics | None
    inference_evaluations: int
    inference_cache_hits: int
    inference_cache_hit_rate_percent: float
    model_inference_calls: int
    model_inference_positions: int
    average_model_batch_size: float
    tree_selection_nanoseconds: int
    board_encoding_nanoseconds: int
    result_processing_nanoseconds: int
    tree_backup_nanoseconds: int
    tree_owner_wait_nanoseconds: int
    direct_inference_nanoseconds: int
    direct_worker_utilization: float
    peak_cuda_memory_mib: float


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: tuple[str, ...]
    git_revision: str
    git_dirty: bool | None
    model_path: str
    model_sha256: str
    python_version: str
    torch_version: str
    platform: str
    processor: str
    logical_cpu_count: int | None
    cuda_available: bool
    cuda_device_names: tuple[str, ...]


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_unix_nanoseconds: int
    starting_fen: str
    moves_uci: tuple[str, ...]
    reference_searches: int
    provenance: Provenance


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    completed_runs: int
    skipped_runs: int
    error_runs: int
    best_searches_per_second: float
    reference_move_uci: str
    reference_value: float


class ParsedArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: Path
    output_directory: Path
    starting_fen: str
    moves_uci: tuple[str, ...]
    devices: tuple[InferenceTarget, ...]
    device_id: int
    inference_workers: tuple[int, ...]
    outstanding_batches_per_worker: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    time_budgets_seconds: tuple[int, ...]
    search_limits: tuple[int, ...]
    reference_searches: int
    c_param: float
    cache_capacity: int
    source_revision: str | None


def _parse_integer_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(","))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return parsed


def _parse_devices(value: str) -> tuple[InferenceTarget, ...]:
    try:
        return tuple(InferenceTarget(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must contain auto, cpu, or cuda"
        ) from error


def parse_arguments() -> ParsedArguments:
    parser = argparse.ArgumentParser(
        description="Benchmark one long-lived interactive game position."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--starting-fen", default=chess.STARTING_FEN)
    parser.add_argument("--move", action="append", default=[])
    parser.add_argument(
        "--devices", type=_parse_devices, default=(InferenceTarget.CPU,)
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--inference-workers", type=_parse_integer_list, default=(1, 2, 3, 4)
    )
    parser.add_argument(
        "--outstanding-batches-per-worker",
        type=_parse_integer_list,
        default=(1,),
    )
    parser.add_argument("--batch-sizes", type=_parse_integer_list, default=(1, 8, 32))
    parser.add_argument(
        "--time-budgets-seconds", type=_parse_integer_list, default=(1, 3)
    )
    parser.add_argument(
        "--search-limits", type=_parse_integer_list, default=(256, 1024)
    )
    parser.add_argument("--reference-searches", type=int, default=4096)
    parser.add_argument("--c-param", type=float, default=1.0)
    parser.add_argument("--cache-capacity", type=int, default=250_000)
    parser.add_argument(
        "--source-revision",
        help="Source revision for exported source trees without Git metadata.",
    )
    namespace = parser.parse_args()
    return ParsedArguments(
        model=namespace.model,
        output_directory=namespace.output_directory,
        starting_fen=namespace.starting_fen,
        moves_uci=tuple(namespace.move),
        devices=namespace.devices,
        device_id=namespace.device_id,
        inference_workers=namespace.inference_workers,
        outstanding_batches_per_worker=namespace.outstanding_batches_per_worker,
        batch_sizes=namespace.batch_sizes,
        time_budgets_seconds=namespace.time_budgets_seconds,
        search_limits=namespace.search_limits,
        reference_searches=namespace.reference_searches,
        c_param=namespace.c_param,
        cache_capacity=namespace.cache_capacity,
        source_revision=namespace.source_revision,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(arguments: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_provenance(arguments: ParsedArguments) -> Provenance:
    cuda_names = tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )
    detected_revision = _git_output(("rev-parse", "HEAD"))
    status = _git_output(("status", "--porcelain"))
    return Provenance(
        command=tuple(sys.argv),
        git_revision=arguments.source_revision or detected_revision or "unavailable",
        git_dirty=None if status is None else bool(status),
        model_path=str(arguments.model.resolve()),
        model_sha256=_sha256(arguments.model),
        python_version=sys.version,
        torch_version=torch.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
        cuda_available=torch.cuda.is_available(),
        cuda_device_names=cuda_names,
    )


def _quality(result: AnalysisResult, reference: AnalysisResult) -> QualityMetrics:
    ranks = {
        candidate.move_uci: index + 1
        for index, candidate in enumerate(result.candidates)
    }
    shares = {
        candidate.move_uci: candidate.visit_share for candidate in result.candidates
    }
    common_prefix = 0
    for actual_move, reference_move in zip(
        result.principal_variation, reference.principal_variation
    ):
        if actual_move != reference_move:
            break
        common_prefix += 1
    return QualityMetrics(
        reference_move_agreement=result.chosen_move_uci == reference.chosen_move_uci,
        reference_move_rank=ranks.get(reference.chosen_move_uci),
        reference_move_visit_share=shares.get(reference.chosen_move_uci, 0.0),
        root_value_delta=abs(result.value - reference.value),
        principal_variation_prefix=common_prefix,
    )


def _completed_record(
    configuration: BenchmarkConfiguration,
    result: AnalysisResult,
    root_visits_before: int,
    root_visits_after: int,
    reference: AnalysisResult,
    inference_metrics: InferenceMetrics,
    peak_cuda_memory_mib: float,
) -> BenchmarkRecord:
    elapsed_seconds = result.elapsed_milliseconds / 1000.0
    overshoot = (
        result.elapsed_milliseconds - configuration.budget * 1000
        if configuration.workload is WorkloadKind.TIMED
        else None
    )
    return BenchmarkRecord(
        status=RunStatus.COMPLETED,
        configuration=configuration,
        skip_reason=None,
        chosen_move_uci=result.chosen_move_uci,
        root_value=result.value,
        candidates=tuple(
            CandidateRecord(
                move_uci=candidate.move_uci,
                prior=candidate.policy_prior,
                visits=candidate.visits,
                visit_share=candidate.visit_share,
                mean_value=candidate.mean_value,
            )
            for candidate in result.candidates
        ),
        searches=result.searches,
        searches_per_second=result.searches / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0,
        maximum_depth=result.maximum_depth,
        elapsed_milliseconds=result.elapsed_milliseconds,
        deadline_overshoot_milliseconds=overshoot,
        root_visits_before=root_visits_before,
        root_visits_after=root_visits_after,
        principal_variation=result.principal_variation,
        quality=_quality(result, reference),
        inference_evaluations=inference_metrics.evaluations,
        inference_cache_hits=inference_metrics.cache_hits,
        inference_cache_hit_rate_percent=inference_metrics.cache_hit_rate_percent,
        model_inference_calls=inference_metrics.model_calls,
        model_inference_positions=inference_metrics.model_positions,
        average_model_batch_size=inference_metrics.average_model_batch_size,
        tree_selection_nanoseconds=inference_metrics.tree_selection_nanoseconds,
        board_encoding_nanoseconds=inference_metrics.board_encoding_nanoseconds,
        result_processing_nanoseconds=inference_metrics.result_processing_nanoseconds,
        tree_backup_nanoseconds=inference_metrics.tree_backup_nanoseconds,
        tree_owner_wait_nanoseconds=inference_metrics.tree_owner_wait_nanoseconds,
        direct_inference_nanoseconds=inference_metrics.direct_inference_nanoseconds,
        direct_worker_utilization=inference_metrics.direct_worker_utilization,
        peak_cuda_memory_mib=peak_cuda_memory_mib,
    )


def _skipped_record(
    configuration: BenchmarkConfiguration, reason: str
) -> BenchmarkRecord:
    return BenchmarkRecord(
        status=RunStatus.SKIPPED,
        configuration=configuration,
        skip_reason=reason,
        chosen_move_uci=None,
        root_value=None,
        candidates=(),
        searches=0,
        searches_per_second=0.0,
        maximum_depth=0,
        elapsed_milliseconds=0,
        deadline_overshoot_milliseconds=None,
        root_visits_before=0,
        root_visits_after=0,
        principal_variation=(),
        quality=None,
        inference_evaluations=0,
        inference_cache_hits=0,
        inference_cache_hit_rate_percent=0.0,
        model_inference_calls=0,
        model_inference_positions=0,
        average_model_batch_size=0.0,
        tree_selection_nanoseconds=0,
        board_encoding_nanoseconds=0,
        result_processing_nanoseconds=0,
        tree_backup_nanoseconds=0,
        tree_owner_wait_nanoseconds=0,
        direct_inference_nanoseconds=0,
        direct_worker_utilization=0.0,
        peak_cuda_memory_mib=0.0,
    )


def _error_record(
    configuration: BenchmarkConfiguration, reason: str
) -> BenchmarkRecord:
    skipped = _skipped_record(configuration, reason)
    return skipped.model_copy(update={"status": RunStatus.ERROR})


def _configurations(arguments: ParsedArguments) -> tuple[BenchmarkConfiguration, ...]:
    configurations: list[BenchmarkConfiguration] = []
    for target in arguments.devices:
        for inference_workers in arguments.inference_workers:
            for outstanding_batches in arguments.outstanding_batches_per_worker:
                for batch_size in arguments.batch_sizes:
                    for budget in arguments.time_budgets_seconds:
                        configurations.append(
                            BenchmarkConfiguration(
                                inference_target=target,
                                device_id=arguments.device_id,
                                inference_workers=inference_workers,
                                inference_batch_size=batch_size,
                                outstanding_batches_per_worker=outstanding_batches,
                                workload=WorkloadKind.TIMED,
                                budget=budget,
                            )
                        )
                    for budget in arguments.search_limits:
                        configurations.append(
                            BenchmarkConfiguration(
                                inference_target=target,
                                device_id=arguments.device_id,
                                inference_workers=inference_workers,
                                inference_batch_size=batch_size,
                                outstanding_batches_per_worker=outstanding_batches,
                                workload=WorkloadKind.FIXED_SEARCHES,
                                budget=budget,
                            )
                        )
    return tuple(configurations)


def _engine(
    arguments: ParsedArguments, configuration: BenchmarkConfiguration
) -> InteractiveEngine:
    return InteractiveEngine(
        model_path=str(arguments.model),
        device_id=configuration.device_id,
        parallel_searches=configuration.inference_batch_size,
        c_param=arguments.c_param,
        maximum_batch_size=configuration.inference_batch_size,
        inference_workers=configuration.inference_workers,
        outstanding_batches_per_worker=configuration.outstanding_batches_per_worker,
        cache_capacity=arguments.cache_capacity,
        inference_target=configuration.inference_target,
    )


def run_configuration(
    arguments: ParsedArguments,
    configuration: BenchmarkConfiguration,
    reference: AnalysisResult,
) -> BenchmarkRecord:
    if (
        configuration.inference_target is InferenceTarget.CUDA
        and not torch.cuda.is_available()
    ):
        return _skipped_record(configuration, "CUDA requested but unavailable")
    if (
        configuration.inference_target is InferenceTarget.CUDA
        and configuration.device_id >= torch.cuda.device_count()
    ):
        return _skipped_record(configuration, "CUDA device ID is unavailable")

    if configuration.inference_target is InferenceTarget.CUDA:
        torch.cuda.reset_peak_memory_stats(configuration.device_id)
    try:
        engine = _engine(arguments, configuration)
        engine.new_game(arguments.starting_fen, arguments.moves_uci).analyze(
            AnalysisMode.POLICY
        )
        game = engine.new_game(arguments.starting_fen, arguments.moves_uci)
        root_visits_before = game.root_visits
        result = (
            game.analyze(AnalysisMode.MCTS, time_limit_seconds=configuration.budget)
            if configuration.workload is WorkloadKind.TIMED
            else game.analyze(AnalysisMode.MCTS, search_limit=configuration.budget)
        )
        peak_cuda_memory_mib = (
            torch.cuda.max_memory_reserved(configuration.device_id) / (1024 * 1024)
            if configuration.inference_target is InferenceTarget.CUDA
            else 0.0
        )
        return _completed_record(
            configuration,
            result,
            root_visits_before,
            game.root_visits,
            reference,
            engine.inference_metrics(),
            peak_cuda_memory_mib,
        )
    except (MemoryError, RuntimeError) as error:
        return _error_record(configuration, f"{type(error).__name__}: {error}")
    finally:
        gc.collect()
        if configuration.inference_target is InferenceTarget.CUDA:
            torch.cuda.empty_cache()


def main() -> None:
    arguments = parse_arguments()
    if arguments.reference_searches <= 0:
        raise ValueError("reference_searches must be positive")
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    manifest = BenchmarkManifest(
        created_unix_nanoseconds=time_ns(),
        starting_fen=arguments.starting_fen,
        moves_uci=arguments.moves_uci,
        reference_searches=arguments.reference_searches,
        provenance=collect_provenance(arguments),
    )
    (arguments.output_directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    reference_configuration = BenchmarkConfiguration(
        inference_target=arguments.devices[0],
        device_id=arguments.device_id,
        inference_workers=1,
        inference_batch_size=1,
        outstanding_batches_per_worker=1,
        workload=WorkloadKind.FIXED_SEARCHES,
        budget=arguments.reference_searches,
    )
    reference_engine = _engine(arguments, reference_configuration)
    reference = reference_engine.new_game(
        arguments.starting_fen, arguments.moves_uci
    ).analyze(AnalysisMode.MCTS, search_limit=arguments.reference_searches)
    del reference_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records: list[BenchmarkRecord] = []
    results_path = arguments.output_directory / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for configuration in _configurations(arguments):
            record = run_configuration(arguments, configuration, reference)
            records.append(record)
            results_file.write(record.model_dump_json() + "\n")
            results_file.flush()

    completed = [record for record in records if record.status is RunStatus.COMPLETED]
    skipped = [record for record in records if record.status is RunStatus.SKIPPED]
    errors = [record for record in records if record.status is RunStatus.ERROR]
    summary = BenchmarkSummary(
        completed_runs=len(completed),
        skipped_runs=len(skipped),
        error_runs=len(errors),
        best_searches_per_second=max(
            (record.searches_per_second for record in completed), default=0.0
        ),
        reference_move_uci=reference.chosen_move_uci,
        reference_value=reference.value,
    )
    (arguments.output_directory / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
