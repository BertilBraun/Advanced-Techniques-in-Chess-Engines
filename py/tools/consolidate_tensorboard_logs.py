from __future__ import annotations

import argparse
import hashlib
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

from pydantic import BaseModel, ConfigDict, Field
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from src.evaluation.tensorboard import (
    EVALUATION_TAG_PATTERN,
    MATCH_OUTCOME_METRICS,
    discovered_evaluation_tensorboard_categories,
)
from src.util.tensorboard import TensorboardCustomScalarCategory, TensorboardMultilineChart


EVENT_FILE_PATTERN = 'events.out.tfevents.*'
RUN_NAME_PATTERN = re.compile(r'run_(\d+)')


@dataclass(frozen=True)
class SummaryIdentity:
    tag: str
    plugin_name: str
    value_type: str
    step: int


@dataclass(frozen=True)
class SummaryState:
    wall_time: float
    value_sha256: str
    serialized_value: bytes
    time_series_tag: str
    source_layout_run_name: str
    source_layout_tag: str


@dataclass(frozen=True)
class EventFileFingerprint:
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True)
class EventFileSelection:
    event_files: tuple[Path, ...]
    source_event_file_count: int
    skipped_self_play_event_file_count: int
    representative_self_play_processes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class TensorboardSourceSegment:
    source_root: Path
    minimum_wall_time: float | None = None
    maximum_wall_time: float | None = None
    evaluation_step_offset: int = 0


@dataclass(frozen=True)
class TimeSeriesRoute:
    run_name: str
    tag: str


class RepresentativeSelfPlayProcess(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_name: str
    process_id: int = Field(ge=1)


class TagSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    tag: str
    plugin_name: str
    value_type: str
    point_count: int = Field(ge=1)
    minimum_step: int
    maximum_step: int


class SourceSegmentSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_root: str
    minimum_wall_time: float | None
    maximum_wall_time: float | None
    evaluation_step_offset: int


class ConsolidationManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 3
    generated_at_utc: datetime
    source_segments: tuple[SourceSegmentSummary, ...]
    output_root: str
    source_event_file_count: int = Field(ge=0)
    selected_event_file_count: int = Field(ge=0)
    skipped_self_play_event_file_count: int = Field(ge=0)
    excluded_text_summary_count: int = Field(ge=0)
    emitted_summary_count: int = Field(ge=0)
    unique_summary_count: int = Field(ge=0)
    replaced_summary_count: int = Field(ge=0)
    representative_self_play_processes: tuple[RepresentativeSelfPlayProcess, ...]
    tags: tuple[TagSummary, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path)
    parser.add_argument(
        '--source-segment',
        action='append',
        nargs=4,
        metavar=('SOURCE_ROOT', 'MINIMUM_WALL_TIME', 'MAXIMUM_WALL_TIME', 'EVALUATION_STEP_OFFSET'),
    )
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--watch-interval-seconds', type=float)
    parser.add_argument('--custom-layout-only', action='store_true')
    parser.add_argument('--time-series-view', action='store_true')
    parser.add_argument('--evaluation-outcome-overlays', action='store_true')
    parser.add_argument('--preserve-source-layout', action='store_true')
    parser.add_argument('--single-run-source', action='store_true')
    return parser.parse_args()


def _optional_wall_time(raw_value: str) -> float | None:
    return None if raw_value == 'none' else float(raw_value)


def _source_segments(arguments: argparse.Namespace) -> tuple[TensorboardSourceSegment, ...]:
    if arguments.source_root is not None and arguments.source_segment is not None:
        raise ValueError('--source-root and --source-segment are mutually exclusive.')
    if arguments.source_segment is not None:
        return tuple(
            TensorboardSourceSegment(
                source_root=Path(raw_source_root),
                minimum_wall_time=_optional_wall_time(raw_minimum_wall_time),
                maximum_wall_time=_optional_wall_time(raw_maximum_wall_time),
                evaluation_step_offset=int(raw_evaluation_step_offset),
            )
            for (
                raw_source_root,
                raw_minimum_wall_time,
                raw_maximum_wall_time,
                raw_evaluation_step_offset,
            ) in arguments.source_segment
        )
    if arguments.source_root is None:
        raise ValueError('--source-root or --source-segment is required unless --custom-layout-only is selected.')
    return (TensorboardSourceSegment(source_root=arguments.source_root),)


def _validate_paths(
    source_segments: tuple[TensorboardSourceSegment, ...],
    output_root: Path,
) -> tuple[tuple[TensorboardSourceSegment, ...], Path]:
    if not source_segments:
        raise ValueError('At least one TensorBoard source segment is required.')
    resolved_output_root = output_root.resolve()
    resolved_source_segments: list[TensorboardSourceSegment] = []
    for source_segment in source_segments:
        resolved_source_root = source_segment.source_root.resolve()
        if not resolved_source_root.is_dir():
            raise ValueError(f'TensorBoard source root does not exist: {resolved_source_root}')
        if resolved_output_root == resolved_source_root or resolved_source_root in resolved_output_root.parents:
            raise ValueError('Output root must not be a source root or one of its descendants.')
        if (
            source_segment.minimum_wall_time is not None
            and source_segment.maximum_wall_time is not None
            and source_segment.minimum_wall_time > source_segment.maximum_wall_time
        ):
            raise ValueError('Source segment minimum wall time must not exceed its maximum wall time.')
        resolved_source_segments.append(
            TensorboardSourceSegment(
                source_root=resolved_source_root,
                minimum_wall_time=source_segment.minimum_wall_time,
                maximum_wall_time=source_segment.maximum_wall_time,
                evaluation_step_offset=source_segment.evaluation_step_offset,
            )
        )
    if resolved_output_root.exists() and any(resolved_output_root.iterdir()):
        raise ValueError(f'TensorBoard output root is not empty: {resolved_output_root}')
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    return tuple(resolved_source_segments), resolved_output_root


def _run_directories(source_root: Path, single_run_source: bool) -> tuple[Path, ...]:
    if single_run_source:
        return (source_root,)
    run_directories = tuple(
        path for path in source_root.iterdir() if path.is_dir() and RUN_NAME_PATTERN.fullmatch(path.name) is not None
    )
    return tuple(sorted(run_directories, key=lambda path: int(path.name.removeprefix('run_'))))


def _representative_self_play_process(run_directory: Path) -> int | None:
    self_play_directory = run_directory / 'self_play'
    if not self_play_directory.is_dir():
        return None
    process_ids = tuple(
        int(path.name) for path in self_play_directory.iterdir() if path.is_dir() and path.name.isdigit()
    )
    return min(process_ids, default=None)


def _event_relative_parts(source_root: Path, event_file: Path, single_run_source: bool) -> tuple[str, ...]:
    relative_parts = event_file.relative_to(source_root).parts
    return ('run_0', *relative_parts) if single_run_source else relative_parts


def select_event_files(source_root: Path, single_run_source: bool = False) -> EventFileSelection:
    all_event_files = tuple(source_root.rglob(EVENT_FILE_PATTERN))
    representative_processes = tuple(
        ('run_0' if single_run_source else run_directory.name, process_id)
        for run_directory in _run_directories(source_root, single_run_source)
        if (process_id := _representative_self_play_process(run_directory)) is not None
    )
    representative_process_by_run = dict(representative_processes)

    selected_event_files: list[Path] = []
    skipped_self_play_event_file_count = 0
    for event_file in all_event_files:
        relative_parts = _event_relative_parts(source_root, event_file, single_run_source)
        if len(relative_parts) >= 4 and relative_parts[1] == 'self_play' and relative_parts[2].isdigit():
            representative_process_id = representative_process_by_run[relative_parts[0]]
            if int(relative_parts[2]) != representative_process_id:
                skipped_self_play_event_file_count += 1
                continue
        selected_event_files.append(event_file)

    selected_event_files.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return EventFileSelection(
        event_files=tuple(selected_event_files),
        source_event_file_count=len(all_event_files),
        skipped_self_play_event_file_count=skipped_self_play_event_file_count,
        representative_self_play_processes=representative_processes,
    )


def _expanded_summary_tag(
    source_root: Path,
    event_file: Path,
    original_tag: str,
    single_run_source: bool,
) -> str:
    relative_parts = _event_relative_parts(source_root, event_file, single_run_source)
    if len(relative_parts) < 2:
        raise ValueError(f'Event file is not inside a run directory: {event_file}')
    category = relative_parts[1]
    writer_directory_parts = relative_parts[2:-1]
    if (
        category in {'self_play', 'cpu_usage_self_play', 'cpu_usage_trainer'}
        and writer_directory_parts
        and writer_directory_parts[0].isdigit()
    ):
        writer_directory_parts = writer_directory_parts[1:]
    original_tag_parts = tuple(original_tag.split('/'))
    if (
        len(writer_directory_parts) > len(original_tag_parts)
        and writer_directory_parts[: len(original_tag_parts)] == original_tag_parts
    ):
        child_series_parts = writer_directory_parts[len(original_tag_parts) :]
        return '/'.join((*original_tag_parts, *child_series_parts))
    return original_tag


def _namespaced_tag(category: str, summary_tag: str) -> str:
    match category:
        case 'self_play':
            return f'self_play/{summary_tag}'
        case 'cpu_usage_self_play':
            return f'system/cpu_self_play/{summary_tag}'
        case 'cpu_usage_trainer':
            return f'system/cpu_trainer/{summary_tag}'
        case 'gpu_usage':
            return f'system/gpu/{summary_tag}'
        case 'training_args':
            return f'configuration/{summary_tag}'
        case _ if category.startswith('evaluation_'):
            return summary_tag
        case 'trainer' | 'dataset':
            return summary_tag
        case _:
            return f'{category}/{summary_tag}'


def canonical_tag(source_root: Path, event_file: Path, original_tag: str, single_run_source: bool = False) -> str:
    category = _event_relative_parts(source_root, event_file, single_run_source)[1]
    return _namespaced_tag(
        category,
        _expanded_summary_tag(source_root, event_file, original_tag, single_run_source),
    )


def time_series_tag(source_root: Path, event_file: Path, original_tag: str, single_run_source: bool = False) -> str:
    category = _event_relative_parts(source_root, event_file, single_run_source)[1]
    grouped_tag = original_tag
    evaluation_match = EVALUATION_TAG_PATTERN.fullmatch(original_tag)
    if evaluation_match is not None and evaluation_match.group(3) in MATCH_OUTCOME_METRICS:
        grouped_tag = '/'.join(evaluation_match.groups()[:2])
    return _namespaced_tag(category, grouped_tag)


def time_series_route(canonical_summary_tag: str, grouped_summary_tag: str) -> TimeSeriesRoute:
    return TimeSeriesRoute(run_name=canonical_summary_tag, tag=grouped_summary_tag)


def source_layout_route(
    source_root: Path,
    event_file: Path,
    original_tag: str,
    single_run_source: bool,
) -> TimeSeriesRoute:
    relative_parts = _event_relative_parts(source_root, event_file, single_run_source)
    category = relative_parts[1]
    writer_directory_parts = relative_parts[2:-1]
    if (
        category in {'self_play', 'cpu_usage_self_play', 'cpu_usage_trainer'}
        and writer_directory_parts
        and writer_directory_parts[0].isdigit()
    ):
        writer_directory_parts = writer_directory_parts[1:]
    return TimeSeriesRoute(
        run_name=str(Path(category, *writer_directory_parts)),
        tag=original_tag,
    )


def _plugin_name(value: summary_pb2.Summary.Value) -> str:
    if value.HasField('metadata') and value.metadata.HasField('plugin_data'):
        return value.metadata.plugin_data.plugin_name
    return ''


def custom_scalar_categories(scalar_tags: set[str]) -> tuple[TensorboardCustomScalarCategory, ...]:
    return (
        *discovered_evaluation_tensorboard_categories(scalar_tags),
        TensorboardCustomScalarCategory(
            title='Optimization',
            charts=(
                TensorboardMultilineChart(
                    title='Training losses',
                    tags=(
                        'train/policy_loss',
                        'train/value_loss',
                        'train/total_loss',
                    ),
                ),
                TensorboardMultilineChart(
                    title='Validation losses',
                    tags=(
                        'validation/policy_loss',
                        'validation/value_loss',
                        'validation/total_loss',
                    ),
                ),
            ),
        ),
    )


def _custom_scalar_layout(scalar_tags: set[str]) -> layout_pb2.Layout:
    categories: list[layout_pb2.Category] = []
    for category_definition in custom_scalar_categories(scalar_tags):
        charts: list[layout_pb2.Chart] = []
        for chart_definition in category_definition.charts:
            charts.append(
                layout_pb2.Chart(
                    title=chart_definition.title,
                    multiline=layout_pb2.MultilineChartContent(tag=chart_definition.tags),
                )
            )
        categories.append(layout_pb2.Category(title=category_definition.title, chart=charts))
    return layout_pb2.Layout(category=categories)


def _write_custom_scalar_layout(writer: EventFileWriter, scalar_tags: set[str]) -> bytes:
    layout = _custom_scalar_layout(scalar_tags)
    serialized_layout = layout.SerializeToString()
    metadata = summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name='custom_scalars')
    )
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        string_val=(serialized_layout,),
        tensor_shape=tensor_shape_pb2.TensorShapeProto(),
    )
    writer.add_event(
        event_pb2.Event(
            wall_time=time.time(),
            summary=summary_pb2.Summary(
                value=(
                    summary_pb2.Summary.Value(
                        tag='custom_scalars__config__',
                        tensor=tensor,
                        metadata=metadata,
                    ),
                )
            ),
        )
    )
    return serialized_layout


def _event_scalar_tags(event_files: tuple[Path, ...]) -> set[str]:
    scalar_tags: set[str] = set()
    for event_file in event_files:
        loader = LegacyEventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.HasField('summary'):
                continue
            for value in event.summary.value:
                if value.WhichOneof('value') == 'simple_value':
                    scalar_tags.add(value.tag)
    return scalar_tags


def append_custom_scalar_layout(output_root: Path) -> None:
    resolved_output_root = output_root.resolve()
    if not resolved_output_root.is_dir():
        raise ValueError(f'TensorBoard output root does not exist: {resolved_output_root}')
    event_files = tuple(resolved_output_root.rglob(EVENT_FILE_PATTERN))
    scalar_tags = _event_scalar_tags(event_files)
    writer = EventFileWriter(str(resolved_output_root))
    try:
        _write_custom_scalar_layout(writer, scalar_tags)
    finally:
        writer.close()


def _summary_identity(
    canonical_summary_tag: str,
    step: int,
    value: summary_pb2.Summary.Value,
) -> SummaryIdentity:
    value_type = value.WhichOneof('value')
    assert value_type is not None, 'TensorBoard summary value has no value type'
    return SummaryIdentity(
        tag=canonical_summary_tag,
        plugin_name=_plugin_name(value),
        value_type=value_type,
        step=step,
    )


class TensorboardLogConsolidator:
    def __init__(
        self,
        source_root: Path | None,
        output_root: Path,
        time_series_view: bool = False,
        single_run_source: bool = False,
        evaluation_outcome_overlays: bool = False,
        preserve_source_layout: bool = False,
        source_segments: tuple[TensorboardSourceSegment, ...] | None = None,
    ) -> None:
        if source_root is not None and source_segments is not None:
            raise ValueError('source_root and source_segments are mutually exclusive.')
        requested_source_segments = source_segments
        if requested_source_segments is None:
            if source_root is None:
                raise ValueError('A TensorBoard source root or source segments are required.')
            requested_source_segments = (TensorboardSourceSegment(source_root=source_root),)
        self.source_segments, self.output_root = _validate_paths(requested_source_segments, output_root)
        if time_series_view and evaluation_outcome_overlays:
            raise ValueError('Time-series view and evaluation outcome overlays are mutually exclusive.')
        if preserve_source_layout and not single_run_source:
            raise ValueError('Preserving the source layout requires a single-run source directory.')
        self.time_series_view = time_series_view
        self.single_run_source = single_run_source
        self.evaluation_outcome_overlays = evaluation_outcome_overlays
        self.preserve_source_layout = preserve_source_layout
        self.writer = (
            None
            if time_series_view or preserve_source_layout
            else EventFileWriter(str(self.output_root), max_queue_size=1_000, flush_secs=10)
        )
        self.time_series_writers: dict[str, EventFileWriter] = {}
        self.written_custom_scalar_layout: bytes | None = None
        self.summary_states: dict[SummaryIdentity, SummaryState] = {}
        self.emitted_summary_states: dict[SummaryIdentity, SummaryState] = {}
        self.event_file_fingerprints: dict[Path, EventFileFingerprint] = {}
        self.emitted_summary_count = 0
        self.replaced_summary_count = 0
        self.excluded_text_summary_count = 0

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        for writer in self.time_series_writers.values():
            writer.close()

    def scan(self) -> ConsolidationManifest:
        selections: list[EventFileSelection] = []
        for source_segment in self.source_segments:
            selection = select_event_files(source_segment.source_root, self.single_run_source)
            selections.append(selection)
            for event_file in selection.event_files:
                fingerprint = EventFileFingerprint(
                    size_bytes=event_file.stat().st_size,
                    modified_time_ns=event_file.stat().st_mtime_ns,
                )
                if self.event_file_fingerprints.get(event_file) == fingerprint:
                    continue
                self._load_event_file(source_segment, event_file)
                self.event_file_fingerprints[event_file] = fingerprint
        self._emit_changed_summaries()
        if self.writer is not None:
            scalar_tags = {identity.tag for identity in self.summary_states if identity.value_type == 'simple_value'}
            layout = _custom_scalar_layout(scalar_tags).SerializeToString()
            if layout != self.written_custom_scalar_layout:
                self.written_custom_scalar_layout = _write_custom_scalar_layout(self.writer, scalar_tags)
            self.writer.flush()
        for writer in self.time_series_writers.values():
            writer.flush()
        manifest = self._manifest(tuple(selections))
        (self.output_root / 'consolidation-manifest.json').write_text(
            manifest.model_dump_json(indent=2),
            encoding='utf-8',
        )
        return manifest

    def _load_event_file(self, source_segment: TensorboardSourceSegment, event_file: Path) -> None:
        loader = LegacyEventFileLoader(str(event_file))
        for event in loader.Load():
            if source_segment.minimum_wall_time is not None and event.wall_time < source_segment.minimum_wall_time:
                continue
            if source_segment.maximum_wall_time is not None and event.wall_time > source_segment.maximum_wall_time:
                continue
            if not event.HasField('summary'):
                continue
            for original_value in event.summary.value:
                category = _event_relative_parts(source_segment.source_root, event_file, self.single_run_source)[1]
                is_evaluation_summary = original_value.tag.startswith('evaluation/summaries/')
                if _plugin_name(original_value) == 'text' and category != 'training_args' and not is_evaluation_summary:
                    self.excluded_text_summary_count += 1
                    continue
                self._select_if_newer(source_segment, event_file, event, original_value)

    def _select_if_newer(
        self,
        source_segment: TensorboardSourceSegment,
        event_file: Path,
        event: event_pb2.Event,
        original_value: summary_pb2.Summary.Value,
    ) -> None:
        consolidated_value = summary_pb2.Summary.Value()
        consolidated_value.CopyFrom(original_value)
        consolidated_value.tag = canonical_tag(
            source_segment.source_root,
            event_file,
            original_value.tag,
            self.single_run_source,
        )
        grouped_summary_tag = time_series_tag(
            source_segment.source_root,
            event_file,
            original_value.tag,
            self.single_run_source,
        )
        source_route = source_layout_route(
            source_segment.source_root,
            event_file,
            original_value.tag,
            self.single_run_source,
        )
        adjusted_step = event.step
        if original_value.tag.startswith(('evaluation/', 'evaluation_metadata/')):
            adjusted_step += source_segment.evaluation_step_offset
        identity = _summary_identity(consolidated_value.tag, adjusted_step, consolidated_value)
        serialized_value = consolidated_value.SerializeToString()
        value_sha256 = hashlib.sha256(serialized_value).hexdigest()
        previous_state = self.summary_states.get(identity)
        if previous_state is not None:
            if event.wall_time < previous_state.wall_time:
                return
            if event.wall_time == previous_state.wall_time and value_sha256 == previous_state.value_sha256:
                return
            if value_sha256 != previous_state.value_sha256:
                self.replaced_summary_count += 1
        self.summary_states[identity] = SummaryState(
            wall_time=event.wall_time,
            value_sha256=value_sha256,
            serialized_value=serialized_value,
            time_series_tag=grouped_summary_tag,
            source_layout_run_name=source_route.run_name,
            source_layout_tag=source_route.tag,
        )

    def _emit_changed_summaries(self) -> None:
        changed_summaries = tuple(
            (identity, state)
            for identity, state in self.summary_states.items()
            if self.emitted_summary_states.get(identity) != state
        )
        for identity, state in sorted(
            changed_summaries,
            key=lambda item: (item[1].wall_time, item[0].step, item[0].tag),
        ):
            consolidated_value = summary_pb2.Summary.Value.FromString(state.serialized_value)
            writer, output_tag = self._writer_and_tag(
                identity.tag,
                state.time_series_tag,
                state.source_layout_run_name,
                state.source_layout_tag,
            )
            consolidated_value.tag = output_tag
            writer.add_event(
                event_pb2.Event(
                    wall_time=state.wall_time,
                    step=identity.step,
                    summary=summary_pb2.Summary(value=(consolidated_value,)),
                )
            )
            self.emitted_summary_states[identity] = state
            self.emitted_summary_count += 1

    def _writer_and_tag(
        self,
        canonical_summary_tag: str,
        grouped_summary_tag: str,
        source_layout_run_name: str,
        source_layout_tag: str,
    ) -> tuple[EventFileWriter, str]:
        if self.preserve_source_layout:
            run_name = source_layout_run_name
            output_tag = source_layout_tag
            evaluation_match = EVALUATION_TAG_PATTERN.fullmatch(source_layout_tag)
            if (
                self.evaluation_outcome_overlays
                and evaluation_match is not None
                and evaluation_match.group(3) in MATCH_OUTCOME_METRICS
            ):
                evaluation_prefix, definition_id, metric = evaluation_match.groups()
                run_name = str(Path(source_layout_run_name, evaluation_prefix, definition_id, metric))
                output_tag = f'{evaluation_prefix}/{definition_id}'
            writer = self.time_series_writers.get(run_name)
            if writer is None:
                writer = EventFileWriter(
                    str(self.output_root / Path(run_name)),
                    max_queue_size=1_000,
                    flush_secs=10,
                )
                self.time_series_writers[run_name] = writer
            return writer, output_tag
        if not self.time_series_view:
            if self.evaluation_outcome_overlays and canonical_summary_tag.endswith(
                tuple(f'/{metric}' for metric in MATCH_OUTCOME_METRICS)
            ):
                route = time_series_route(canonical_summary_tag, grouped_summary_tag)
                writer = self.time_series_writers.get(route.run_name)
                if writer is None:
                    writer = EventFileWriter(
                        str(self.output_root / Path(route.run_name)),
                        max_queue_size=1_000,
                        flush_secs=10,
                    )
                    self.time_series_writers[route.run_name] = writer
                return writer, route.tag
            assert self.writer is not None
            return self.writer, canonical_summary_tag
        route = time_series_route(canonical_summary_tag, grouped_summary_tag)
        writer = self.time_series_writers.get(route.run_name)
        if writer is None:
            writer = EventFileWriter(
                str(self.output_root / Path(route.run_name)),
                max_queue_size=1_000,
                flush_secs=10,
            )
            self.time_series_writers[route.run_name] = writer
        return writer, route.tag

    def _manifest(self, selections: tuple[EventFileSelection, ...]) -> ConsolidationManifest:
        tag_identities: dict[tuple[str, str, str], list[SummaryIdentity]] = {}
        for identity in self.summary_states:
            key = (identity.tag, identity.plugin_name, identity.value_type)
            tag_identities.setdefault(key, []).append(identity)
        tags = tuple(
            TagSummary(
                tag=tag,
                plugin_name=plugin_name,
                value_type=value_type,
                point_count=len(identities),
                minimum_step=min(identity.step for identity in identities),
                maximum_step=max(identity.step for identity in identities),
            )
            for (tag, plugin_name, value_type), identities in sorted(tag_identities.items())
        )
        representatives = tuple(
            RepresentativeSelfPlayProcess(run_name=run_name, process_id=process_id)
            for selection in selections
            for run_name, process_id in selection.representative_self_play_processes
        )
        return ConsolidationManifest(
            generated_at_utc=datetime.now(timezone.utc),
            source_segments=tuple(
                SourceSegmentSummary(
                    source_root=str(source_segment.source_root),
                    minimum_wall_time=source_segment.minimum_wall_time,
                    maximum_wall_time=source_segment.maximum_wall_time,
                    evaluation_step_offset=source_segment.evaluation_step_offset,
                )
                for source_segment in self.source_segments
            ),
            output_root=str(self.output_root),
            source_event_file_count=sum(selection.source_event_file_count for selection in selections),
            selected_event_file_count=sum(len(selection.event_files) for selection in selections),
            skipped_self_play_event_file_count=sum(
                selection.skipped_self_play_event_file_count for selection in selections
            ),
            excluded_text_summary_count=self.excluded_text_summary_count,
            emitted_summary_count=self.emitted_summary_count,
            unique_summary_count=len(self.summary_states),
            replaced_summary_count=self.replaced_summary_count,
            representative_self_play_processes=representatives,
            tags=tags,
        )


class StopSignal:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _: int, __: FrameType | None) -> None:
        self.requested = True


def consolidate_once(
    source_root: Path | None,
    output_root: Path,
    time_series_view: bool = False,
    single_run_source: bool = False,
    evaluation_outcome_overlays: bool = False,
    preserve_source_layout: bool = False,
    source_segments: tuple[TensorboardSourceSegment, ...] | None = None,
) -> ConsolidationManifest:
    consolidator = TensorboardLogConsolidator(
        source_root,
        output_root,
        time_series_view,
        single_run_source,
        evaluation_outcome_overlays,
        preserve_source_layout,
        source_segments,
    )
    try:
        return consolidator.scan()
    finally:
        consolidator.close()


def watch(
    source_root: Path | None,
    output_root: Path,
    interval_seconds: float,
    time_series_view: bool = False,
    single_run_source: bool = False,
    evaluation_outcome_overlays: bool = False,
    preserve_source_layout: bool = False,
    source_segments: tuple[TensorboardSourceSegment, ...] | None = None,
) -> None:
    if interval_seconds <= 0:
        raise ValueError('Watch interval must be positive.')
    stop_signal = StopSignal()
    signal.signal(signal.SIGTERM, stop_signal.request)
    signal.signal(signal.SIGINT, stop_signal.request)
    consolidator = TensorboardLogConsolidator(
        source_root,
        output_root,
        time_series_view,
        single_run_source,
        evaluation_outcome_overlays,
        preserve_source_layout,
        source_segments,
    )
    try:
        while not stop_signal.requested:
            consolidator.scan()
            time.sleep(interval_seconds)
    finally:
        consolidator.close()


def main() -> None:
    arguments = parse_arguments()
    if arguments.custom_layout_only:
        append_custom_scalar_layout(arguments.output_root)
        return
    source_segments = _source_segments(arguments)
    if arguments.watch_interval_seconds is None:
        manifest = consolidate_once(
            None,
            arguments.output_root,
            arguments.time_series_view,
            arguments.single_run_source,
            arguments.evaluation_outcome_overlays,
            arguments.preserve_source_layout,
            source_segments,
        )
        print(manifest.model_dump_json(indent=2))
        return
    watch(
        None,
        arguments.output_root,
        arguments.watch_interval_seconds,
        arguments.time_series_view,
        arguments.single_run_source,
        arguments.evaluation_outcome_overlays,
        arguments.preserve_source_layout,
        source_segments,
    )


if __name__ == '__main__':
    main()
