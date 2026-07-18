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
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


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


class ConsolidationManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at_utc: datetime
    source_root: str
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
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--watch-interval-seconds', type=float)
    return parser.parse_args()


def _validate_paths(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    resolved_source_root = source_root.resolve()
    resolved_output_root = output_root.resolve()
    if not resolved_source_root.is_dir():
        raise ValueError(f'TensorBoard source root does not exist: {resolved_source_root}')
    if resolved_output_root == resolved_source_root or resolved_source_root in resolved_output_root.parents:
        raise ValueError('Output root must not be the source root or one of its descendants.')
    if resolved_output_root.exists() and any(resolved_output_root.iterdir()):
        raise ValueError(f'TensorBoard output root is not empty: {resolved_output_root}')
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    return resolved_source_root, resolved_output_root


def _run_directories(source_root: Path) -> tuple[Path, ...]:
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


def select_event_files(source_root: Path) -> EventFileSelection:
    all_event_files = tuple(source_root.rglob(EVENT_FILE_PATTERN))
    representative_processes = tuple(
        (run_directory.name, process_id)
        for run_directory in _run_directories(source_root)
        if (process_id := _representative_self_play_process(run_directory)) is not None
    )
    representative_process_by_run = dict(representative_processes)

    selected_event_files: list[Path] = []
    skipped_self_play_event_file_count = 0
    for event_file in all_event_files:
        relative_parts = event_file.relative_to(source_root).parts
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


def _expanded_summary_tag(source_root: Path, event_file: Path, original_tag: str) -> str:
    relative_parts = event_file.relative_to(source_root).parts
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


def canonical_tag(source_root: Path, event_file: Path, original_tag: str) -> str:
    relative_parts = event_file.relative_to(source_root).parts
    category = relative_parts[1]
    expanded_summary_tag = _expanded_summary_tag(source_root, event_file, original_tag)
    match category:
        case 'self_play':
            return f'self_play/{expanded_summary_tag}'
        case 'cpu_usage_self_play':
            return f'system/cpu_self_play/{expanded_summary_tag}'
        case 'cpu_usage_trainer':
            return f'system/cpu_trainer/{expanded_summary_tag}'
        case 'gpu_usage':
            return f'system/gpu/{expanded_summary_tag}'
        case 'training_args':
            return f'configuration/{expanded_summary_tag}'
        case _ if category.startswith('evaluation_'):
            return expanded_summary_tag
        case 'trainer' | 'dataset':
            return expanded_summary_tag
        case _:
            return f'{category}/{expanded_summary_tag}'


def _plugin_name(value: summary_pb2.Summary.Value) -> str:
    if value.HasField('metadata') and value.metadata.HasField('plugin_data'):
        return value.metadata.plugin_data.plugin_name
    return ''


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
    def __init__(self, source_root: Path, output_root: Path) -> None:
        self.source_root, self.output_root = _validate_paths(source_root, output_root)
        self.writer = EventFileWriter(str(self.output_root), max_queue_size=1_000, flush_secs=10)
        self.summary_states: dict[SummaryIdentity, SummaryState] = {}
        self.emitted_summary_states: dict[SummaryIdentity, SummaryState] = {}
        self.event_file_fingerprints: dict[Path, EventFileFingerprint] = {}
        self.emitted_summary_count = 0
        self.replaced_summary_count = 0
        self.excluded_text_summary_count = 0

    def close(self) -> None:
        self.writer.close()

    def scan(self) -> ConsolidationManifest:
        selection = select_event_files(self.source_root)
        for event_file in selection.event_files:
            fingerprint = EventFileFingerprint(
                size_bytes=event_file.stat().st_size,
                modified_time_ns=event_file.stat().st_mtime_ns,
            )
            if self.event_file_fingerprints.get(event_file) == fingerprint:
                continue
            self._load_event_file(event_file)
            self.event_file_fingerprints[event_file] = fingerprint
        self._emit_changed_summaries()
        self.writer.flush()
        manifest = self._manifest(selection)
        (self.output_root / 'consolidation-manifest.json').write_text(
            manifest.model_dump_json(indent=2),
            encoding='utf-8',
        )
        return manifest

    def _load_event_file(self, event_file: Path) -> None:
        loader = LegacyEventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.HasField('summary'):
                continue
            for original_value in event.summary.value:
                if _plugin_name(original_value) == 'text' and event_file.relative_to(self.source_root).parts[1] != (
                    'training_args'
                ):
                    self.excluded_text_summary_count += 1
                    continue
                self._select_if_newer(event_file, event, original_value)

    def _select_if_newer(
        self,
        event_file: Path,
        event: event_pb2.Event,
        original_value: summary_pb2.Summary.Value,
    ) -> None:
        consolidated_value = summary_pb2.Summary.Value()
        consolidated_value.CopyFrom(original_value)
        consolidated_value.tag = canonical_tag(self.source_root, event_file, original_value.tag)
        identity = _summary_identity(consolidated_value.tag, event.step, consolidated_value)
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
            self.writer.add_event(
                event_pb2.Event(
                    wall_time=state.wall_time,
                    step=identity.step,
                    summary=summary_pb2.Summary(value=(consolidated_value,)),
                )
            )
            self.emitted_summary_states[identity] = state
            self.emitted_summary_count += 1

    def _manifest(self, selection: EventFileSelection) -> ConsolidationManifest:
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
            for run_name, process_id in selection.representative_self_play_processes
        )
        return ConsolidationManifest(
            generated_at_utc=datetime.now(timezone.utc),
            source_root=str(self.source_root),
            output_root=str(self.output_root),
            source_event_file_count=selection.source_event_file_count,
            selected_event_file_count=len(selection.event_files),
            skipped_self_play_event_file_count=selection.skipped_self_play_event_file_count,
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


def consolidate_once(source_root: Path, output_root: Path) -> ConsolidationManifest:
    consolidator = TensorboardLogConsolidator(source_root, output_root)
    try:
        return consolidator.scan()
    finally:
        consolidator.close()


def watch(source_root: Path, output_root: Path, interval_seconds: float) -> None:
    if interval_seconds <= 0:
        raise ValueError('Watch interval must be positive.')
    stop_signal = StopSignal()
    signal.signal(signal.SIGTERM, stop_signal.request)
    signal.signal(signal.SIGINT, stop_signal.request)
    consolidator = TensorboardLogConsolidator(source_root, output_root)
    try:
        while not stop_signal.requested:
            consolidator.scan()
            time.sleep(interval_seconds)
    finally:
        consolidator.close()


def main() -> None:
    arguments = parse_arguments()
    if arguments.watch_interval_seconds is None:
        manifest = consolidate_once(arguments.source_root, arguments.output_root)
        print(manifest.model_dump_json(indent=2))
        return
    watch(arguments.source_root, arguments.output_root, arguments.watch_interval_seconds)


if __name__ == '__main__':
    main()
