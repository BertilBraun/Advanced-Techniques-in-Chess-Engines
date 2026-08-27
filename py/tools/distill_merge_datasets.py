from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.distillation.dataset import DistillationDatasetManifest, manifest_path, read_manifest, record_dtype
from src.util.atomic_file import write_text_atomically

COMPATIBILITY_FIELDS = (
    'game',
    'action_size',
    'payload_bytes',
    'maximum_policy_entries',
    'maximum_legal_actions',
    'teacher_generation',
    'teacher_weights_sha256',
    'teacher_parameter_count',
)


def _require_compatible(sources: tuple[DistillationDatasetManifest, ...]) -> None:
    first = sources[0]
    for manifest in sources[1:]:
        for field in COMPATIBILITY_FIELDS:
            if getattr(manifest, field) != getattr(first, field):
                raise ValueError(
                    f'Datasets disagree on {field}: {getattr(first, field)!r} against {getattr(manifest, field)!r}.'
                )

    # Shards generated with one seed hold identical games, so merging them multiplies the file size and not the
    # content. The seeds are the only evidence of that available without comparing the rows themselves.
    seeds = [manifest.random_seed for manifest in sources]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Datasets must come from distinct random seeds; got {seeds}.')


def merge_datasets(inputs: tuple[Path, ...], output: Path) -> DistillationDatasetManifest:
    manifests = tuple(read_manifest(path) for path in inputs)
    _require_compatible(manifests)

    itemsize = record_dtype(manifests[0].payload_bytes).itemsize
    for path, manifest in zip(inputs, manifests, strict=True):
        expected = itemsize * manifest.position_count
        if path.stat().st_size != expected:
            raise ValueError(f'{path} holds {path.stat().st_size} bytes but its manifest implies {expected}.')

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as destination:
        for path in inputs:
            with path.open('rb') as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

    # Generator settings are taken from the last source because the held-out tail comes from it.
    merged = manifests[-1].model_copy(
        update={
            'position_count': sum(manifest.position_count for manifest in manifests),
            'merged_sources': tuple(path.name for path in inputs),
        }
    )
    write_text_atomically(manifest_path(output), merged.model_dump_json(indent=2))
    return merged


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action='append', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if len(arguments.input) < 2:
        raise ValueError('Merging requires at least two input datasets.')
    if arguments.output.exists():
        raise ValueError(f'Refusing to overwrite {arguments.output}.')
    merged = merge_datasets(tuple(arguments.input), arguments.output)
    print(f'Merged {len(arguments.input)} datasets into {arguments.output}: {merged.position_count} positions.')


if __name__ == '__main__':
    main()
