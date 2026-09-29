"""Registers the scripted Lc0 teacher as a checkpoint so the existing match tooling can play it.

The Stockfish gauntlet loads a model through a run directory's `checkpoint_<generation>.json`, and
for inference only validates the inference artifact and its hash. The teacher is copied in beside a
manifest that points at it, so it is played by exactly the same search, openings and opponents as a
project checkpoint and the two results differ only in the network.

The network block is copied from a project manifest because the schema requires one; nothing on the
inference path builds that architecture. Only the input dimensions are changed, to Lc0's 112 planes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.games.chess.contract import LC0_CHANNEL_COUNT
from src.util.atomic_file import write_text_atomically
from src.util.hashing import file_sha256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-model', type=Path, required=True, help='Output of build_lc0_teacher_model.py.')
    parser.add_argument('--template-manifest', type=Path, required=True, help='Any project checkpoint_N.json.')
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--generation', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.run_directory.mkdir(parents=True, exist_ok=True)
    model_name = f'model_{arguments.generation}.jit.pt'
    target = arguments.run_directory / model_name
    shutil.copyfile(arguments.teacher_model, target)
    digest = file_sha256(target)

    manifest = json.loads(arguments.template_manifest.read_text(encoding='utf-8'))
    manifest['generation'] = arguments.generation
    manifest['network']['dimensions']['channels'] = LC0_CHANNEL_COUNT
    manifest['network']['auxiliary_heads'] = []
    for key in ('model', 'optimizer', 'inference_model'):
        manifest[f'{key}_path'] = model_name
        manifest[f'{key}_sha256'] = digest
    manifest['qat'] = None
    manifest['policy_prior_calibration'] = None

    manifest_path = arguments.run_directory / f'checkpoint_{arguments.generation}.json'
    write_text_atomically(manifest_path, json.dumps(manifest, indent=2) + '\n')
    print(f'Registered {target} ({digest}) as generation {arguments.generation} in {manifest_path}')


if __name__ == '__main__':
    main()
