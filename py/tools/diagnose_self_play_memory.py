from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from AlphaZeroCpp import InferenceClientParams, MCTS, MCTSBoard, MCTSParams

if TYPE_CHECKING:
    from AlphaZeroCpp import MCTSRoot


INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
KIB_PER_MIB = 1024


@dataclass(frozen=True)
class MemorySnapshot:
    transition: int
    boundary: str
    rss_mib: float
    pss_mib: float
    anonymous_mib: float
    pss_anon_mib: float
    private_dirty_mib: float
    thread_count: int


class SyntheticInferenceModel(nn.Module):
    def __init__(self, padding_elements: int) -> None:
        super().__init__()
        self.policy_logits = nn.Parameter(torch.zeros(1880))
        self.value_logits = nn.Parameter(torch.zeros(3))
        self.padding = nn.Parameter(torch.zeros(padding_elements))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = inputs.size(0)
        retained_padding = self.padding[0] * 0.0
        policies = torch.softmax(self.policy_logits + retained_padding, dim=0).expand(batch_size, -1)
        values = torch.softmax(self.value_logits + retained_padding, dim=0).expand(batch_size, -1)
        return policies, values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure native self-play memory at model-transition lifecycle boundaries.'
    )
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--verification-model-path', type=Path)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--transitions', type=int, default=12)
    parser.add_argument('--roots', type=int, default=24)
    parser.add_argument('--searches', type=int, default=50)
    parser.add_argument('--parallel-searches', type=int, default=3)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--synthetic-model-mib', type=int, default=32)
    parser.add_argument('--maximum-growth-mib', type=float)
    parser.add_argument(
        '--legacy-order',
        action='store_true',
        help='Release roots after replacement construction, reproducing the pre-fix lifecycle.',
    )
    parser.add_argument(
        '--torchscript-only',
        action='store_true',
        help='Isolate repeated TorchScript loading without constructing the native client.',
    )
    parser.add_argument(
        '--persistent-mcts',
        action='store_true',
        help='Reload the model in place using the fixed persistent-client lifecycle.',
    )
    return parser.parse_args()


def read_memory_snapshot(transition: int, boundary: str) -> MemorySnapshot:
    smaps_values: dict[str, int] = {}
    for line in Path('/proc/self/smaps_rollup').read_text(encoding='utf-8').splitlines():
        key, separator, value = line.partition(':')
        if separator and value.strip().endswith('kB'):
            smaps_values[key] = int(value.split()[0])

    return MemorySnapshot(
        transition=transition,
        boundary=boundary,
        rss_mib=smaps_values['Rss'] / KIB_PER_MIB,
        pss_mib=smaps_values['Pss'] / KIB_PER_MIB,
        anonymous_mib=smaps_values['Anonymous'] / KIB_PER_MIB,
        pss_anon_mib=smaps_values['Pss_Anon'] / KIB_PER_MIB,
        private_dirty_mib=smaps_values['Private_Dirty'] / KIB_PER_MIB,
        thread_count=len(os.listdir('/proc/self/task')),
    )


def save_synthetic_model(path: Path, model_mib: int) -> None:
    if model_mib <= 0:
        raise ValueError('Synthetic model size must be positive.')
    padding_elements = model_mib * 2**20 // torch.tensor([], dtype=torch.float32).element_size()
    torch.jit.script(SyntheticInferenceModel(padding_elements)).save(str(path))


def create_mcts(
    model_path: Path,
    searches: int,
    parallel_searches: int,
    threads: int,
    device_id: int,
    use_inference_cache: bool = False,
) -> MCTS:
    client_parameters = InferenceClientParams(
        device_id,
        currentModelPath=str(model_path),
        maxBatchSize=256,
        microsecondsTimeoutInferenceThread=500,
        cacheCapacity=4096,
    )
    mcts_parameters = create_mcts_parameters(searches, parallel_searches, threads)
    return MCTS(client_parameters, mcts_parameters, use_inference_cache=use_inference_cache)


def create_mcts_parameters(searches: int, parallel_searches: int, threads: int) -> MCTSParams:
    return MCTSParams(
        num_parallel_searches=parallel_searches,
        num_full_searches=searches,
        num_fast_searches=searches,
        c_param=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        min_visit_count=0,
        num_threads=threads,
    )


def populate_roots(mcts: MCTS, root_count: int) -> list[MCTSRoot]:
    roots = [mcts.new_root(INITIAL_FEN) for _ in range(root_count)]
    boards = [MCTSBoard(root, False) for root in roots]
    results = mcts.search(boards, collect_statistics=True)
    return [result.root for result in results.results]


def verify_model_update(
    initial_model_path: Path,
    updated_model_path: Path,
    searches: int,
    parallel_searches: int,
    threads: int,
    device_id: int,
) -> None:
    for use_inference_cache in (False, True):
        mcts = create_mcts(
            initial_model_path,
            searches,
            parallel_searches,
            threads,
            device_id,
            use_inference_cache=use_inference_cache,
        )
        initial_output = mcts.inference(INITIAL_FEN)
        mcts.refresh_model(1, str(updated_model_path))
        updated_output = mcts.inference(INITIAL_FEN)
        if updated_output == initial_output:
            client_kind = 'caching' if use_inference_cache else 'non-caching'
            raise RuntimeError(f'{client_kind} model output did not change after the update.')


def run_diagnostic(
    model_path: Path,
    transitions: int,
    root_count: int,
    searches: int,
    parallel_searches: int,
    threads: int,
    device_id: int,
    legacy_order: bool,
    persistent_mcts: bool,
) -> list[MemorySnapshot]:
    snapshots = [read_memory_snapshot(-1, 'startup')]
    mcts = create_mcts(model_path, searches, parallel_searches, threads, device_id)
    roots = populate_roots(mcts, root_count)
    snapshots.append(read_memory_snapshot(-1, 'initial_gameplay'))

    for transition in range(transitions):
        snapshots.append(read_memory_snapshot(transition, 'before_statistics'))
        mcts.get_inference_statistics()
        snapshots.append(read_memory_snapshot(transition, 'after_statistics'))

        if persistent_mcts:
            mcts.refresh_model(transition + 1, str(model_path))
            snapshots.append(read_memory_snapshot(transition, 'after_model_update'))
            roots.clear()
            gc.collect()
            snapshots.append(read_memory_snapshot(transition, 'after_root_release'))
            roots = populate_roots(mcts, root_count)
            snapshots.append(read_memory_snapshot(transition, 'after_gameplay'))
            continue

        if not legacy_order:
            roots.clear()
            gc.collect()
            snapshots.append(read_memory_snapshot(transition, 'after_root_release'))

        del mcts
        gc.collect()
        snapshots.append(read_memory_snapshot(transition, 'after_mcts_destruction'))
        mcts = create_mcts(model_path, searches, parallel_searches, threads, device_id)
        snapshots.append(read_memory_snapshot(transition, 'after_model_construction'))

        if legacy_order:
            roots.clear()
            gc.collect()
            snapshots.append(read_memory_snapshot(transition, 'after_root_release'))

        roots = populate_roots(mcts, root_count)
        snapshots.append(read_memory_snapshot(transition, 'after_gameplay'))

    roots.clear()
    del mcts
    gc.collect()
    snapshots.append(read_memory_snapshot(transitions, 'shutdown'))
    return snapshots


def run_torchscript_diagnostic(
    model_path: Path,
    transitions: int,
    device_id: int,
) -> list[MemorySnapshot]:
    snapshots = [read_memory_snapshot(-1, 'startup')]
    device = torch.device('cuda', device_id) if torch.cuda.is_available() else torch.device('cpu')
    data_type = torch.bfloat16 if device.type == 'cuda' else torch.float32
    inputs = torch.zeros((1, 29, 8, 8), device=device, dtype=data_type)

    for transition in range(transitions):
        snapshots.append(read_memory_snapshot(transition, 'before_model_load'))
        model = torch.jit.load(str(model_path), map_location=device)
        model.to(dtype=data_type)
        model.eval()
        snapshots.append(read_memory_snapshot(transition, 'after_model_load'))
        with torch.no_grad():
            outputs = model(inputs)
        snapshots.append(read_memory_snapshot(transition, 'after_model_forward'))
        del outputs
        del model
        gc.collect()
        snapshots.append(read_memory_snapshot(transition, 'after_model_destruction'))

    return snapshots


def main() -> None:
    arguments = parse_arguments()
    if arguments.transitions <= 0 or arguments.roots <= 0:
        raise ValueError('Transition and root counts must be positive.')

    with tempfile.TemporaryDirectory(prefix='self-play-memory-') as temporary_directory:
        model_path = arguments.model_path
        if model_path is None:
            model_path = Path(temporary_directory) / 'synthetic.jit.pt'
            save_synthetic_model(model_path, arguments.synthetic_model_mib)
        if not model_path.is_file():
            raise ValueError(f'Model does not exist: {model_path}')
        if arguments.verification_model_path is not None:
            if not arguments.verification_model_path.is_file():
                raise ValueError(f'Verification model does not exist: {arguments.verification_model_path}')
            verify_model_update(
                initial_model_path=model_path,
                updated_model_path=arguments.verification_model_path,
                searches=arguments.searches,
                parallel_searches=arguments.parallel_searches,
                threads=arguments.threads,
                device_id=arguments.device_id,
            )

        snapshots = (
            run_torchscript_diagnostic(
                model_path=model_path,
                transitions=arguments.transitions,
                device_id=arguments.device_id,
            )
            if arguments.torchscript_only
            else run_diagnostic(
                model_path=model_path,
                transitions=arguments.transitions,
                root_count=arguments.roots,
                searches=arguments.searches,
                parallel_searches=arguments.parallel_searches,
                threads=arguments.threads,
                device_id=arguments.device_id,
                legacy_order=arguments.legacy_order,
                persistent_mcts=arguments.persistent_mcts,
            )
        )

    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(
        json.dumps([asdict(snapshot) for snapshot in snapshots], indent=2) + '\n',
        encoding='utf-8',
    )
    gameplay_snapshots = [snapshot for snapshot in snapshots if snapshot.boundary == 'after_gameplay']
    if arguments.maximum_growth_mib is not None and len(gameplay_snapshots) >= 2:
        rss_growth_mib = gameplay_snapshots[-1].rss_mib - gameplay_snapshots[0].rss_mib
        if rss_growth_mib > arguments.maximum_growth_mib:
            raise RuntimeError(
                f'RSS grew by {rss_growth_mib:.3f} MiB; limit is {arguments.maximum_growth_mib:.3f} MiB.'
            )


if __name__ == '__main__':
    main()
