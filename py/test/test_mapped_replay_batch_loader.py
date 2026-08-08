from pathlib import Path

import torch

from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.contracts import WdlTarget
from src.replay.batch_loader import MappedReplayBatchLoader
from src.replay.contracts import EligibleNextPolicyTarget, ReplaySample, SparsePolicyTarget
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.self_play.completed_game import SparseSearchVisit
from src.training.targets import NextPolicyHeadLayout, TrainingTargetLayout


def _layout() -> ReplayLayout:
    action_size = CHESS_STATE_CONTRACT.action_size
    return ReplayLayout(
        packed_planes=CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=TrainingTargetLayout(
            action_size=action_size,
            wdl_size=3,
            auxiliary_heads=(NextPolicyHeadLayout(kind='next_policy', action_size=action_size, ply_offset=1),),
        ),
        maximum_policy_entries=2,
    )


def _sample(weight: float) -> ReplaySample:
    position = CHESS_STATE_CONTRACT.initial_position()
    first, second = CHESS_STATE_CONTRACT.legal_action_ids(position)[:2]
    primary = SparsePolicyTarget(
        visits=(
            SparseSearchVisit(action_id=first, visit_count=3),
            SparseSearchVisit(action_id=second, visit_count=1),
        )
    )
    return ReplaySample(
        encoded_state=CHESS_STATE_CONTRACT.encode_network_input(position),
        policy=primary,
        wdl_target=WdlTarget(win=0.25, draw=0.5, loss=0.25),
        root_value=0.125,
        auxiliary_targets=(EligibleNextPolicyTarget(policy=primary),),
        sample_weight=weight,
        source_model_generation=1,
        source_created_at_seconds=10.0,
    )


def _description(path: Path, store: ReplayStore) -> ReplayDescription:
    state = store.state
    return ReplayDescription(
        path=path,
        head=state.head,
        size=state.size,
        logical_capacity=state.logical_capacity,
        maximum_capacity=state.maximum_capacity,
        layout=store.layout,
    )


def test_mapped_loader_builds_canonical_batches_and_disjoint_rank_slices(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(), maximum_capacity=4, logical_capacity=4)
    for weight in (1.0, 2.0, 3.0, 4.0):
        store.append(_sample(weight))
    store.flush()
    description = _description(path, store)
    store.close()

    common = {
        'replay': description,
        'state': CHESS_STATE_CONTRACT,
        'source_optimizer_step': 20,
        'optimizer_steps': 1,
        'global_batch_size': 4,
        'world_size': 2,
        'sampler_seed': 91,
        'pin_memory': False,
    }
    rank_zero = next(iter(MappedReplayBatchLoader(rank=0, **common)))
    rank_one = next(iter(MappedReplayBatchLoader(rank=1, **common)))

    assert rank_zero.states.shape == (2, CHESS_STATE_CONTRACT.representation.channels, 8, 8)
    assert rank_zero.policy_targets.shape == (2, CHESS_STATE_CONTRACT.action_size)
    assert torch.allclose(rank_zero.policy_targets.sum(dim=1), torch.ones(2))
    assert torch.allclose(rank_zero.wdl_targets, torch.tensor(((0.25, 0.5, 0.25),) * 2))
    assert torch.all(rank_zero.auxiliary_eligibility[0])
    assert set(rank_zero.sample_weights.tolist()).isdisjoint(rank_one.sample_weights.tolist())
