from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt
import torch
from src.games.contracts import GameStateContract
from src.replay.batch_loader import decode_augmented_states
from src.replay.columnar import ReplaySearchBudgetColumnViews
from src.replay.description import ReplayDescription
from src.replay.store import ReplayStore


@dataclass(frozen=True)
class SearchBudgetHeadBatch:
    """Every row carries a search-budget label; there is no eligibility mask because nothing is masked out."""

    states: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def to_device(self, device: torch.device, non_blocking: bool) -> SearchBudgetHeadBatch:
        return SearchBudgetHeadBatch(
            states=self.states.to(device=device, non_blocking=non_blocking),
            targets=self.targets.to(device=device, non_blocking=non_blocking),
        )


def build_search_budget_head_batch(
    store: ReplayStore,
    state: GameStateContract,
    auxiliary_index: int,
    logical_indices: npt.NDArray[np.int64],
    augmentation_indices: npt.NDArray[np.int64],
) -> SearchBudgetHeadBatch:
    if len(logical_indices) == 0:
        raise ValueError('Search-budget head batches cannot be empty.')
    if len(logical_indices) != len(augmentation_indices):
        raise ValueError('Every search-budget head row requires one augmentation index.')
    columns = store.gather_logical(logical_indices)
    budget = columns.auxiliary[auxiliary_index]
    if not isinstance(budget, ReplaySearchBudgetColumnViews):
        raise ValueError(f'Replay auxiliary head {auxiliary_index} is not a search-budget head.')
    if not bool(np.all(budget.eligible)):
        raise ValueError('Search-budget head batches must be drawn from labelled replay rows only.')
    states = decode_augmented_states(columns.encoded_state, state, augmentation_indices)
    return SearchBudgetHeadBatch(
        states=torch.from_numpy(states),
        targets=torch.from_numpy(np.ascontiguousarray(budget.value, dtype=np.float32)),
    )


class SearchBudgetLabelPool:
    """Live labelled replay rows for one search-budget head, indexed once per training quantum."""

    def __init__(
        self,
        replay: ReplayDescription,
        state: GameStateContract,
        auxiliary_index: int,
    ) -> None:
        self.state = state
        self.auxiliary_index = auxiliary_index
        self._store = _open_pinned_store(replay)
        try:
            self._logical_indices = self._store.eligible_logical_indices(auxiliary_index)
        except BaseException:
            self._store.close()
            raise

    @property
    def size(self) -> int:
        return int(self._logical_indices.shape[0])

    def select_logical_indices(self, generator: np.random.Generator, rows: int) -> npt.NDArray[np.int64]:
        if not 0 < rows <= self.size:
            raise ValueError('Search-budget head batch rows must fit the labelled pool.')
        return np.asarray(generator.choice(self._logical_indices, size=rows, replace=False), dtype=np.int64)

    def batch(
        self,
        logical_indices: npt.NDArray[np.int64],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> SearchBudgetHeadBatch:
        return build_search_budget_head_batch(
            self._store,
            self.state,
            self.auxiliary_index,
            logical_indices,
            augmentation_indices,
        )

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> SearchBudgetLabelPool:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _open_pinned_store(replay: ReplayDescription) -> ReplayStore:
    store = ReplayStore.open(Path(replay.path), replay.layout, writable=False)
    state = store.state
    if state.head != replay.head or state.size != replay.size:
        store.close()
        raise ValueError('Replay changed after the training description was captured.')
    return store
