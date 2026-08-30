from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from AlphaZeroCpp import (
        GameSearchVisit,
        InferenceStatistics,
        SearchCheckpoint,
        SearchCheckpointDetail,
        SearchStopReason,
        SelfPlaySearchParameters,
    )


PositionT = TypeVar('PositionT')
NativeRootT = TypeVar('NativeRootT', bound='NativeSearchRoot')
NativeRequestT = TypeVar('NativeRequestT', bound='NativeSearchRequest')
NativeResultT = TypeVar('NativeResultT', bound='NativeSearchResult')
NativeSearchT = TypeVar('NativeSearchT', bound='NativeSelfPlaySearch')


class NativeSearchRoot(Protocol[PositionT]):
    @property
    def position(self) -> PositionT: ...

    @property
    def is_terminal(self) -> bool: ...

    @property
    def visits(self) -> int: ...

    @property
    def live_nodes(self) -> int: ...

    def play(self, action_id: int) -> None: ...

    def reset(self) -> None: ...

    def discount(self, retained_fraction: float) -> None: ...


class NativeSearchRequest(Protocol[NativeRootT]):
    @property
    def root(self) -> NativeRootT: ...

    @property
    def assigned_additional_visits(self) -> int | None: ...

    @property
    def policy_checkpoint_visits(self) -> list[int]: ...

    @property
    def parallel_searches(self) -> int | None: ...

    @property
    def add_root_noise(self) -> bool: ...

    @property
    def force_root_playouts(self) -> bool: ...

    @property
    def checkpoint_detail(self) -> SearchCheckpointDetail: ...


class NativeSearchResult(Protocol[NativeRootT]):
    @property
    def root_value(self) -> float: ...

    @property
    def highest_visited_child_action_id(self) -> int: ...

    @property
    def highest_visited_child_visit_count(self) -> int: ...

    @property
    def highest_visited_child_q(self) -> float: ...

    @property
    def search_visits(self) -> list[GameSearchVisit]: ...

    @property
    def policy_target_visits(self) -> list[GameSearchVisit]: ...

    @property
    def network_root_value(self) -> float: ...

    @property
    def policy_correction(self) -> float: ...

    @property
    def value_correction(self) -> float: ...

    @property
    def predicted_budget_curve(self) -> list[float]: ...

    @property
    def selected_budget_index(self) -> int: ...

    @property
    def assigned_additional_visits(self) -> int: ...

    @property
    def parallel_searches(self) -> int: ...

    @property
    def spend_residual(self) -> int: ...

    @property
    def starting_visits(self) -> int: ...

    @property
    def final_visits(self) -> int: ...

    @property
    def stop_reason(self) -> SearchStopReason: ...

    @property
    def checkpoints(self) -> list[SearchCheckpoint]: ...

    @property
    def root(self) -> NativeRootT: ...


class NativeSearchBatch(Protocol[NativeResultT]):
    @property
    def results(self) -> list[NativeResultT]: ...

    @property
    def simulations_completed(self) -> int: ...


class NativeSelfPlaySearch(Protocol[PositionT, NativeRootT, NativeRequestT, NativeResultT]):
    def new_root(self, position: PositionT) -> NativeRootT: ...

    def request(
        self,
        root: NativeRootT,
        assigned_additional_visits: int | None = None,
        policy_checkpoint_visits: list[int] = ...,
        parallel_searches: int | None = None,
        add_root_noise: bool = True,
        force_root_playouts: bool = True,
        checkpoint_detail: SearchCheckpointDetail = ...,
        root_ply: int = 0,
    ) -> NativeRequestT: ...

    def search(
        self,
        requests: list[NativeRequestT],
        collect_statistics: bool = False,
    ) -> NativeSearchBatch[NativeResultT]: ...

    def refresh_model(self, model_generation: int, model_path: str) -> None: ...

    def update_search_schedule(self, search_parameters: SelfPlaySearchParameters) -> bool: ...

    def reset_spend_residual(self) -> None: ...

    @property
    def spend_residual(self) -> int: ...

    def inference_statistics(self) -> InferenceStatistics: ...
