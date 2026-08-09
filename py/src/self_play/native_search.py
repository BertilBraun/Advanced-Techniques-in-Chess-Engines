from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceStatistics, SelfPlaySearchParameters


PositionT = TypeVar('PositionT')
NativeRootT = TypeVar('NativeRootT', bound='NativeSearchRoot')
NativeRequestT = TypeVar('NativeRequestT', bound='NativeSearchRequest')
NativeResultT = TypeVar('NativeResultT', bound='NativeSearchResult')
NativeSearchT = TypeVar('NativeSearchT', bound='NativeSelfPlaySearch')


class NativeSearchVisit(Protocol):
    @property
    def action_id(self) -> int: ...

    @property
    def visit_count(self) -> int: ...


class NativeSearchRoot(Protocol[PositionT]):
    @property
    def position(self) -> PositionT: ...

    @property
    def is_terminal(self) -> bool: ...

    def play(self, action_id: int) -> None: ...

    def reset(self) -> None: ...

    def discount(self, retained_fraction: float) -> None: ...


class NativeSearchRequest(Protocol[NativeRootT]):
    @property
    def root(self) -> NativeRootT: ...

    @property
    def full_search(self) -> bool: ...


class NativeSearchResult(Protocol[NativeRootT]):
    @property
    def root_value(self) -> float: ...

    @property
    def visits(self) -> list[NativeSearchVisit]: ...

    @property
    def root(self) -> NativeRootT: ...


class NativeSearchBatch(Protocol[NativeResultT]):
    @property
    def results(self) -> list[NativeResultT]: ...

    @property
    def simulations_completed(self) -> int: ...


class NativeSelfPlaySearch(Protocol[PositionT, NativeRootT, NativeRequestT, NativeResultT]):
    def new_root(self, position: PositionT) -> NativeRootT: ...

    def request(self, root: NativeRootT, full_search: bool) -> NativeRequestT: ...

    def search(
        self,
        requests: list[NativeRequestT],
        collect_statistics: bool = False,
    ) -> NativeSearchBatch[NativeResultT]: ...

    def refresh_model(self, model_generation: int, model_path: str) -> None: ...

    def update_search_schedule(self, search_parameters: SelfPlaySearchParameters) -> bool: ...

    def inference_statistics(self) -> InferenceStatistics: ...
