from dataclasses import dataclass


@dataclass(frozen=True)
class SearchScheduleState:
    schedule_version: int
    num_parallel_searches: int
    num_full_searches: int
    num_fast_searches: int
    endgame_shortcut_strength: float

    @property
    def arena_capacity(self) -> int:
        return max(self.num_full_searches, self.num_fast_searches) + self.num_parallel_searches + 1
