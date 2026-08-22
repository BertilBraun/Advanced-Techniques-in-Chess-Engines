from __future__ import annotations

from typing import TYPE_CHECKING

from src.self_play.completed_game import SearchObservation

if TYPE_CHECKING:
    from AlphaZeroCpp import GameSearchVisit


def ordered_search_visits(observation: SearchObservation) -> tuple[GameSearchVisit, ...]:
    # Imported here so replay materialization stays importable without the native extension.
    from AlphaZeroCpp import GameSearchVisit

    return tuple(
        GameSearchVisit(action_id=action_id, visit_count=visit_count)
        for action_id, visit_count in sorted(
            zip(
                observation.policy_target_visits.action_ids,
                observation.policy_target_visits.visit_counts,
                strict=True,
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
