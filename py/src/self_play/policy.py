from AlphaZeroCpp import GameSearchVisit

from src.self_play.completed_game import SearchObservation


def ordered_search_visits(observation: SearchObservation) -> tuple[GameSearchVisit, ...]:
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
