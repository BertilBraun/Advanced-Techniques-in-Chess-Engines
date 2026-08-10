from src.self_play.completed_game import SearchObservation, SparseSearchVisit


def ordered_search_visits(observation: SearchObservation) -> tuple[SparseSearchVisit, ...]:
    return tuple(sorted(observation.policy_target_visits, key=lambda visit: (-visit.visit_count, visit.action_id)))
