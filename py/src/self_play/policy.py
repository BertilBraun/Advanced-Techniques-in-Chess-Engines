from src.self_play.completed_game import SearchObservation, SparseSearchVisit


def preprocessed_search_visits(observation: SearchObservation) -> tuple[SparseSearchVisit, ...]:
    adjusted_visits = (
        SparseSearchVisit(
            action_id=visit.action_id,
            visit_count=visit.visit_count - observation.minimum_root_visits,
        )
        for visit in observation.visits
        if visit.visit_count > observation.minimum_root_visits
    )
    return tuple(sorted(adjusted_visits, key=lambda visit: (-visit.visit_count, visit.action_id)))
