def select_validation_iteration(current_iteration: int, current_iteration_has_files: bool) -> int:
    if current_iteration < 0:
        raise ValueError('Current iteration cannot be negative.')
    if current_iteration_has_files:
        return current_iteration
    if current_iteration == 0:
        raise ValueError('Iteration zero cannot fall back to an earlier validation dataset.')
    return current_iteration - 1
