def evaluation_device_for_task(device_cycle: tuple[int, ...], task_index: int) -> int:
    if not device_cycle:
        raise ValueError('Evaluation device cycle cannot be empty.')
    if task_index < 0:
        raise ValueError('Evaluation task index cannot be negative.')
    return device_cycle[task_index % len(device_cycle)]


def select_historical_model_iterations(
    current_iteration: int,
    historical_model_iterations: tuple[int, ...],
    milestone_interval: int,
    rotation_period: int,
    evaluation_interval: int = 1,
) -> tuple[int, ...]:
    if milestone_interval <= 0:
        raise ValueError('Milestone interval must be positive.')
    if rotation_period <= 0:
        raise ValueError('Historical-model rotation period must be positive.')
    if evaluation_interval <= 0:
        raise ValueError('Evaluation interval must be positive.')

    current_bucket = (current_iteration // evaluation_interval) % rotation_period
    return tuple(
        historical_iteration
        for historical_iteration in historical_model_iterations
        if historical_iteration < current_iteration
        and (historical_iteration // milestone_interval) % rotation_period == current_bucket
    )
