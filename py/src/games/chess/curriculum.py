def curriculum_progress(iteration: int, warmup_iterations: int) -> float:
    if iteration < 0:
        raise ValueError('Curriculum iteration cannot be negative.')
    if warmup_iterations < 0:
        raise ValueError('Curriculum warm-up cannot be negative.')
    if warmup_iterations == 0:
        return 1.0
    return min(iteration / warmup_iterations, 1.0)


def curriculum_fade(iteration: int, fade_iterations: int) -> float:
    return 1.0 - curriculum_progress(iteration, fade_iterations)
