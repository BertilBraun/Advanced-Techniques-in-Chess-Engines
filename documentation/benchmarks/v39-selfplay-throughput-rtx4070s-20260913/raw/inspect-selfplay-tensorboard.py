from __future__ import annotations

import glob
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


event_paths = glob.glob('/workspace/tensorboard/vast-chess-8gpu-progressive-v39-int8/self_play/*/events*')
results: dict[str, object] = {'event_paths': event_paths, 'scalars': {}}
for event_path in event_paths:
    accumulator = EventAccumulator(event_path)
    accumulator.Reload()
    scalar_results: dict[str, list[dict[str, float | int]]] = {}
    for tag in accumulator.Tags()['scalars']:
        scalar_results[tag] = [
            {'step': event.step, 'wall_time': event.wall_time, 'value': event.value}
            for event in accumulator.Scalars(tag)
        ]
    results['scalars'] = scalar_results
    results['event_file'] = str(Path(event_path))
    break
print(json.dumps(results, indent=2, sort_keys=True))
