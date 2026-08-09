from __future__ import annotations

from pathlib import Path

from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.configuration import TrainingLifecycleParams


class CheckpointRetention:
    def __init__(self, run_path: Path, configuration: TrainingLifecycleParams) -> None:
        self.run_path = run_path
        self.full_checkpoint_interval = configuration.credit.retained_checkpoint_interval_generations
        self.recent_inference_count = configuration.inference_retention.recent_checkpoint_count
        self.inference_milestone_interval = configuration.inference_retention.milestone_interval

    def apply(self, active_generation: int, required_inference_generations: tuple[int, ...]) -> None:
        generations = self._published_generations()
        if active_generation not in generations:
            raise ValueError(f'Active checkpoint manifest does not exist for generation {active_generation}.')

        full_checkpoints = {
            generation
            for generation in generations
            if generation == 0 or generation == active_generation or generation % self.full_checkpoint_interval == 0
        }
        first_recent_generation = max(0, active_generation - self.recent_inference_count + 1)
        inference_checkpoints = {
            generation
            for generation in generations
            if generation == 0
            or generation >= first_recent_generation
            or generation % self.inference_milestone_interval == 0
        }
        inference_checkpoints.update(required_inference_generations)

        for generation in generations:
            manifest = read_checkpoint_manifest(generation, self.run_path)
            if generation not in full_checkpoints:
                (self.run_path / manifest.model_path).unlink(missing_ok=True)
                (self.run_path / manifest.optimizer_path).unlink(missing_ok=True)
            if generation not in inference_checkpoints:
                (self.run_path / manifest.inference_model_path).unlink(missing_ok=True)

    def _published_generations(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(path.stem.removeprefix('checkpoint_'))
                for path in self.run_path.glob('checkpoint_*.json')
                if path.stem.removeprefix('checkpoint_').isdigit()
            )
        )
