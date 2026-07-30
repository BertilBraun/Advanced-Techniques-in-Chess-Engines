from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import UUID

from src.az.evaluation.models import EvaluationGameResult


class EvaluationResultRepository:
    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute():
            raise ValueError('Evaluation result directory must be absolute.')
        self._directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, evaluation_id: UUID, pair_index: int, game_in_pair: int) -> Path:
        return self._directory / f'{evaluation_id.hex}-pair-{pair_index:06d}-game-{game_in_pair}.json'

    def load(self, evaluation_id: UUID, pair_index: int, game_in_pair: int) -> EvaluationGameResult | None:
        path = self.path(evaluation_id, pair_index, game_in_pair)
        if not path.exists():
            return None
        return EvaluationGameResult.model_validate_json(path.read_bytes())

    def publish(self, result: EvaluationGameResult) -> EvaluationGameResult:
        path = self.path(result.evaluation_id, result.pair_index, result.game_in_pair)
        contents = result.model_dump_json(indent=2).encode() + b'\n'
        if path.exists():
            existing = path.read_bytes()
            if hashlib.sha256(existing).digest() != hashlib.sha256(contents).digest():
                raise ValueError('Evaluation game identity already has a different result.')
            return EvaluationGameResult.model_validate_json(existing)
        partial = path.with_suffix('.partial')
        if partial.exists():
            partial.unlink()
        with partial.open('xb') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        return result
