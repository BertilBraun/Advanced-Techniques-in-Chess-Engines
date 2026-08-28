from __future__ import annotations

from pydantic import Field
from src.self_play.completed_game import GameIdentity
from src.util.frozen_model import FrozenModel


class ReplayLabelGameLocator(FrozenModel):
    identity: GameIdentity
    action_ids: tuple[int, ...]
    observation_plies: tuple[int, ...] = Field(min_length=1)
    first_absolute_replay_row: int = Field(ge=0)

    def model_post_init(self, __context: object) -> None:
        if self.observation_plies != tuple(sorted(set(self.observation_plies))):
            raise ValueError('Replay label observations must use unique increasing plies.')
        if any(ply > len(self.action_ids) for ply in self.observation_plies):
            raise ValueError('Replay label observations cannot follow the game trajectory.')


class ReplayLabelCohortShard(FrozenModel):
    shard_identity: str = Field(pattern=r'^[0-9a-f]{64}$')
    games: tuple[ReplayLabelGameLocator, ...] = Field(min_length=1)

    def model_post_init(self, __context: object) -> None:
        identities = tuple(game.identity.archive_key for game in self.games)
        if len(set(identities)) != len(identities):
            raise ValueError('Replay label cohort game identities must be unique within a shard.')
