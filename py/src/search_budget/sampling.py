from __future__ import annotations

import hashlib
from fractions import Fraction

from pydantic import Field
from src.util.frozen_model import FrozenModel

DEFAULT_LABEL_SAMPLE_FRACTION = Fraction(1, 50)
LABEL_SHARD_SIZE = 512


class LabelPositionIdentity(FrozenModel):
    source_generation: int = Field(ge=0)
    game_identity: str = Field(min_length=1)
    ply: int = Field(ge=0)


def select_generation_sample(
    identities: tuple[LabelPositionIdentity, ...],
    run_seed: int,
    sample_fraction: Fraction = DEFAULT_LABEL_SAMPLE_FRACTION,
) -> tuple[LabelPositionIdentity, ...]:
    if not 0 <= run_seed < 2**64:
        raise ValueError('Run seed must fit in an unsigned 64-bit integer.')
    if not 0 <= sample_fraction <= 1:
        raise ValueError('Label sample fraction must be in [0, 1].')
    if len(set(identities)) != len(identities):
        raise ValueError('Generation position identities must be unique.')
    source_generations = {identity.source_generation for identity in identities}
    if len(source_generations) > 1:
        raise ValueError('One deterministic sample cannot span source generations.')

    sample_count = len(identities) * sample_fraction.numerator // sample_fraction.denominator
    ranked = sorted(identities, key=lambda identity: (_selection_digest(identity, run_seed), _identity_bytes(identity)))
    return tuple(ranked[:sample_count])


def partition_generation_sample(
    selected_identities: tuple[LabelPositionIdentity, ...],
) -> tuple[tuple[LabelPositionIdentity, ...], ...]:
    if len(set(selected_identities)) != len(selected_identities):
        raise ValueError('Selected generation position identities must be unique.')
    return tuple(
        selected_identities[start : start + LABEL_SHARD_SIZE]
        for start in range(0, len(selected_identities), LABEL_SHARD_SIZE)
    )


def _selection_digest(identity: LabelPositionIdentity, run_seed: int) -> bytes:
    return hashlib.sha256(run_seed.to_bytes(8, 'big') + _identity_bytes(identity)).digest()


def _identity_bytes(identity: LabelPositionIdentity) -> bytes:
    game_identity = identity.game_identity.encode('utf-8')
    return b''.join(
        (
            identity.source_generation.to_bytes(8, 'big'),
            identity.ply.to_bytes(8, 'big'),
            len(game_identity).to_bytes(8, 'big'),
            game_identity,
        )
    )
