from __future__ import annotations

import hashlib
from fractions import Fraction

from pydantic import Field
from src.util.frozen_model import FrozenModel


class AuditPositionIdentity(FrozenModel):
    source_generation: int = Field(ge=0)
    game_identity: str = Field(min_length=1)
    ply: int = Field(ge=0)


def is_audit_position(identity: AuditPositionIdentity, run_seed: int, audit_fraction: Fraction) -> bool:
    """Deterministic per-position audit selection: the identity digest falls in the audit band."""
    if not 0 <= run_seed < 2**64:
        raise ValueError('Run seed must fit in an unsigned 64-bit integer.')
    if not 0 <= audit_fraction <= 1:
        raise ValueError('The audit fraction must lie in [0, 1].')
    digest = hashlib.sha256(run_seed.to_bytes(8, 'big') + _identity_bytes(identity)).digest()
    rank = int.from_bytes(digest[:8], 'big')
    return rank * audit_fraction.denominator < audit_fraction.numerator * 2**64


def _identity_bytes(identity: AuditPositionIdentity) -> bytes:
    game_identity = identity.game_identity.encode('utf-8')
    return b''.join(
        (
            identity.source_generation.to_bytes(8, 'big'),
            identity.ply.to_bytes(8, 'big'),
            len(game_identity).to_bytes(8, 'big'),
            game_identity,
        )
    )
