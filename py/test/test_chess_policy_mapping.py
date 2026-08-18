import hashlib

from AlphaZeroCpp import ChessPosition, chess_policy_mapping, mirror_chess_action_id


EXPECTED_MAPPING_SHA256 = 'a09ae830eabaac367ae4cd91015d7d85f93127cd62187cc6bfcaf45aff7992af'


def _mapping_digest(indices: tuple[int, ...]) -> str:
    payload = b''.join(index.to_bytes(2, byteorder='little', signed=False) for index in indices)
    return hashlib.sha256(payload).hexdigest()


def test_chess_policy_mapping_is_stable_bounded_and_mirror_consistent() -> None:
    mapping = chess_policy_mapping()
    indices = mapping.action_plane_indices

    assert mapping.plane_count == 76
    assert len(indices) == 1880
    assert len(set(indices)) == 1880
    assert min(indices) >= 0
    assert max(indices) < mapping.plane_count * 64
    assert _mapping_digest(indices) == EXPECTED_MAPPING_SHA256

    for action_id, slot in enumerate(indices):
        mirrored_action_id = mirror_chess_action_id(action_id)
        mirrored_slot = indices[mirrored_action_id]
        origin_square = slot % 64
        mirrored_origin_square = mirrored_slot % 64
        assert mirrored_origin_square // 8 == origin_square // 8
        assert mirrored_origin_square % 8 == 7 - origin_square % 8
        assert mirror_chess_action_id(mirrored_action_id) == action_id


def test_white_and_black_straight_and_capture_promotions_share_canonical_planes() -> None:
    mapping = chess_policy_mapping()
    white = ChessPosition('r1b4k/1P6/8/8/8/8/8/7K w - - 0 1')
    black = ChessPosition('7k/8/8/8/8/8/1p6/R1B4K b - - 0 1')

    for promotion in ('q', 'r', 'b', 'n'):
        white_planes = tuple(
            mapping.action_plane_indices[white.action_id_from_uci(move)] // 64
            for move in (f'b7b8{promotion}', f'b7a8{promotion}', f'b7c8{promotion}')
        )
        black_planes = tuple(
            mapping.action_plane_indices[black.action_id_from_uci(move)] // 64
            for move in (f'b2b1{promotion}', f'b2a1{promotion}', f'b2c1{promotion}')
        )

        assert white_planes == black_planes
        assert all(64 <= plane < mapping.plane_count for plane in white_planes)
        assert (
            mapping.action_plane_indices[mirror_chess_action_id(white.action_id_from_uci(f'b7a8{promotion}'))] // 64
            == white_planes[2]
        )
