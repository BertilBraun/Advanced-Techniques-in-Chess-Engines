"""Executable verification of colour-symmetry of the REAL C++ chess encoders.

Uses ./harness2 (compiled from /tmp/az/head/cpp sources + Stockfish fork) as an oracle.
"""
import random, subprocess, sys, collections
import chess
import numpy as np

sys.path.insert(0, '/tmp/az/work/codeA/stub')  # AlphaZeroCpp stub (only used for mirror id; we compare to C++ anyway)
sys.path.insert(0, '/tmp/az/head/py')
from src.games.representation import decode_packed_planes, encode_packed_planes
from src.games.chess.contract import CHESS_STATE_CONTRACT

HARNESS = '/tmp/az/work/flip/harness2'
rng = random.Random(12345)


def random_positions(n_games, max_plies=120):
    out = []
    for _ in range(n_games):
        b = chess.Board()
        # random Chess960-free start; occasionally start from a position with many promotions
        plies = rng.randint(0, max_plies)
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            # bias towards promotions / captures sometimes to reach rich endgames
            promos = [m for m in moves if m.promotion]
            m = rng.choice(promos) if promos and rng.random() < 0.7 else rng.choice(moves)
            b.push(m)
            if b.is_game_over():
                break
        if not b.is_game_over():
            out.append(b.copy())
    return out


def ep_positions(n):
    """positions where the last move was a double pawn push and an en-passant capture is legal"""
    out = []
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        b = chess.Board()
        for _ in range(rng.randint(4, 60)):
            moves = list(b.legal_moves)
            if not moves: break
            dbl = [m for m in moves if b.piece_type_at(m.from_square) == chess.PAWN and abs(m.to_square - m.from_square) == 16]
            m = rng.choice(dbl) if dbl and rng.random() < 0.5 else rng.choice(moves)
            b.push(m)
            if b.is_game_over(): break
        if not b.is_game_over() and any(b.is_en_passant(m) for m in b.legal_moves):
            out.append(b.copy())
    return out


def handcrafted():
    fens = [
        'r3k2r/pppqppbp/2np1np1/8/2PP4/2N2NP1/PP2PPBP/R2QK2R b KQkq - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R b Qk - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1',
        'rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2',
        'rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3',
        '8/P7/8/8/8/8/7p/K6k b - - 0 1',
        '1n6/P7/8/8/8/8/1p6/K1N4k b - - 0 1',
        'k7/8/8/8/8/8/8/4K2R b K - 0 1',
        '4k3/8/8/8/8/8/8/R3K3 b Q - 0 1',
        'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3',
        'rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq f3 0 2',
        '8/8/8/8/k2Pp3/8/8/4K3 b - d3 0 1',
        '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    ]
    return [chess.Board(f) for f in fens]


def run_harness(fens):
    proc = subprocess.run([HARNESS], input='\n'.join(fens) + '\n', capture_output=True, text=True, check=True)
    results = {}
    cur = None
    for line in proc.stdout.splitlines():
        tag, _, rest = line.partition(' ')
        if tag == 'FEN':
            cur = {'fen': rest, 'moves': {}}
        elif tag == 'PLANES':
            cur['planes'] = np.array([int(x) for x in rest.split(',')], dtype=np.int8)
        elif tag == 'PACKED':
            cur['packed'] = bytes.fromhex(rest)
        elif tag == 'MOVE':
            uci, aid, dec, fsq, tsq, mir = rest.split()
            cur['moves'][uci] = (int(aid), dec, int(fsq), int(tsq), int(mir))
        elif tag == 'END':
            results[cur['fen']] = cur
    return results


def mirror_move_uci(m: chess.Move) -> str:
    mm = chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
    return mm.uci()


def main():
    boards = handcrafted() + random_positions(1500) + random_positions(1200, max_plies=24) + ep_positions(150)
    black = [b for b in boards if b.turn == chess.BLACK]
    white = [b for b in boards if b.turn == chess.WHITE]
    # Use both: for each position P (either colour) compare with P.mirror() (colour swap + rank flip)
    pairs = []
    for b in boards:
        bm = b.mirror()  # python-chess: flips ranks, swaps colours, swaps castling rights & ep square
        assert bm.turn != b.turn
        pairs.append((b, bm))
    fens = []
    for b, bm in pairs:
        fens.append(b.fen())
        fens.append(bm.fen())
    res = run_harness(fens)

    stats = collections.Counter()
    failures = []
    ids_seen = set()
    rep = CHESS_STATE_CONTRACT.representation
    for b, bm in pairs:
        r, rm = res[b.fen()], res[bm.fen()]
        stats['positions'] += 1
        stats['black_to_move' if b.turn == chess.BLACK else 'white_to_move'] += 1
        if b.has_castling_rights(chess.WHITE) or b.has_castling_rights(chess.BLACK):
            stats['with_castling_rights'] += 1
        if b.ep_square is not None:
            stats['with_ep_square'] += 1
        if b.is_check():
            stats['in_check'] += 1
        # (1) input planes identical between P and colour-mirrored P'
        if not np.array_equal(r['planes'], rm['planes']):
            diff = np.nonzero(r['planes'] != rm['planes'])[0]
            failures.append(('PLANES', b.fen(), bm.fen(), [(int(i) // 64, int(i) % 64) for i in diff[:10]]))
        # (1b) python decode of packed payload == C++ tensor
        dec = decode_packed_planes(rep.packed_planes.value(r["packed"]),
                                   rep.packed_planes, rep.binary_channels, rep.scalar_channels)
        if not np.array_equal(dec.reshape(-1), r['planes']):
            failures.append(('PACKED_DECODE', b.fen()))
        # (2) legal move set from stockfish == python-chess, ids equal under mirror
        legal_py = {m.uci() for m in b.legal_moves}
        legal_cpp = set(r['moves'])
        # stockfish castling uci: check ChessAction::toUci normalises to king-dest form
        if legal_py != legal_cpp:
            failures.append(('LEGALSET', b.fen(), sorted(legal_py ^ legal_cpp)))
            continue
        ids_p = set()
        ids_pm = set()
        for m in b.legal_moves:
            u = m.uci()
            um = mirror_move_uci(m)
            aid, dec_u, fsq, tsq, mir = r['moves'][u]
            if um not in rm['moves']:
                failures.append(('MIRROR_MOVE_MISSING', b.fen(), u, um))
                continue
            aid_m, dec_um, _, _, _ = rm['moves'][um]
            stats['moves'] += 1
            if m.promotion:
                stats['promotions'] += 1
                if m.promotion != chess.QUEEN:
                    stats['underpromotions'] += 1
            if b.is_castling(m):
                stats['castling_moves'] += 1
            if b.is_en_passant(m):
                stats['ep_captures'] += 1
            if aid != aid_m:
                failures.append(('ACTIONID', b.fen(), u, aid, bm.fen(), um, aid_m))
            if dec_u != u:
                failures.append(('DECODE_ROUNDTRIP', b.fen(), u, aid, dec_u))
            ids_p.add(aid)
            ids_pm.add(aid_m)
            # from-square of canonical id == canonical from square
            canon_from = fsq if b.turn == chess.WHITE else fsq ^ 56
            if aid % 64 != canon_from:
                failures.append(('FROMSQ', b.fen(), u, aid, canon_from))
            # the piece on the canonical from-square must be an OWN piece in the input planes (planes 0..5 = own pieces)
            own = r['planes'][:6 * 64].reshape(6, 64)[:, aid % 64].sum()
            if own != 1:
                failures.append(('OWN_PIECE_AT_FROM', b.fen(), u, aid, int(own)))
            ids_seen.add(aid)
        if ids_p != ids_pm:
            failures.append(('LEGAL_ID_SET', b.fen(), sorted(ids_p ^ ids_pm)))
        if len(ids_p) != len(list(b.legal_moves)):
            failures.append(('ID_COLLISION', b.fen()))

    print('STATS', dict(stats))
    print('distinct ids seen', len(ids_seen))
    print('FAILURES', len(failures))
    for f in failures[:40]:
        print('  ', f)

    # ---- mirror augmentation check (file mirror) ----
    # For each position P, build file-mirrored position Pf with python-chess (castling rights dropped -> we compare
    # only positions WITHOUT castling rights exactly; with castling rights we report the castling-plane behaviour).
    mir_stats = collections.Counter(); mir_fail = []
    sub = [b for b, _ in pairs if not b.has_castling_rights(chess.WHITE) and not b.has_castling_rights(chess.BLACK)]
    sub = sub[:800]
    def file_mirror(b: chess.Board) -> chess.Board:
        nb = chess.Board(None)
        for sq, pc in b.piece_map().items():
            nb.set_piece_at(chess.square(7 - chess.square_file(sq), chess.square_rank(sq)), pc)
        nb.turn = b.turn
        nb.ep_square = None if b.ep_square is None else chess.square(7 - chess.square_file(b.ep_square), chess.square_rank(b.ep_square))
        nb.halfmove_clock = b.halfmove_clock; nb.fullmove_number = b.fullmove_number
        return nb
    mirrored = [file_mirror(b) for b in sub]
    resf = run_harness([b.fen() for b in sub] + [m.fen() for m in mirrored])
    for b, bf in zip(sub, mirrored):
        r, rf = resf[b.fen()], resf[bf.fen()]
        mir_stats['positions'] += 1
        # input planes: python augmentation np.flip(axis=2) of P must equal encoding of file-mirrored position
        planes = r['planes'].reshape(29, 8, 8)
        aug = np.flip(planes, axis=2)
        if not np.array_equal(aug.reshape(-1), rf['planes']):
            mir_fail.append(('MIRROR_PLANES', b.fen()))
        # via the actual contract code path
        payload = CHESS_STATE_CONTRACT.packed_plane_layout.value(r['packed'])
        aug2 = CHESS_STATE_CONTRACT.transform_encoded_state(payload, 1)
        if bytes(aug2.payload) != rf['packed']:
            mir_fail.append(('MIRROR_PACKED', b.fen()))
        ids_f = {v[0] for v in rf['moves'].values()}
        ids_mapped = {v[4] for v in r['moves'].values()}
        mir_stats['moves'] += len(r['moves'])
        if ids_f != ids_mapped:
            mir_fail.append(('MIRROR_IDS', b.fen(), sorted(ids_f ^ ids_mapped)[:10]))
        # stub mirror == C++ mirror
        from AlphaZeroCpp import mirror_chess_action_id
        for u, v in r['moves'].items():
            if mirror_chess_action_id(v[0]) != v[4]:
                mir_fail.append(('STUB_MIRROR_MISMATCH', u, v[0]))
    print('MIRROR_AUG STATS', dict(mir_stats), 'FAILURES', len(mir_fail))
    for f in mir_fail[:20]:
        print('  ', f)

    # castling-plane behaviour under file mirror, one explicit example
    b = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1')
    r = run_harness([b.fen()])[b.fen()]
    planes = r['planes'].reshape(29, 8, 8)
    aug = np.flip(planes, axis=2)
    print('CASTLING PLANES example (own K-side only): orig planes12..15 sums',
          [int(planes[i].sum()) for i in (12, 13, 14, 15)], 'after file-mirror aug', [int(aug[i].sum()) for i in (12, 13, 14, 15)],
          'king square after aug', int(np.argmax(aug[5].reshape(-1))), 'e1g1 id', r['moves']['e1g1'][0], '-> mirrored id', r['moves']['e1g1'][4],
          '= plane', r['moves']['e1g1'][4] // 64, 'from', r['moves']['e1g1'][4] % 64)
    return 0 if not failures and not mir_fail else 1


if __name__ == '__main__':
    sys.exit(main())
