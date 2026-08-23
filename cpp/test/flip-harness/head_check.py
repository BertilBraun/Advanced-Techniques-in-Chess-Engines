"""Spatial-mapping check of the 76-plane direct policy head (conv + attention backbones).

Feeds one-hot inputs (plane p, square s) and checks that ONLY logits q*64+s (q=0..75) respond
when the backbone is made spatially local (1x1-equivalent), i.e. head flatten order == input square index.
Then ties this to the C++ action ids from harness2: id % 64 == canonical from-square == input square index.
"""

from __future__ import annotations

import os
import sys

for extra_path in reversed(
    os.environ.get('FLIP_HARNESS_PYTHON_PATHS', '/tmp/az/work/codeA/stub:/tmp/az/head/py').split(os.pathsep)
):
    if extra_path:
        sys.path.insert(0, extra_path)
import numpy as np
import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.network import (
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    Network,
    NetworkParams,
)

torch.manual_seed(0)
dev = torch.device('cpu')
dims = CHESS_NETWORK_DIMENSIONS
print('dims', dims)
if dims.actions != 76 * 64:
    print('skipped: the 76-plane head check only applies to the policy-plane action encoding')
    sys.exit(0)


def test(net, name):
    net.eval()
    fails = 0
    with torch.no_grad():
        base = net.logit_forward(torch.zeros(1, dims.channels, 8, 8))[0]
        for p in (0, 5, 11, 19, 28):
            for s in range(64):
                x = torch.zeros(1, dims.channels, 8, 8)
                x.view(1, -1)[0, p * 64 + s] = 1.0  # flat index = plane*64 + square, as written by C++ writeTensorInto
                logits = net.logit_forward(x)[0]
                delta = (logits - base).abs().view(76, 64)
                responding = set(torch.nonzero(delta.sum(0) > 1e-6).flatten().tolist())
                if responding != {s}:
                    fails += 1
                    if fails < 5:
                        print('  FAIL', name, 'plane', p, 'square', s, 'responding squares', sorted(responding)[:8])
    print(name, 'one-hot spatial test: failures', fails, 'of', 5 * 64)
    return fails


# conv backbone: make it spatially local by zeroing all off-centre 3x3 weights
conv = Network(
    NetworkParams(policy_head=Chess76PlaneDirectPolicyHeadConfiguration(), num_layers=2, hidden_size=16), dev, dims
)
with torch.no_grad():
    for m in conv.modules():
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3):
            w = m.weight.clone()
            w[:, :, :, :] = 0
            w[:, :, 1, 1] = m.weight[:, :, 1, 1]
            m.weight.copy_(w)
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
f1 = test(conv, 'conv(12x16, centre-tap only)')

# attention backbone: make attention spatially local by zeroing query/key so softmax is uniform? No - uniform attention mixes.
# Instead zero the value projection (attention contributes nothing) so only the per-token MLP acts -> local.
att = Network(
    AttentionNetworkParams(
        policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
        num_layers=2,
        embedding_size=16,
        num_heads=2,
        feedforward_size=32,
    ),
    dev,
    dims,
)
with torch.no_grad():
    for name, p in att.named_parameters():
        if 'query_key_value_projection' in name:
            # zero the value third of the projection output
            out = p.shape[0]
            p[2 * out // 3 :] = 0
f2 = test(att, 'attention(2x16, value-proj zeroed)')

# Now tie to C++ ids: for a black-to-move position, id%64 must equal the square index where the own piece sits in input planes.
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_checks import run_harness

b = chess.Board('r3k2r/pppqppbp/2np1np1/8/2PP4/2N2NP1/PP2PPBP/R2QK2R b KQkq - 0 1')
r = run_harness([b.fen()])[b.fen()]
planes = r['planes'].reshape(29, 64)
bad = 0
for u, (aid, dec, fsq, tsq, mir) in r['moves'].items():
    s = aid % 64
    own_piece_planes = np.nonzero(planes[:6, s])[0]
    if len(own_piece_planes) != 1 or s != (fsq ^ 56):
        bad += 1
print('black-to-move id%64 == canonical from-square with own piece present: mismatches', bad, 'of', len(r['moves']))
print(
    'example: black move e8g8 (castle) ->',
    r['moves']['e8g8'][0],
    'plane',
    r['moves']['e8g8'][0] // 64,
    'from sq',
    r['moves']['e8g8'][0] % 64,
    '(e1=4 canonical)',
)
sys.exit(1 if (f1 or f2 or bad) else 0)
