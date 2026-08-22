import sys, torch, types
sys.path.insert(0,'.')
# stub the C++ module so network.py imports
m = types.ModuleType('AlphaZeroCpp'); 
class _S: pass
for n in ['GameSearchVisit']: setattr(m,n,_S)
sys.modules['AlphaZeroCpp']=m
from src.training.network import Network, AttentionNetworkParams, NetworkParams, Chess76PlaneDirectPolicyHeadConfiguration, GlobalPoolingResidualContext, ResidualContextPlacement
from src.games.representation import NetworkDimensions
D = NetworkDimensions(channels=29, rows=8, columns=8, actions=4864)
torch.manual_seed(0)
x = (torch.rand(64, D.channels, 8, 8) < 0.15).float()
x[:,22:] = torch.rand(64,7,1,1)
def probe(args, name, train_mode=True):
    net = Network(args, torch.device('cpu'), D)
    net.train(train_mode)
    with torch.no_grad():
        f = net._features(x)
        pl, vl = net.logit_forward(x)
    n = sum(p.numel() for p in net.parameters())
    ce = torch.nn.functional.cross_entropy(pl, torch.randint(0,4864,(64,)))
    print(f"{name}: params={n/1e6:.2f}M feat_rms={f.pow(2).mean().sqrt():.1f} policy_logit_std={pl.std():.1f} max|logit|={pl.abs().max():.0f} CE(rand target)={ce:.1f} value_logit_std={vl.std():.2f}")
ph = Chess76PlaneDirectPolicyHeadConfiguration()
for L,E,H,F,nm in [(6,96,3,192,'att 6x96 pilot'),(8,128,4,256,'att 8x128 1m'),(10,160,5,320,'att 10x160 2m'),(15,192,6,384,'att 15x192 4m5')]:
    probe(AttentionNetworkParams(policy_head=ph, num_layers=L, embedding_size=E, num_heads=H, feedforward_size=F), nm)
probe(NetworkParams(policy_head=ph, num_layers=12, hidden_size=144, residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK)), 'cnn 12x144 76plane')
probe(NetworkParams(policy_head=ph, num_layers=12, hidden_size=112, residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK)), 'cnn 12x112 76plane')
