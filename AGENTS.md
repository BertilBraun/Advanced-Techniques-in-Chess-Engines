# Agent instructions

## Rework authority

Read `documentation/operations/experiment-platform.md` completely before ordinary experiment configuration, fresh
node provisioning, queue operation, monitoring, or result export. Read
`documentation/architecture/platform-rework.md` before architectural changes or when investigating historical
platform decisions. For architectural Python runtime work, also read
`documentation/architecture/python-runtime-rework.md`. Preserve unrelated changes and make feature-sized commits
after relevant validation.

At every handoff, report completed work, outstanding phase work, commits,
validation results, changes needing special review, and unresolved decisions.
Only the user accepts a phase or authorizes another phase.

## Remote execution

Use `deployment/setup_remote.sh` for a fresh compute node. It clones the
requested revision, installs the locked training dependencies, builds the
Release C++ extension, exports `ENGINE_SOURCE_REVISION`, and starts the supplied
runner command. Keep production run configuration and approval files explicit.

### KataGo CUDA requirement

KataGo must always use a CUDA/cuDNN release on compute nodes. The engine
installer defaults to the pinned `cuda12.8-cudnn9.8.0` KataGo 1.17.1 archive
and rejects CPU, OpenCL, and TensorRT builds. Never select the Eigen fallback.

Before provisioning a new node, record `nvidia-smi`, the driver version, GPU
model/count, visible devices, and the locked PyTorch CUDA/cuDNN runtime. The
`CUDA Version` reported by `nvidia-smi` is the driver's maximum supported CUDA
version, not necessarily the user-space runtime used by PyTorch or KataGo.

Omit all KataGo archive variables to use the checked-in CUDA default. To select
a different official CUDA/cuDNN archive, set all three variables together; a
partial override is rejected:

```bash
export ENGINE_KATAGO_BACKEND=cuda-version-cudnn-version
export ENGINE_KATAGO_ARCHIVE_URL=OFFICIAL_KATAGO_ASSET_URL
export ENGINE_KATAGO_ARCHIVE_SHA256=OFFICIAL_ASSET_SHA256
```

`setup_remote.sh` exposes the locked PyTorch wheel's NVIDIA libraries while it
installs and smokes the engines. Do not bypass its engine smoke. Provisioning
is successful only if `engines/INSTALLATION.txt` records a CUDA backend,
`katago version` reports `Using CUDA backend`, and the 7x7 and 9x9 analysis
smokes pass on an assigned GPU.

### Current Vast screening node

The current rented node is Vast instance `47400225` on machine `112617`, host `115630`: eight RTX 3060 12-GiB GPUs,
64 logical AMD EPYC 7452 CPUs, approximately 188 GiB RAM, and a 150-GB ephemeral root disk at $0.4608888889/hour.
The cgroup exposes CPUs `0-63` with a 61.44-CPU quota; an idle audit found no hidden host load, and pinned SHA-256
throughput was uniform at 1.61-1.62 GB/s. The local disk measured 1.4 GB/s write and 3.4 GB/s direct read; the offer
advertised 843-Mbps down and 1289-Mbps up networking with 94.7% reliability. The four 16-CPU, 44-GiB production
slots use GPU pairs `[0,1]`, `[2,3]`, `[4,5]`, and `[6,7]`. Pairs `[2,3]`, `[4,5]`, and `[6,7]` are NUMA-local;
`[0,1]` crosses NUMA domains because GPU 0 is the only GPU on NUMA node 3. Connect from Windows with the dedicated
Vast key:

```powershell
ssh -i C:\Users\berti\.ssh\codex_vast_ed25519 -p 39529 root@1.193.139.91 -L 8080:localhost:8080
```

The image is `vastai/pytorch:cuda-13.0.3-auto`; the locked training environment uses Python 3.12.3, PyTorch
2.12.1+cu126, CUDA 12.6, and cuDNN 9.10.2. The host driver is 580.126.20 and reports CUDA 13.0 as its maximum
supported driver API. Use a separate `-L 6006:localhost:16006` forward for the node's TensorBoard service.

The private key is local-only and must never be copied into the repository or
onto the node. Read `/etc/vast-agents-guide.md` completely before changing the
instance. The node filesystem is ephemeral; copy required run evidence off the
node before it is destroyed or recycled.

## Pytest

Run tests from `py`:

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Always retain `--import-mode=importlib`; otherwise the repository's `py` package
can cause `No module named 'py.test'` during collection.
