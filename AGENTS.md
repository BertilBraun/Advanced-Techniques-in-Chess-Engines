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

### Current node

Node identity, hardware facts and connection details live in exactly one place:
`documentation/operations/current-node.md`, updated on every rental and release.
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
