# Phase A test node — provisioning note (2026-08-21)

Purpose: Phase A validation only (native build, NativeTests, full `py/test`, WP6 config resolution). Not a
benchmark or run node. Rented after two candidate nodes were rejected for dead CDN downlinks
(191.223.212.127: PyPI/GitHub ≈ 6–9 kB/s; 38.49.42.46: PyPI stalled at 0 B/s).

- Node: Vast.ai container `cae651504bf0`, SSH `root@80.59.54.98:10574`.
- GPU: 1× RTX 4070 12 GiB, driver 595.71.05 (initial probe reported 580.142 before provisioning), compute capability 8.9.
- CPU: AMD Ryzen 7 7700X, 16 effective CPUs. RAM 30 GiB. Disk 150 GiB overlay.
- `/workspace` is NOT volume-backed (`workspace_is_volume: false`): nothing on the node survives
  recycle/destroy. All evidence must be fetched before release.
- Network at provisioning time: PyPI 37.8 MB/s, PyTorch CDN 22.8 MB/s.
- Provisioned with `deployment/setup_remote.sh` from GitHub `phase-a` (locked
  `py/requirements-training.lock`, `--torch-backend cu126`): torch 2.12.1+cu126, CUDA runtime 12.6,
  cuDNN 91002, Python 3.12 venv at `/workspace/alphazero-engine-venv`.
- Engines: Stockfish UCI and KataGo CUDA JSON-analysis smoke checks passed during setup. Syzygy WDL 3-4-5
  at `/workspace/syzygy/wdl345`.
- Deviations found and applied during setup (candidates for `setup_remote.sh` fixes):
  1. `uv` needed `UV_HTTP_TIMEOUT=300` (default 30 s aborts on large wheels).
  2. The cmake configure lacks `-DBUILD_TESTING=ON`, so `NativeTests` is not built by default.
  3. `pytest` is not in the training lock; installed ad hoc into the venv for validation.
- Repository state under test: `phase-a` @ `7c146d79` (all four Phase A streams merged).
