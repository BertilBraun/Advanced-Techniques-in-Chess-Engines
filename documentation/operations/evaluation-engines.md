# Stockfish and KataGo evaluation engines

This runbook provisions the external engines required by the accepted evaluation architecture. The checked-in
experiment templates use these explicit repository-relative paths:

| Artifact | Configured path | Protocol |
| --- | --- | --- |
| Stockfish | `engines/stockfish` | UCI through `python-chess` |
| KataGo | `engines/katago` | asynchronous JSON analysis |
| KataGo network | `engines/katago.bin.gz` | KataGo binary network format |
| KataGo configuration | `engines/katago-analysis.cfg` | analysis-engine configuration |

The paths are intentionally not discovered from `PATH`. Run manifests hash the executable and, for KataGo, the
network and configuration. The generated `engines/` directory is ignored by Git; do not commit binaries or neural
networks.

## Pinned artifacts

The defaults were verified against primary upstream sources on 2026-08-09.

| Artifact | Pin | Archive SHA-256 |
| --- | --- | --- |
| Stockfish portable Linux x86-64 | [Stockfish 18 `stockfish-ubuntu-x86-64.tar`](https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar) | `5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff` |
| KataGo portable CPU | [KataGo 1.17.1 Eigen Linux x86-64](https://github.com/lightvector/KataGo/releases/download/v1.17.1/katago-v1.17.1-eigen-linux-x64.zip) | `cca71fff39abd19bd9acfc17750025d4bb0ee6adbad99d7513a2c6401b0a7af3` |
| KataGo general network | [`b10c384h6nbttflrs.bin.gz`](https://github.com/lightvector/KataGo/releases/download/v1.17.1/b10c384h6nbttflrs.bin.gz) | `0ba27eced5180b3e3d0b898b280c541112989765e789d1eb6cd0d31b2b2c1229` |

Stockfish 18 is the latest stable Stockfish release; development builds are deliberately excluded. KataGo 1.17.2
only replaces TensorRT builds affected by its fixes, so 1.17.1 remains the current official release for Eigen,
OpenCL, and CUDA/cuDNN. The selected transformer network is an official 1.17.1 release asset and supports the 7x7
and 9x9 requests used here. Upstream also publishes a [9x9-only finetuned
network](https://katagotraining.org/extra_networks/), but it is not selected: using one pinned general network keeps
the initial external baseline and redistribution terms uniform across both board sizes. A later benchmark decision
may pin separate networks by changing the explicit experiment path and provenance.

The Stockfish archive is the baseline x86-64 build. On a confirmed AVX2 host, an explicitly selected official
alternative is
[`stockfish-ubuntu-x86-64-avx2.tar`](https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-avx2.tar),
SHA-256 `536c0c2c0cf06450df0bfb5e876ef0d3119950703a8f143627f990c7b5417964`.

## Local WSL installation

Use WSL 2 with an x86-64 Linux distribution. Install the small bootstrap prerequisites using the distribution's
package manager; for Ubuntu:

```bash
sudo apt-get update
sudo apt-get install --yes curl unzip tar coreutils
cd /mnt/c/path/to/repository
bash deployment/install_evaluation_engines.sh
bash deployment/smoke_evaluation_engines.sh
```

Installation refuses to overwrite an existing `engines/` directory. To reinstall, first move or remove that exact
directory after confirming it contains only generated upstream artifacts. `ARTIFACTS.sha256` and
`INSTALLATION.txt` inside it record what was fetched.

The smoke script verifies:

- Stockfish reports version 18, completes `uci`, and answers `isready`;
- KataGo starts with the pinned model and configuration;
- 7x7 and 9x9 Chinese-rules requests return nonempty weighted `moveInfos`;
- closing stdin lets KataGo drain work and exit cleanly, with a 120-second safety timeout.

After the native extension and Python dependencies are built, exercise the application clients from `py/`:

```bash
export EVALUATION_STOCKFISH_EXECUTABLE=../engines/stockfish
export EVALUATION_KATAGO_EXECUTABLE=../engines/katago
export EVALUATION_KATAGO_MODEL=../engines/katago.bin.gz
export EVALUATION_KATAGO_CONFIGURATION=../engines/katago-analysis.cfg
python -m pytest --import-mode=importlib test/integration/test_external_evaluation_engines.py -q
```

## Fresh compute node

[`deployment/setup_remote.sh`](../../deployment/setup_remote.sh) remains the single fresh-node bootstrap. It clones
the requested revision, creates the locked Python environment, builds the Release native extension, installs and
smokes the pinned evaluation engines, exports the four integration-test variables above plus
`ENGINE_SOURCE_REVISION`, and finally runs the supplied command.

The node must initially provide `git`, `cmake`, Python with `venv`, `curl`, `tar`, `unzip`, and `coreutils`. Example:

```bash
export ENGINE_REPOSITORY_REF=reviewed-branch-or-tag
bash deployment/setup_remote.sh \
  python py/train.py \
  --run-config py/configs/go-7x7-experiment-template.yaml \
  --expected-source-revision EXPECTED_FULL_COMMIT \
  --approval-file /absolute/path/to/approval.json
```

Production configuration and approval files remain explicit inputs. The templates still describe unconfirmed local
hardware and CPU training; they must be copied, reviewed, and updated for the rented offer before approval.

## Selecting a KataGo accelerator backend

KataGo officially supports Eigen CPU, OpenCL, CUDA/cuDNN, TensorRT, and Metal backends. Linux release binaries are
published separately. Eigen is the validated default because it does not assume a GPU model, driver, device count,
CUDA runtime, cuDNN, or TensorRT installation. It is suitable for provisioning and protocol smoke tests, not for
the final performance baseline.

Do not choose a GPU archive until the rented node is known. On that node:

1. Record `nvidia-smi`, driver version, GPU model/count, and visible devices.
2. Confirm the installed CUDA plus cuDNN or TensorRT versions against the exact version in the official KataGo
   asset name. Upstream recommends matching those versions rather than assuming ABI compatibility.
3. Select an official archive and its GitHub-published SHA-256. KataGo 1.17.1 CUDA/cuDNN or 1.17.2 TensorRT are the
   current choices for this pin family.
4. Provision with explicit inputs, for example:

```bash
export ENGINE_KATAGO_BACKEND=cuda12.8-cudnn9.8.0
export ENGINE_KATAGO_ARCHIVE_URL=https://github.com/lightvector/KataGo/releases/download/v1.17.1/katago-v1.17.1-cuda12.8-cudnn9.8.0-linux-x64.zip
export ENGINE_KATAGO_ARCHIVE_SHA256=458d226c2c8533600251bba3b2ee612d3aee0c796f592a2b53839a6a05b0826e
bash deployment/setup_remote.sh COMMAND ARGUMENTS
```

The installer requires URL and hash together for every non-default backend. It never chooses a GPU build by
probing the host. The application exposes only the evaluation job's assigned device through
`CUDA_VISIBLE_DEVICES`; the analysis configuration uses one model server and does not encode a physical device ID
or GPU count. The rented-node gate is therefore: confirm the backend starts, run the real client test, then measure
evaluation/training contention under the approved topology before freezing a benchmark configuration.

The checked-in analysis configuration is derived from KataGo's [1.17.1 analysis
example](https://github.com/lightvector/KataGo/blob/v1.17.1/cpp/configs/analysis_example.cfg). It permits concurrent
position analysis, caps NN buffers at 9x9, and leaves rules, komi, board size, and visit limits to every JSON request,
matching the application client. See the [official asynchronous analysis protocol](https://github.com/lightvector/KataGo/blob/v1.17.1/docs/Analysis_Engine.md).

## Evaluation openings and datasets

Engine installation does not generate openings or fixed evaluation datasets. That work must remain in run
preparation because each immutable manifest binds the exact experiment rules, representation, engine hashes,
search limits, random seed, builder source revision, and output hash. `python py/train.py ...` calls preparation
after hardware and approval validation and before the coordinator starts. Missing artifacts are generated;
matching artifacts are reused; mismatches fail instead of being overwritten.

The generated paths under `py/reference/` are ignored by Git. Preserve and back up the resulting data and manifests
with the run, but do not commit them. There is intentionally no install-time or implicit global dataset cache.

## Licensing and redistribution

- Stockfish is GPLv3. Running it locally imposes no source-distribution step, but redistributing its binary requires
  the license and corresponding source obligations described in [`Copying.txt`](https://github.com/official-stockfish/Stockfish/blob/sf_18/Copying.txt).
  The installer retains the official release tree, including `Copying.txt` and source, below
  `engines/upstream/stockfish-18/`. Preserve that tree or otherwise satisfy GPLv3 when exporting an image or bundle.
- KataGo's own code is permissively licensed, while its release contains separately licensed third-party
  dependencies listed in the [KataGo 1.17.1 license](https://github.com/lightvector/KataGo/blob/v1.17.1/LICENSE).
  Preserve the release's license/readme files when redistributing the executable.
- Official KataGo network files use the [KataGo network license](https://katagotraining.org/network_license/), which
  requires the copyright and permission notice with copies or substantial portions. Confirm the selected network
  is covered before replacing the default, especially for third-party or extra networks.

## Troubleshooting

- `Illegal instruction`: use the portable Stockfish or Eigen build, or select an archive matching the CPU features.
- Missing CUDA, cuDNN, TensorRT, or OpenCL libraries: the chosen KataGo binary does not match the node runtime. Use
  Eigen to validate protocols, then install the exact accelerator runtime or select a matching official archive.
- KataGo reports a model-version error: the network is incompatible with the executable; restore the pinned pair.
- KataGo returns errors for rules or coordinates: confirm the application is sending Chinese rules, half-integer
  komi, complete move history, and board sizes no larger than the configuration's 9x9 buffer.
- Existing immutable datasets fail validation after an engine change: use new versioned output paths and regenerate
  during approved run preparation. Do not overwrite evidence from the old engine pin.
