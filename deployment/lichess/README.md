# Running the AlphaZero engine as a Lichess bot

This directory contains the runtime glue for one standard-chess Lichess BOT account. The deployment does not expose a web server and Lichess never connects inbound to the Vast instance. The instance keeps outbound HTTPS streams open to Lichess.

## Short version

Rent an Ubuntu 22.04 or newer CUDA/PyTorch development instance with SSH access, connect as root, and run:

```text
curl -fsSL https://raw.githubusercontent.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/master/deployment/lichess/setup_vast.sh | bash
bash /workspace/alphazero-engine/deployment/lichess/smoke_vast.sh

read -rsp "Lichess bot token: " LICHESS_BOT_TOKEN; echo
export LICHESS_BOT_TOKEN
bash /workspace/alphazero-engine/deployment/lichess/run_vast_bot.sh
```

The first command clones the latest `master`, installs dependencies, compiles `AlphaZeroCpp`, runs tests, and installs `lichess-bot`. The smoke test downloads the default public TorchScript model and verifies CUDA plus a legal UCI move. The final command runs the authenticated bot in the foreground. No Docker image build, inbound Lichess port, JSON token file, or `tmux` session is required.

## SSH-first Vast workflow

### 1. Rent a development instance

Choose a Vast recommended Ubuntu 22.04 or newer CUDA/PyTorch **development** template with SSH launch mode; the setup uses the distribution's Python 3 packages and requires Python 3.10 or newer. For an RTX 50-series GPU, Vast currently recommends its `[Automatic]` template so CUDA 12.8 and a compatible PyTorch are selected. Allocate about 50 GB disk, one GPU, 4+ CPU cores, and 8–16 GB system RAM. Prefer on-demand and a reliable/verified host for the first run.

SSH launch mode gives an ordinary interactive shell. Nothing in this workflow automatically starts the chess bot.

Connect using the SSH command shown by Vast, then confirm:

```text
nvidia-smi
df -h
free -h
```

### 2. Run the standalone setup script

The setup defaults to the latest commit on `origin/master`. Download the current script and let it clone the repository:

```text
curl -fsSL https://raw.githubusercontent.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/master/deployment/lichess/setup_vast.sh | bash
```

This deliberately executes the current script directly from GitHub without saving a temporary copy. Inspect or download it first instead if you want to review the exact script before execution.

Alternatively, clone manually and execute the checked-out script:

```text
git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git \
  /workspace/alphazero-engine
bash /workspace/alphazero-engine/deployment/lichess/setup_vast.sh
```

Set `ENGINE_REVISION` only when intentionally reproducing an older run or testing a branch, tag, or commit. With the default `master`, the script fetches the remote and checks out the current `origin/master` commit in detached-HEAD mode. The exact resolved commit is still recorded for later reporting.

`setup_vast.sh` is idempotent where practical and refuses to overwrite a non-Git directory. It:

1. checks `nvidia-smi`;
2. installs the compiler, CMake, Ninja, the distribution's Python 3 development packages, and Git;
3. clones/checks out the latest `origin/master`, or `ENGINE_REVISION` when supplied;
4. creates `/workspace/alphazero-venv` and installs the hash-locked CUDA dependencies;
5. asserts PyTorch can see CUDA and prints the GPU;
6. compiles `AlphaZeroCpp` with `-march=native` on the rented CPU;
7. runs the native CTest suite and focused Python deployment/UCI tests;
8. installs `/usr/local/bin/alphazero-uci` and clones the pinned official `lichess-bot`.

The resolved revisions and paths are recorded in `/data/alphazero-install.txt`. Re-running the setup preserves local edits: Git checkout or compilation will fail visibly instead of resetting the tree.

Run it from the default root SSH account. Set `ENGINE_BUILD_JOBS` lower than the machine's CPU count if compilation approaches the RAM limit, for example `export ENGINE_BUILD_JOBS=4`.

Default locations can be overridden before setup:

```text
export ENGINE_REPOSITORY_ROOT=/workspace/my-engine-checkout
export ENGINE_VENV_ROOT=/workspace/my-engine-venv
export ENGINE_BUILD_ROOT=/workspace/my-engine-build
export LICHESS_BOT_ROOT=/workspace/my-lichess-bot
```

### 3. Download the model and perform a real UCI/GPU smoke test

By default, the smoke script downloads the public model `BertilBraun/alphazero-chess/latest.jit.pt` from the repository's current default revision:

```text
bash /workspace/alphazero-engine/deployment/lichess/smoke_vast.sh
```

`latest.jit.pt` is the TorchScript artifact loaded by the C++ engine. The repository's `latest.pt` is a Python training/state-dictionary checkpoint and is not needed by the UCI bot.

For a private or different Hugging Face model, enter the token without putting it in shell history and override only what differs:

```text
read -rsp "Hugging Face read token: " HF_TOKEN; echo
export HF_TOKEN
export HF_MODEL_REPO=OWNER/REPOSITORY
export HF_MODEL_FILE=path/to/model.jit.pt

bash /workspace/alphazero-engine/deployment/lichess/smoke_vast.sh
```

`HF_MODEL_REVISION` is optional and may be a branch, tag, or commit. When it is omitted, the downloader resolves the repository's current default revision to an exact commit before downloading and writes that commit to `/data/model-source.txt`. For a formal repeatable evaluation, preserve that file or set `HF_MODEL_REVISION` explicitly. For a public model, omit `HF_TOKEN`. If the model was copied by SCP instead, use:

```text
export MODEL_PATH=/data/models/model.jit.pt
bash /workspace/alphazero-engine/deployment/lichess/smoke_vast.sh
```

The smoke script checks CUDA twice (`nvidia-smi` and PyTorch), resolves and downloads the selected model if needed, starts the real UCI process, sends `uci`, `isready`, a two-ply position, and `go movetime 1000`, then verifies that exactly one returned `bestmove` is legal in that position. This exercises model loading, native inference, search, UCI transport, and GPU availability without contacting Lichess.

This smoke process exits. UCI is stdin/stdout IPC, not a daemon listening on a port, so there is no useful standalone UCI service to leave running while idle.

### 4. Start the public bot manually

Create and upgrade the BOT account as described below. Then enter the bearer token without saving it to disk:

```text
read -rsp "Lichess bot token: " LICHESS_BOT_TOKEN; echo
export LICHESS_BOT_TOKEN
```

Keep any model overrides from the smoke step, or export `MODEL_PATH`, then run directly in the persistent Vast shell:

```text
bash /workspace/alphazero-engine/deployment/lichess/run_vast_bot.sh
```

The script downloads the model only if `MODEL_PATH` is absent, copies the selected YAML to `/data/lichess-bot-config.yml`, adjusts `working_dir` for the actual checkout, prints every non-secret runtime path, and starts `lichess-bot.py` in the foreground. The foreground output and `/data/logs/lichess-bot.log` make engine startup and API failures visible. Stop gracefully with one `Ctrl-C`; because `quit_after_all_games_finish` is enabled, active games are allowed to finish. A second `Ctrl-C` forces termination.

`tmux` is not required. Its main purpose here would be keeping a process alive and reattachable when an SSH terminal disconnects. If the Vast shell itself remains alive, it adds no necessary protection. Install/use `tmux` or a service supervisor only when the selected access method does terminate foreground processes on disconnect or when reattachment is useful.

### 5. Edit and rerun

The default source configuration is:

```text
/workspace/alphazero-engine/deployment/lichess/config.yml
```

For machine-local experiments, copy it and point the runner to the copy:

```text
cp /workspace/alphazero-engine/deployment/lichess/config.yml /data/my-bot-config.yml
export LICHESS_BOT_CONFIG_SOURCE=/data/my-bot-config.yml
nano /data/my-bot-config.yml
bash /workspace/alphazero-engine/deployment/lichess/run_vast_bot.sh
```

The runner always writes a derived `/data/lichess-bot-config.yml` with the correct checkout working directory. It does not mutate the source config. This makes it easy to diff the experiment config against Git while keeping machine-specific edits under `/data`.

## What starts what

There are three processes/components with different jobs:

```text
Lichess API
    ^ outbound HTTPS/NDJSON
    |
lichess-bot.py                         account, challenges, games, PGNs
    | launches a UCI engine process
    v
/usr/local/bin/alphazero-uci          small shell wrapper
    | execs Python with model arguments
    v
python -m src.uci                     UCI stdin/stdout server
    | imports
    v
AlphaZeroCpp.so + TorchScript model   native board/MCTS and GPU inference
```

`lichess-bot.py` is the public-bot supervisor. It authenticates to Lichess, maintains the event streams, accepts or creates challenges according to `config.yml`, and saves PGNs. When a game starts, it reads the `engine` section and launches `/usr/local/bin/alphazero-uci`.

`alphazero-uci` is not another network service. It changes to `/workspace/py` and runs:

```text
python3.10 -m src.uci --model "$MODEL_PATH" ...
```

That Python module owns the production `src.eval.InteractiveEngine` for the UCI process. The facade constructs the native indexed search tree and direct inference pipeline, with two CUDA model workers and pipelined batches of 64 by default. It loads `AlphaZeroCpp.so` and the TorchScript model once, then answers the `uci`, `position`, `go movetime`, `stop`, and other UCI commands sent by `lichess-bot`. Timed UCI searches run the optimized native search in one-second slices so `stop` is observed between slices without returning to the obsolete `EvalMCTS` adapter. With `challenge.concurrency: 1`, one game uses one UCI process serially. A new game may cause `lichess-bot` to launch a new process, so model and worker lifetime currently spans one game process, not the entire life of the public-bot supervisor.

`SearchMode` is a UCI option advertised by our server. The config causes `lichess-bot` to send `setoption name SearchMode value mcts`. `mcts` chooses the most-visited root move after search. `policy` skips tree search and chooses the highest network prior; it is useful for diagnostics but is not the recommended strength measurement.

The launcher accepts optional performance overrides through `ENGINE_PARALLEL_SEARCHES`, `ENGINE_INFERENCE_WORKERS`, `ENGINE_OUTSTANDING_BATCHES_PER_WORKER`, and `ENGINE_MAXIMUM_BATCH_SIZE`. The defaults match the integrated RTX 3060 benchmarks. Leave them unchanged for the first deployment.

## Runtime paths

The setup script creates these paths on the rented instance:

| Path | Purpose | Created when |
|---|---|---|
| `/workspace/alphazero-engine` | Git checkout and compiled `AlphaZeroCpp.so` | Setup |
| `/workspace/alphazero-venv` | Python environment | Setup |
| `/workspace/lichess-bot` | Commit-pinned official client | Setup |
| `/usr/local/bin/alphazero-uci` | Installed executable engine wrapper | Setup |
| `/data/models` | Downloaded model | Smoke test or bot startup |
| `/data/pgn` | One PGN per completed game | Runtime |
| `/data/logs` | Persistent `lichess-bot` log | Runtime |

The runner derives `engine.working_dir` from the actual engine checkout. `engine.dir` plus `engine.name` identifies the executable that `lichess-bot` launches.

Vast instance storage survives a stop but is destroyed when the instance is destroyed. Export `/data/pgn` and `/data/logs` first.

## What is compiled

`setup_vast.sh` compiles the native `AlphaZeroCpp` Python extension directly on the rented node. CMake fetches pinned revisions of the adapted Stockfish board code, pybind11, and nlohmann/json. Compiling on the target allows `-march=native` safely and leaves the build tree available for inspection, incremental rebuilds, `ctest`, or debugger work.

After SSH setup completes, no compilation occurs during normal bot startup:

- `lichess-bot` is Python;
- the UCI transport and wrappers are Python/shell;
- the model is already TorchScript;
- the native extension has already been compiled on the node.

## Hardware to rent

Start conservatively with:

- one NVIDIA CUDA GPU; no multi-GPU instance is needed for one serial game;
- at least 8 GB VRAM if the actual TorchScript model is known to fit, otherwise 16 GB is the safer first rental;
- 4 CPU cores and 8–16 GB system RAM;
- 40–50 GB container disk for CUDA/PyTorch, build artifacts, the model, and logs;
- reliable outbound internet access to `github.com`, `lichess.org`, and Hugging Face;
- an on-demand rather than interruptible rental for the first public campaign.

An RTX 3090/4090, A10-class GPU, or comparable datacenter GPU is ample for one game if the model fits. A cheaper T4 may be adequate but should be benchmarked because fixed 10-second moves can hide throughput differences while affecting search counts. Do not rent several GPUs unless the interactive engine is deliberately changed to use them.

The host NVIDIA driver and development template must support CUDA 12.8. Verify after launch:

```text
nvidia-smi
python3.10 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the model's true VRAM requirement is unknown, run a short casual game first and inspect `nvidia-smi`. Do not infer fit solely from the model file size because inference activations and caches also consume memory.

## Authentication: no JSON token file

Lichess uses one bearer-token string, not a JSON credential file.

### Create and upgrade the BOT account

1. Create a dedicated Lichess account that has never played a game.
2. While logged into that account, open `https://lichess.org/account/oauth/token/create?scopes[]=bot:play`.
3. Create a token with `bot:play`. Copy it immediately.
4. Upgrade the unused account once from a trusted local machine:

```text
curl -X POST https://lichess.org/api/bot/account/upgrade \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d ""
```

5. Confirm the account profile shows the `BOT` title. Conversion is irreversible.

The deployment intentionally does not upgrade accounts automatically. A wrong token must never convert an unintended account.

### Pass secrets to Vast

Set these as runtime environment variables in the Vast template/instance configuration:

```text
LICHESS_BOT_TOKEN=the_plain_token_string
HF_TOKEN=optional_read_only_hugging_face_token
```

The model defaults are `HF_MODEL_REPO=BertilBraun/alphazero-chess` and `HF_MODEL_FILE=latest.jit.pt`; `HF_MODEL_REVISION` defaults to the repository's latest default revision. Override these variables only for another artifact. `HF_TOKEN` is required only for a private/gated model. The Lichess token is required. Do not put either token in `config.yml`, Git, or an on-start command. The placeholder token in `config.yml` is never used because current `lichess-bot` replaces it with `LICHESS_BOT_TOKEN`.

For direct challenges, `bot:play` is sufficient. Joining or creating an Arena through the tournament API additionally requires `tournament:write`. If tournament participation is planned, create a separate appropriately scoped token or rotate the runtime token; do not add that scope to a direct-match token without a reason.

## Startup checks

Expected startup sequence in `/data/logs/lichess-bot.log`:

1. model download/verification completes;
2. account profile is fetched;
3. engine configuration check launches the UCI wrapper successfully;
4. the client reports that it is connected and awaiting challenges.

Useful checks over SSH:

```text
test -f /data/models/latest.jit.pt
cat /data/model-source.txt
ls /workspace/py/AlphaZeroCpp*.so
tail -f /data/logs/lichess-bot.log
```

The compiled filename may contain a Python ABI suffix, such as `AlphaZeroCpp.cpython-310-x86_64-linux-gnu.so`; use `ls /workspace/py/AlphaZeroCpp*.so` if the exact name differs.

## Understanding `config.yml`

### Engine

| Setting | Meaning |
|---|---|
| `dir`, `name` | Resolve `/usr/local/bin/alphazero-uci`. |
| `working_dir` | Start the child in `/workspace/py`. |
| `protocol: uci` | Communicate through standard UCI stdin/stdout. |
| `ponder: false` | Do not search on the opponent's clock. |
| `debug: false` | Do not log every UCI message in normal operation. |
| `uci_options.SearchMode: mcts` | Use visit-count MCTS selection rather than network-policy-only selection. |
| `go_commands.movetime: 10000` | Always send `go movetime 10000`, i.e. ten seconds of search per move. |
| book/tablebase/online sections | Disabled so the measured move comes from this model/search only. |
| draw/resign settings | Disabled so early heuristic resignation/draw policy cannot bias calibration. |

The Lichess clock still applies. Ten seconds is an engine compute limit, not a Lichess time-control type.

### Incoming challenges

`challenge` controls which challenges sent **to this account** are accepted.

| Setting | Meaning |
|---|---|
| `concurrency: 1` | Play at most one game simultaneously. This matches the serial engine contract and one GPU worker. |
| `accept_bot: true` | Bot challengers are eligible. |
| `only_bot: true` | Human challenges are declined. Change to `false` to allow humans, while retaining `accept_bot: true`. |
| `min_base`, `max_base` | Allowed initial clock, in seconds. Both 600 means accept only games starting with 10 minutes. |
| `min_increment`, `max_increment` | Allowed increment added after each move, in seconds. Both 10 means accept only `+10`. |
| `variants: [standard]` | Reject Chess960 and variants. |
| `time_controls: [rapid]` | Accept only challenges Lichess classifies as rapid. |
| `modes: [casual]` | Accept unrated challenges only. Change this to `[rated]` for the registered campaign after the shakeout passes. |

Thus the checked-in filter accepts one casual standard bot-vs-bot game at a time at exactly `10+10`. It does not mean the instance is continuously available: the account is online only while `lichess-bot.py` is running. `only_bot: true` merely prevents humans from consuming the single slot during the bot evaluation window.

### Matchmaking is not a tournament

`matchmaking` is an optional `lichess-bot` feature that automatically challenges individual online bots while this bot is idle. It does not create a bracket, Swiss, Arena, opening schedule, or fixed round robin.

The checked-in config uses `allow_matchmaking: false`, so it never sends unsolicited challenges. Other fields describe what would happen if enabled:

- wait 30 idle minutes before looking for an opponent;
- challenge at `600+10`, standard chess, casual mode;
- consider opponent ratings from 600 to 4000.

For a limited automatic discovery run, edit the config before starting the bot:

```yaml
matchmaking:
  allow_matchmaking: true
  challenge_timeout: 5
  challenge_initial_time: [600]
  challenge_increment: [10]
  challenge_variant: "standard"
  challenge_mode: "casual"
  opponent_min_rating: 1600
  opponent_max_rating: 2200
  opponent_rating_difference: 250
```

Use this only after a casual smoke test. Automatic matchmaking cannot guarantee balanced colors, openings, opponent consent, or exactly 100 completed games. For the formal campaign, keep it disabled and issue a pre-registered list of direct challenges with requested colors.

### Correspondence and operational settings

The `correspondence` section matters only if correspondence challenges are allowed; the current `time_controls` list does not allow them. It remains explicit so accidental future enablement has bounded behavior.

- `pgn_directory` and grouping save one audited PGN per game under `/data/pgn`.
- `quit_after_all_games_finish: true` lets the first interrupt stop new work and finish active games.
- `abort_time: 30` controls inactivity abort behavior before substantive play.
- `move_overhead: 2000` reserves two seconds for communication in clock calculations; fixed `movetime` is still 10 seconds.
- `rate_limiting_delay: 100` adds a small delay after moves.
- `max_takebacks_accepted: 0` makes rated/casual results reproducible.

## Direct match, matchmaking, and Arena options

### Controlled direct-match campaign — recommended

Run the supervisor with matchmaking disabled. Identify willing bot opponents, then create challenges through the Lichess challenge API or their profile. Request `color=white` and `color=black` in matched pairs, `clock.limit=600`, `clock.increment=10`, `variant=standard`, and the chosen rated/casual mode.

For example, challenge one consenting account as White, then reverse colors for the paired game:

```text
curl -X POST https://lichess.org/api/challenge/OPPONENT_BOT \
  -H "Authorization: Bearer $LICHESS_BOT_TOKEN" \
  -d "rated=false" -d "color=white" -d "variant=standard" \
  -d "clock.limit=600" -d "clock.increment=10"

curl -X POST https://lichess.org/api/challenge/OPPONENT_BOT \
  -H "Authorization: Bearer $LICHESS_BOT_TOKEN" \
  -d "rated=false" -d "color=black" -d "variant=standard" \
  -d "clock.limit=600" -d "clock.increment=10"
```

Use `rated=true` only after changing the incoming policy to rated and agreeing on the rated campaign. Real-time challenges expire if they are not accepted promptly; automation can use the API's `keepAliveStream=true` option.

Lichess direct standard challenges cannot force an opening sequence. Color pairing is available; paired openings are not. Use the local Cute Chess gauntlet for controlled openings.

### Automatic individual matchmaking

Enable the block shown above. The client finds online bots and challenges them according to the rating/time filters. This is convenient for availability testing but less defensible as a calibrated experiment.

### Lichess Arena tournament

Only use an Arena whose conditions explicitly allow bots. Ordinary Arenas reject BOT accounts. The organizer can create an Arena with `conditions.bots=true`, or the bot can join an existing bot-enabled Arena.

Joining via API requires a token with `tournament:write`:

```text
curl -X POST https://lichess.org/api/tournament/TOURNAMENT_ID/join \
  -H "Authorization: Bearer TOKEN_WITH_TOURNAMENT_WRITE" \
  -d "pairMeAsap=true"
```

Keep `lichess-bot.py` running so `gameStart` events are received and games are played. Arena pairings, colors, opponents, and openings are controlled by Lichess, not this YAML. The `matchmaking` block should remain disabled during an Arena. An Arena is useful for public play and opponent discovery, but it is not the controlled 96-game calibration design.

Creating a private bot-enabled Arena also uses the tournament API with `tournament:write`, clock/tournament-duration fields, and `conditions.bots=true`. Creation does not define a fixed round robin; Lichess Arena pairing rules still apply. For an exact opponent list, exact color pairs, or paired openings, use direct challenges or local Cute Chess instead.

## Shutdown and recovery

The foreground process handles `SIGINT`. Preferred shutdown:

1. Disable new direct challenges/matchmaking if applicable.
2. Send one `SIGINT` and wait for the active game to finish and its PGN to appear.
3. Copy `/data/pgn` and `/data/logs` off the instance.
4. Stop the Vast instance.
5. Verify the export, then destroy the instance to stop storage billing.

If the process or network disconnects, rerunning `run_vast_bot.sh` reconnects the global event stream. Lichess sends current games when that stream opens, each game stream starts with full state, and the UCI `position` command supplies the complete move history. The engine can therefore reconstruct the correct board/history instead of depending on lost process memory.

## Recommended first SSH run

1. Upgrade the unused BOT account locally, but do not yet put its token on the node.
2. Rent one on-demand Ubuntu 22.04 CUDA development instance with a 16 GB GPU, 4+ CPU cores, and 50 GB disk in SSH mode.
3. Run the latest `setup_vast.sh` and inspect every successful checkpoint; retain `/data/alphazero-install.txt`.
4. Run `smoke_vast.sh` with its default public model; confirm a legal best move and GPU activity, then retain `/data/model-source.txt`.
5. Confirm the checked-in `modes: [casual]` and disabled matchmaking are unchanged.
6. Enter the Lichess bearer token, start `run_vast_bot.sh` in the persistent shell, and watch the foreground log.
7. Challenge it from one consenting bot at `10+10`.
8. Verify search duration, reconnect behavior, PGN export, and clean shutdown.
9. Only then enable rated games or a pre-registered direct-match schedule.

For the statistical campaign, rating caveats, PGN reconciliation, and local Stockfish/Cute Chess comparison, see [`documentation/lichess_vast_evaluation.md`](../../documentation/lichess_vast_evaluation.md).
