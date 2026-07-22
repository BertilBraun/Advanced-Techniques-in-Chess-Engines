# Public web play

The web-play deployment consists of a static Vite client in `web/` and a typed
FastAPI service in `py/web_play/`. The API is authoritative for FEN and move
validation. The browser sends the starting FEN and complete UCI history on every
turn, so a request can recover after Modal has scaled the only container to zero.
The client stores the last authoritative game state and play settings in browser
local storage. Reloading the same site origin restores the board immediately;
the next turn reuses the UUID session when it still exists or reconstructs the
game from complete history after a cold start. Starting a new game clears the
stored state.

## Analysis semantics

- `policy` evaluates a fresh history-aware root exactly once and chooses the
  highest policy prior. Ties are broken by ascending UCI text.
- `mcts` reuses the current tree where possible and chooses the largest final
  visit count. Ties are broken by higher policy prior and then ascending UCI
  text. UCB is used only inside native traversal.
- `root_value` is `P(win) - P(loss)` from the side-to-move perspective at the
  analyzed root. `outcome_prediction` preserves the model's win, draw, and loss
  probabilities at that root in the same side-to-move perspective. For MCTS it
  is the root network evaluation, while visits and mean values describe search.
- Candidate `mean_search_value` is converted to the analyzed root player's
  perspective. Candidate visits and shares describe the retained tree; the
  `searches` metric reports work added by the current request.
- Timed MCTS accepts only 1–30 seconds. The native engine stops issuing direct
  inference batches before the deadline using its measured inference latency,
  then drains all submitted work.
  Count-based limits exist only in the engine contract for deterministic tests
  and benchmarks; the public API does not expose them.

Only one request enters the Modal container at a time, and `GameService` also
serializes engine work. The deployment uses the production indexed interactive
search tree with two direct inference workers, batches of up to 64 positions,
and at most two outstanding batches per worker. Policy mode evaluates exactly
one position without mutating the MCTS frontier. An individual game must never
be called concurrently.

## Local backend

Build `AlphaZeroCpp` as described in `cpp/README.md`, with the resulting shared
library in `py/`. Install the web dependencies and start from `py/`:

```powershell
python -m pip install -r .\requirements-web.txt
$env:CHESS_MODEL_PATH='C:\absolute\path\to\model.jit.pt'
$env:CHESS_WEB_ALLOWED_ORIGINS='http://localhost:5173'
python -m uvicorn web_play.local_app:app --host 127.0.0.1 --port 8000
```

The local model path is intentionally required. API tests inject a test-local
engine and therefore do not need the native extension or a trained model:

```powershell
python -m pytest --import-mode=importlib .\test\web_play -q
```

## Local browser client

From `web/`, copy `.env.example` to `.env.local` if the API is not on the
documented local URL, then run:

```powershell
npm install
npm run dev
```

Use `npm run build` for the static production bundle in `web/dist/`. Host that
directory on any static host and set `VITE_API_BASE_URL` at build time to the
deployed Modal API URL.

## Modal deployment

The deployment builds the CUDA-enabled native extension into an ephemeral image.
It requests one NVIDIA A10 GPU, has zero warm containers, at most one container,
four CPU cores, 4 GiB of requested system memory, a 300-second scale-down window,
a 90-second request timeout, and no Modal Volume. CUDA is required explicitly;
startup fails instead of silently falling back to CPU inference. At container
startup it downloads the named `.pt` and `.jit.pt` artifacts into the
revision-aware Hugging Face cache, then loads two TorchScript replicas once for
the direct inference workers. Startup warms the single-position path on both
workers, then runs a fixed 4,096-search warm-up to initialize full batches and
stabilize the native deadline estimator before the first request. The native
extension also disables build-host-specific CPU instructions because Modal may
serve the image on a different host. With revision `main`, each cold container
resolves the branch to one commit before downloading either artifact, so both
files come from the same latest snapshot.

The 300-second window is five minutes after the container becomes idle. Scaling
to zero discards in-memory sessions, inference cache, and search subtrees, but it
does not lose a browser game: each subsequent turn contains the starting FEN and
complete move history. Recovery may pay another cold-start delay and does not
retain the previous search tree, but produces the same correct position.

Create the fixed-name Modal secret with every required setting. Use revision
`main` to load the newest snapshot after scale-to-zero, or a full 40-character
commit hash for a fixed deployment. Other branch names, tags, and abbreviated
hashes are rejected.

```powershell
modal secret create chess-web-play `
  CHESS_MODEL_REPO_ID=owner/repository `
  CHESS_MODEL_REVISION=main `
  CHESS_MODEL_CHECKPOINT_FILENAME=model.pt `
  CHESS_MODEL_TORCHSCRIPT_FILENAME=model.jit.pt `
  CHESS_WEB_ALLOWED_ORIGINS=https://your-static-site.example
```

For a private Hugging Face repository, add `HF_TOKEN` to the same secret. Deploy
from the repository root:

```powershell
modal deploy .\py\web_play\modal_app.py
```

After Modal prints the endpoint URL, build the client with that exact origin:

```powershell
$env:VITE_API_BASE_URL='https://your-workspace--chess-model-web-play-web.modal.run'
Set-Location .\web
npm run build
```

If the static host origin changes, update `CHESS_WEB_ALLOWED_ORIGINS` and redeploy
the API. Do not use `*` in production.
