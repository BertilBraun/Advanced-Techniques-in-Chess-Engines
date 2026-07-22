# Public web play

The web-play deployment consists of a static Vite client in `web/` and a typed
FastAPI service in `py/web_play/`. The API is authoritative for FEN and move
validation. The browser sends the starting FEN and complete UCI history on every
turn, so a request can recover after Modal has scaled the only container to zero.

## Analysis semantics

- `policy` evaluates a fresh history-aware root exactly once and chooses the
  highest policy prior. Ties are broken by ascending UCI text.
- `mcts` reuses the current tree where possible and chooses the largest final
  visit count. Ties are broken by ascending UCI text. UCB is used only inside
  native traversal.
- `root_value` is `P(win) - P(loss)` from the side-to-move perspective at the
  analyzed root. `outcome_prediction` preserves the model's win, draw, and loss
  probabilities at that root in the same side-to-move perspective. For MCTS it
  is the root network evaluation, while visits and mean values describe search.
- Candidate `mean_search_value` is converted to the analyzed root player's
  perspective. Candidate visits and shares describe the retained tree; the
  `searches` metric reports work added by the current request.
- Timed MCTS accepts only 1–30 seconds and stops scheduling native search chunks
  at the deadline. One already-running native inference cannot be interrupted.
  Count-based limits exist only in the engine contract for deterministic tests
  and benchmarks; the public API does not expose them.

Only one request enters the Modal container at a time, and `GameService` also
serializes engine work. The deployment currently uses one native search worker
because the legacy `EvalMCTS` parallel evaluator is not stable for long-running
interactive searches. An individual game must never be called concurrently.

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

The deployment builds the CPU-only native extension into an ephemeral image. It
has zero warm containers, at most one container, four requested CPU cores, 4 GiB
of requested memory, a 300-second scale-down window, a 90-second request timeout,
and no Modal Volume. The explicit memory request is needed because the native
MCTS tree retains expanded positions throughout a search. At container startup
it downloads the named `.pt` and `.jit.pt` artifacts into the revision-aware
Hugging Face cache, then loads the TorchScript model once.

Create the fixed-name Modal secret with every required setting. The revision must
be the full 40-character commit hash; branch names, tags, `main`, and abbreviated
hashes are rejected.

```powershell
modal secret create chess-web-play `
  CHESS_MODEL_REPO_ID=owner/repository `
  CHESS_MODEL_REVISION=0123456789abcdef0123456789abcdef01234567 `
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
