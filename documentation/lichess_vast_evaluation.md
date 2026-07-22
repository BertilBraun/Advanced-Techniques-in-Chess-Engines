# Lichess public bot, Vast.ai deployment, and strength calibration

Verified against the public interfaces and documentation on 2026-07-21. No account, token, challenge, rental, upload, or other external mutation was performed.

## Recommended integration

Use `lichess-bot` with this repository's UCI process. This is cleaner than a `homemade` engine integration:

- UCI keeps the model/search implementation independent from Lichess networking and also makes the same executable usable by Cute Chess and other tournament managers.
- `lichess-bot` already owns challenge filtering, one global event stream, one game stream per active game, reconnects, matchmaking, PGN persistence, and graceful shutdown.
- A homemade integration subclasses Python code inside the separately versioned AGPL `lichess-bot` project and couples model lifecycle to its worker lifecycle. It offers no useful advantage for a standard-chess engine.

The adapter is `python -m src.uci --model MODEL.jit.pt`. It accepts `uci`, `isready`, `setoption name SearchMode value policy|mcts`, `ucinewgame`, `position startpos|fen ... moves ...`, `go movetime 1000..30000`, `stop`, and `quit`. It owns the production `src.eval.InteractiveEngine`, which uses the native indexed search tree and direct pipelined inference workers. It advances a game when the new complete history extends the old one and reconstructs from the supplied starting FEN and complete UCI history after divergence or process recovery. MCTS selects the final move by maximum visit count; UCB remains internal traversal policy. Only UCI protocol output is written to stdout and diagnostics go to stderr.

The native `analyze` binding releases the Python GIL. UCI MCTS divides a requested move time into optimized one-second native searches, preserving the same reusable native tree while allowing `stop` to be observed between slices. The direct-inference defaults are two workers, batches of 64, and two outstanding batches per worker, matching the integrated RTX 3060 benchmark recommendation.

## Creating a BOT account and token

The exact sequence is:

1. Create a separate normal Lichess account and do **not play any game on it**.
2. While logged into that account, create a personal OAuth token at `https://lichess.org/account/oauth/token/create?scopes[]=bot:play` with the `bot:play` (“Play bot moves”) scope.
3. Upgrade once with the official client (`python lichess-bot.py -u`) or `POST /api/bot/account/upgrade` with `Authorization: Bearer TOKEN` and an empty body.
4. Confirm `GET /api/account` returns `title: "BOT"`.
5. Store the token only as the `LICHESS_BOT_TOKEN` runtime secret. `lichess-bot` explicitly overrides the placeholder YAML token with that environment variable.

The [official upgrade operation](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-account-upgrade.yaml) says the account must have played zero games, conversion is irreversible, and the converted account can only play as a bot. Only BOT accounts can call Bot API operations. A personal token needs only `bot:play` for the event/game/challenge flow; do not grant unrelated scopes. Tournament creation is separate and requires `tournament:write`—it is unnecessary for the proposed direct-challenge campaign.

There is no bot callback URL, public listening port, endpoint registration, or IP registration in the upgrade or Bot API. Authentication is the bearer token. A Vast instance initiates ordinary outbound TLS connections to `https://lichess.org`; dynamic public IP is acceptable. This is an inference from the complete official operation schemas, which define token authentication and no registration/allow-list parameter.

### Account and play limitations

The [official Bot API section](https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml) says:

- engine assistance is allowed for BOT accounts;
- bots use challenges, not lobby pools;
- UltraBullet `1/4+0` is forbidden, while `0+1` and `1/2+0` are allowed;
- sandbagging, boosting, constant aborting, and the rest of the Terms of Service still apply;
- testing should initially be casual.

The same section broadly says tournaments are off-limits, but the current [Arena creation schema](https://github.com/lichess-org/api/blob/master/doc/specs/tags/arenatournaments/api-tournament.yaml) has `conditions.bots`, default false, and Lichess runs arenas visibly marked “Bot players are allowed.” The operational rule is therefore: do not enter ordinary arenas; only join an explicitly bot-enabled arena. Arena pairings and openings are uncontrolled, so they are useful for community discovery but not the primary calibration dataset.

## Challenges, matchmaking, and streams

Direct bot-vs-bot games use `POST /api/challenge/{username}`. The [challenge schema](https://github.com/lichess-org/api/blob/master/doc/specs/tags/challenges/api-challenge-username.yaml) supports rated/casual mode, requested color, standard/from-position variants, and real-time clocks. Initial clock values are 0, 15, 30, 45, 60, 90, or a multiple of 60 through three hours; increment is 0–60 seconds. It does **not** offer a literal “10 seconds per move” control. Use `600+10` on Lichess and configure the engine to spend 10 seconds per move. This gives network/long-game headroom while preserving a fixed engine compute budget.

A real-time challenge normally expires after 20 seconds. `keepAliveStream=true` holds it while the response stays connected and ends with accepted/declined/canceled status. After acceptance, the game ID equals the challenge ID and `gameStart` appears on the account event stream. `lichess-bot` can either accept incoming bot challenges or enable its matchmaking block to select from online bots by rating and challenge them. Keep automatic matchmaking disabled until opponents consent and the casual shakeout passes.

Discovery options are the unauthenticated [`GET /api/bot/online`](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-online.yaml), the [community bots page](https://lichess.org/player/bots), bot teams, and explicitly bot-enabled arenas. Read each profile's accepted controls and bot-challenge policy. Prefer an allow-list of willing, established bots around the expected strength rather than repeatedly challenging arbitrary accounts.

### What remains running

`lichess-bot.py` is the long-lived supervisor and must remain running for the account to appear online and play:

1. It holds the single authenticated [`GET /api/stream/event`](https://github.com/lichess-org/api/blob/master/doc/specs/tags/board/api-stream-event.yaml) NDJSON connection. Lichess sends an empty keepalive every seven seconds, current challenges/games on open, then `challenge`, `gameStart`, `gameFinish`, cancellation, and decline events. Opening another global stream with the same token closes the old one, so run one supervisor per token.
2. For every active game it opens [`GET /api/bot/game/stream/{gameId}`](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-game-stream-gameId.yaml). Its first record is always `gameFull`, followed by complete `gameState` move lists, chat, and opponent-away events.
3. Each game worker starts the UCI child, sends the complete `position ... moves ...` history, requests a move, then posts that legal UCI move to the Bot API.

The event stream is not a durable queue, but reopening it gives all current games/challenges. The game stream's first `gameFull` plus complete move list makes recovery deterministic. The pinned `lichess-bot` implementation reconnects the control stream, restarts after network errors, applies exponential backoff to game work, checks ongoing games at startup, and saves a PGN even on game-worker errors. A disconnect therefore causes reconnect/reconstruction rather than reliance on process memory.

Follow [Lichess API rate-limit guidance](https://lichess.org/page/api-tips): make only one non-stream request at a time and, after any HTTP 429, wait a full minute. Do not build a rapid custom retry loop around move or challenge POSTs. The current `lichess-bot` configuration also warns that a bot can play at most 100 games per day against other bots; the 96-game plan leaves margin for shakeout/retries.

With `quit_after_all_games_finish: true`, the first `Ctrl-C` stops new work and lets active games finish; a second forces exit. Leave the foreground process running until there are zero active games and PGNs are flushed, then stop the instance. Never destroy the rental before export.

## Vast.ai one-day runbook

Use the inspectable SSH workflow in [`deployment/lichess/README.md`](../deployment/lichess/README.md): pipe the latest setup script into Bash, build `AlphaZeroCpp` on the rented node, run the native/Python tests and real model/UCI/CUDA smoke test, then start `lichess-bot` in the persistent foreground shell. Select an on-demand instance for the first public run, enough disk for dependencies, build products, model, and logs, reliable networking, and a GPU compatible with CUDA 12.8. [Vast instance management](https://docs.vast.ai/guides/instances/manage-instances) distinguishes stop (data retained, storage billing continues) from destroy (data permanently deleted).

Enter `LICHESS_BOT_TOKEN` interactively in the shell. `HF_TOKEN` is only needed for a private model. Do not put either token in Git, shell history, on-start scripts, or logs. Vast hosts can technically access instance files; use an appropriately trusted/verified host and minimum-scope short-lived secrets. Revoke/rotate both tokens after the campaign.

The default downloader resolves the latest `BertilBraun/alphazero-chess/latest.jit.pt` to a commit and records `/data/model-source.txt`. Hugging Face documents that [`revision` selects a commit/tag/ref](https://huggingface.co/docs/huggingface_hub/guides/download). Preserve the provenance file with the evaluation.

Check before accepting games:

```text
nvidia-smi
test -f /data/models/latest.jit.pt
cat /data/model-source.txt
tail -f /data/logs/lichess-bot.log
```

The instance only needs outbound TCP 443 and no published inbound service port. SSH is operational access, not a Lichess endpoint. PGNs go to `/data/pgn`, persistent logs to `/data/logs`, and the model/cache to `/data/models`.

Shutdown/export sequence:

1. Disable matchmaking or send one interrupt and wait for all active games to finish.
2. Export `/data/pgn`, `/data/logs`, `/data/alphazero-install.txt`, `/data/model-source.txt`, the exact config with secrets redacted, and `nvidia-smi` output. Use Vast Copy Data, SCP/SFTP, or an encrypted object-store sync; [Vast data-movement docs](https://docs.vast.ai/guides/instances/storage/data-movement) cover the platform options.
3. Independently export the account games from [`GET /api/games/user/{username}`](https://github.com/lichess-org/api/blob/master/doc/specs/tags/games/api-games-user-username.yaml) as a reconciliation copy.
4. Stop the instance, verify the export, then destroy it. Stopped storage continues billing; destroyed storage cannot be recovered.

## Approximately 100 Lichess games

Treat this as public performance characterization, not a laboratory Elo test.

1. Run 4–10 casual shakeout games against consenting bots. Verify legal moves, time use, recovery, PGNs, and no repeated challenge/abort behavior. Exclude these from the rating dataset.
2. Pre-register the artifact digest, model commit, UCI configuration, `600+10` clock, fixed 10-second engine move budget, opponent inclusion rules, and analysis method.
3. Recruit 8–12 established BOT accounts spanning roughly ±300 Lichess rapid points around the engine. Require non-provisional opponent rapid ratings where possible and record rating plus rating deviation/snapshot time.
4. Target 48 matched color pairs = 96 completed rated games. For each opponent issue one white and one black challenge, alternating which color is played first across pairs. Do not exceed an opponent's stated limits. Spread games over opponents rather than farming one account.
5. Standard direct challenges cannot force an opening. `fromPosition` can specify a FEN, but it is a separate variant/performance pool and many bots reject it. Therefore color pairing is feasible on Lichess; controlled paired openings are not. Use the local gauntlet for opening control.
6. Count checkmate, resignation, draw, and time-forfeit results as played. Keep aborted games and disconnect logs in the audit trail but replace games that ended before substantive play; state the pre-registered cutoff. Do not silently discard a genuine loss caused by this deployment. Lichess explicitly does not restore rating for ordinary disconnect losses.
7. Reconcile local PGNs against the API export by game ID. Report W/D/L, score `(W + D/2) / N`, opponent/rating distribution, color split, termination reasons, and the final Lichess rapid rating with its provisional marker/RD.

Lichess uses Glicko-2, starts accounts at 1500 with very high uncertainty, and marks a rating provisional when deviation exceeds 110; see the [official rating FAQ](https://lichess.org/faq#ratings) and [rating-system explanation](https://lichess.org/page/rating-systems). A Lichess rapid BOT rating is relative to that platform, time-control, and opponent pool. It is not FIDE Elo, human playing strength, Chess.com rating, or an engine-intrinsic constant.

For the fixed-game result, compute uncertainty by bootstrap-resampling whole color pairs (not individual games) and recomputing score and the logistic performance difference

```text
Delta = 400 * log10(score / (1 - score)).
```

Use a continuity adjustment when a bootstrap sample scores exactly 0 or 1, and report the 2.5/97.5 percentiles. This “Elo-like performance difference” assumes the Elo logistic curve and is separate from the account's Glicko-2 rating. Around a 50% score, even 100 independent decisive-equivalent observations have a best-case 95% half-width near 68 Elo before opponent-rating uncertainty, draws, pairing correlation, and selection bias. Thus 96–100 games only locate a broad strength band.

## Controlled local UCI gauntlet

The defensible primary report metric is the paired match score and paired-bootstrap 95% interval against **named, immutable engine artifacts at fixed settings**. A secondary interpolation may report an “equivalent Stockfish `UCI_Elo` setting,” but never label that FIDE/human/server Elo. Stockfish's limiter is version-calibrated and hardware/time-control dependent.

Use 96 games as four 24-game matches against, for example, Stockfish settings 1600, 1800, 2000, and 2200. Pin the Stockfish binary hash/version. Give both engines 10 seconds per move, one process/game at a time, and the same machine. Use at least 12 neutral opening FENs, each played twice with colors reversed. The repository's prepared evaluation opening suite can be converted to EPD if needed.

`tools/run_cutechess_gauntlet.py` constructs the matches:

```text
python tools/run_cutechess_gauntlet.py \
  --cutechess /path/to/cutechess-cli \
  --stockfish /path/to/pinned/stockfish \
  --model /path/to/model.jit.pt \
  --openings /path/to/openings.epd \
  --output-directory /data/local-gauntlet
```

It uses Cute Chess `st=10`, `-repeat`, sequential round-based openings, color swaps, one concurrent game, crash recovery, and finished-game PGNs. These behaviors are defined in the [official Cute Chess CLI manual](https://github.com/cutechess/cutechess/blob/master/docs/cutechess-cli.6). Preserve all stdout, PGNs, engine stderr, opening file/hash, executable hashes, CPU/GPU identity, and command line.

Report each opponent separately and the bracketing result (“scores above 50% at setting X and below at Y”). If interpolating, fit a logistic model using all game outcomes with pair-clustered/bootstrap uncertainty. Do not pool raw games across different Stockfish settings into one unweighted score. The local gauntlet supports controlled comparative claims; the Lichess campaign supports an ecological public-platform claim. Reporting both, distinctly labeled, is more defensible than a single “engine Elo.”

## Primary sources

- [Lichess OpenAPI specification](https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml)
- [BOT upgrade](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-account-upgrade.yaml), [event stream](https://github.com/lichess-org/api/blob/master/doc/specs/tags/board/api-stream-event.yaml), [game stream](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-game-stream-gameId.yaml), and [challenge creation](https://github.com/lichess-org/api/blob/master/doc/specs/tags/challenges/api-challenge-username.yaml)
- [Official lichess-bot repository](https://github.com/lichess-bot-devs/lichess-bot/tree/36ba4575b3299646956c911b11487b1c348f4b0c) and [default configuration](https://github.com/lichess-bot-devs/lichess-bot/blob/36ba4575b3299646956c911b11487b1c348f4b0c/config.yml.default)
- [Lichess API rate-limit guidance](https://lichess.org/page/api-tips) and [rating systems](https://lichess.org/page/rating-systems)
- [Vast instance lifecycle](https://docs.vast.ai/guides/instances/manage-instances) and [data movement](https://docs.vast.ai/guides/instances/storage/data-movement)
- [Hugging Face revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Cute Chess CLI manual](https://github.com/cutechess/cutechess/blob/master/docs/cutechess-cli.6) and [Stockfish repository](https://github.com/official-stockfish/Stockfish)
