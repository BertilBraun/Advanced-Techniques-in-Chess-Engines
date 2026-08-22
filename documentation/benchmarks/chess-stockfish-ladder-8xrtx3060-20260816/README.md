# Stockfish ladder — four-day-run checkpoints, 8×RTX 3060, 2026-08-16

Gauntlet outputs measured against the four-day run's checkpoints on the 8×3060 node
(source range `d39d5c85..d9888436`, tag `four-day-baseline`). Runbook:
[operations/stockfish-gauntlet.md](../../operations/stockfish-gauntlet.md). These files previously lived in
`chess-results/` next to the unrelated 2024 legacy artifacts (now
[evidence/chess-legacy-a10-2024/](../../evidence/chess-legacy-a10-2024/README.md)).

- [stockfish13-ladder-generation445-search1000.md](stockfish13-ladder-generation445-search1000.md) — the
  generation-445 ladder report (66.0% vs Stockfish 13 at 6,500 nodes).
- [stockfish13-1000nodes-generation308.json](stockfish13-1000nodes-generation308.json) — generation-308
  fixed-node match record.
- [chess-elite-2025-11-balanced-4moves-200-v1-report.json](chess-elite-2025-11-balanced-4moves-200-v1-report.json)
  — elite-opening gauntlet report.

Predates `TEMPLATE.md`; provenance is recorded inside each report.
