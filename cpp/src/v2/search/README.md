# Fixed PUCT accounting and value conventions

The fixed-budget search expands and evaluates a nonterminal root exactly once
before its simulation loop. That root initialization is not a simulation. Each
configured simulation then completes one traversal from the root to one leaf,
evaluates the leaf only when it is nonterminal, and backs its value up through
every node on the path. Consequently, for every completed nonterminal search:

```text
configured_cap
  = actual_simulations
  = root_visit_count
  = sum(root child visits)
```

A terminal root performs no inference and no simulations, has no selected
action, and reports `terminal_root`. Its root value is optional: a scored
terminal (including a real draw) supplies a value, while a censored terminal has
no value. If a simulation reaches a censored terminal leaf, search performs one
value inference without expanding legal actions and backs up that prediction.

Telemetry reports root-initialization, leaf, and total inference request counts
separately from simulations. A simulation that reaches a scored terminal leaf
increments simulation/visit accounting without inference; a censored terminal
leaf increments the leaf-inference count.

Node values use the perspective of the player to act at that node. Moving one
edge toward the parent negates the value and applies `backup_discount`.
Selection therefore reads a visited child's action value as
`-backup_discount * child_mean_value`. Unvisited actions use the arithmetic mean
of visited-child action values, or `no_visited_child_value` when no child has
been visited. PUCT uses
`Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)` and resolves
equal scores by greater prior, then by the legal-action order supplied by the
game. The first simulation is therefore prior-sensitive even though the
standard exploration term is zero before the root has visits.

The current tree does not retain each node's initial inference value because
visited-child-mean FPU never consumes it. Stage 8 must retain that value when it
adds parent-value and reduced-parent-value FPU; it must not re-run inference to
recover it.

Inference policy output is a runtime-sized vector indexed by the game's action
space. Entries must be finite and non-negative. Search masks illegal actions and
renormalizes legal mass; zero legal mass becomes a uniform legal policy. Values
must be finite and in `[-1, 1]`.

The policy target is normalized root visit count, independent of move-selection
temperature. Temperature zero selects the greatest visit count with stable
legal-order tie-breaking. Positive temperature samples from the normalized
`visits^(1 / temperature)` distribution, calculated relative to the maximum
log-visit count to avoid overflow even for very small positive temperatures,
using only the explicitly seeded search stream.
Optional root Dirichlet noise uses that same owned stream. No implicit or global
random source is used.
