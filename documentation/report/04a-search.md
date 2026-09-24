# 4.1. Spending search where it matters

Search was the clearest way to make a fixed network play better, but also one of the most expensive parts of the
learning loop. That made allocation an unusually attractive research target: if easy positions could be recognized
early, their unused simulations could be moved to difficult positions or converted into more games. The project
tested that premise at several levels, from simple stopping rules to learned controllers and a complete graph-search
implementation. The result was not a better allocator. It was a better understanding of why the transparent fixed
budget remained hard to beat.

Throughout this chapter, three outcomes are kept separate. Fixed-network playing strength asks whether search
chooses stronger moves. Target fidelity asks whether a shallow policy resembles a deeper one. End-to-end learning
asks whether the resulting data makes the next network improve faster. Throughput is a fourth, systems-level
quantity. Success on any one of these axes did not guarantee success on the others. Chapter 2 states the
measurement rules used below.

![The measured search alternatives and the gates at which their expected savings failed to improve the learning loop](figures/search-decision-gates.svg)

Figure 2: Each rejected idea met a different constraint. Fast/full search damaged target coverage and endgame
semantics; predicted allocation improved its fidelity proxy but lost online strength; in-search stopping saved mostly
non-critical-path work; and exact graph or cache reuse failed to repay overhead. These are results under the measured
chess workload, not universal impossibility claims.

## The fixed-budget baseline

The native engine performs policy-guided Monte Carlo tree search. The network supplies legal-move priors and a
win/draw/loss value; PUCT selects leaves; neural evaluations are batched across active games; and values are backed
up from the alternating player's perspective. A visit cap is deliberately unambiguous. It does not depend on a
difficulty estimate, a calibrated controller, or a learned model that may become stale as training advances.

The fixed-network sweep established that this simple control was valuable. Against the same opponent, score rose
from 0.318 at 200 visits to 0.537, 0.580, 0.662, and 0.748 at 400, 600, 1,000, and 1,600 visits. Across the better
resolved middle and deep part of the sweep, the fitted trend was about 81 Elo per doubling. Individual adjacent
comparisons were noisy, so the monotone trend is more credible than any single increment. Search was also still
changing the training target: at 600 visits, only 72.4% of positions selected the same best move as the network's
own 10,000-visit reference. These measurements support spending more search when resources permit; they do not
identify an optimal training schedule.

The retained schedule therefore increased the cap in auditable stages—300, 400, 500, 600, and finally 800
visits—rather than presenting those boundaries as independently optimized discoveries. Even this count needs care:
a retained root can begin a move with existing statistics, so a nominal visit cap is not always the amount of new
work performed.

## Why fast and full searches did not transfer

A scheme inspired by KataGo's self-play methods [7]
offered an appealing alternative. Most moves could use a cheap search to advance the game,
while a random minority received a full search and became primary policy targets. In long Go games this can exchange
some target density for many more independent terminal outcomes. Chess did not show the same bottleneck. Games were
shorter, positions were often more decisive, and the value objective was already learning. Discarding cheap-search
positions therefore reduced the supply of policy targets without a demonstrated compensating benefit from more
finished games.

The cheap moves were not free. With a quarter of moves searched to 600 visits and the rest to 150, the cheap moves
still consumed 42.9% of nominal search work. Nor were they isolated from training: they selected played moves,
affected terminal outcomes and retained trees, could supervise a preceding row's next-policy target, and influenced
which positions entered the restart archive. “Not stored as a primary policy row” was never the same as “irrelevant.”

The mixed workload also interacted badly with batching. Once the cheap searches finished, only the full-search
minority remained. In a 512-game test this tail left about 128 active trees and filled only 86 of 320 available batch
slots with one leaf per tree. Allowing four leaves per tree raised the average batch to about 268 and improved search
throughput by roughly 20%, but it did so by making each search less serial. A policy intended to save compute thus
created pressure to accept a search-quality tradeoff merely to keep the accelerator occupied.

Most importantly, the design exposed a semantic failure at the end of games. Forcing cheap searches after a late
ply removed searched endgame rows, while using the shallow cutoff value could propagate a weak soft target back
through the capped game. The later cut-game repair and removal of the cheap tail addressed that failure. The project therefore
kept KataGo's broader insight—that search cost and target eligibility are distinct choices—but superseded the
random fast/full recipe with a full search for every recorded move. This was a systems-and-target-semantics decision,
not a clean matched-compute Elo victory for all-full search; no such long-run ablation was preserved.

## Three attempts to allocate search adaptively

The first attempt asked whether search could stop when the visit leader looked uncatchable. An audit of completed
trees found apparent room: an optimistic reconstruction suggested that a rule requiring at least 75% of the cheap
budget might remove 15.3% of cheap-search visits, or 6.38% of all nominal limits if full searches saved nothing.
But the records contained final visit distributions, not the intermediate traces needed to identify when the leader
first became safe. They also omitted enough state to reconstruct retained visits and forced-playout-pruned targets.
The apparent saving was therefore an oracle-like opportunity estimate, not an observed stopping result.

A wider offline study found a deeper problem. The marginal value of search was not monotone in visit concentration:
very diffuse and already-decided positions gained little, while moderately concentrated but contested positions
gained most. Simple concentration thresholds tended to stop in the very region where more search was useful. The
rule-based approach was audited and declined before an online strength match; it should not be described as an
implemented algorithm that lost Elo.

The next allocator replaced the rule with a learned prediction, related in aim but not identical to
dynamic simulation stopping [3]. An auxiliary head estimated, for several candidate
budgets, the divergence between that budget's policy and a deep-search policy. A calibrated corrector incorporated
root observables, and a dual variable kept average spend near its target. Deep labels, replay persistence, model
publication, native budget selection, safety gates, and telemetry were all implemented. Mechanically, the system
worked. At approximately matched mean spend it captured about 23% of the available policy-divergence headroom; its
mean target fidelity resembled roughly 1.18 times uniform search at 0.967 times the spend. It also behaved plausibly,
assigning more work to contested positions.

Learning nevertheless deteriorated. Repeated online attempts trailed comparable non-adaptive training by roughly
60–100 ladder Elo. More than a third of positions received an average budget fraction near 0.36, and almost 9%
received one eighth of baseline search. Those shallow policies were close to the network's own prior yet were
trained at full weight. The likely failure was therefore objective mismatch: closeness to a deep policy at the
current position does not measure how much a target will improve the next network. That mediation remains a
hypothesis, but the central result does not: the controller improved its stated proxy and still produced worse
learning.

The final adaptive system moved the decision inside search, where a learned stopper could observe the evolving tree
rather than predict difficulty in advance. Its decisive test started from the same checkpoint, optimizer, and
rebuilt replay state for every arm. The most aggressive setting skipped about 14% of nominal search, and its
internal credit-wait measurements changed in the expected direction. Yet generation cadence improved by only about
3%. Self-play overlapped the optimizer, so most of the removed search was slack rather than critical-path work.
Paired strength estimates were +1.7 ± 9.9 Elo and -4.2 ± 10.1 Elo (standard errors) for the two stopping settings: neither resolved a
strength effect. At the observed learning rate, the cadence gain was worth only about one Elo over three hours,
below the experiment's resolution.

These studies failed for different reasons. The threshold rule lacked an identifiable safe signal. The predicted
allocator optimized a measurable but insufficient proxy. Learned stopping saved the resource it targeted, but that
resource was mostly off the critical path. In a non-overlapped or inference-bound system the last conclusion could
change; nothing here proves adaptive search universally useless.

## Parallel leaves: buying latency with search quality

Batching across many independent games is the cleanest way to feed the accelerator, but the number of active roots
eventually runs out. The engine can then keep several leaf traversals in flight from one root. Virtual reservations
discourage those traversals from selecting the same path, yet every selection is based on a tree that is missing the
other in-flight results. Parallel search is therefore not serial MCTS executed faster. It exchanges fresher
decisions for larger batches and lower latency.

The experiments exposed how easy it is to measure the wrong regime. An initial sweep suggested little cost per
doubling, but its batch capacity was too small relative to the number of trees, so the configured per-root
parallelism barely bound. A deliberately oversized-batch rerun activated the knob and estimated a steeper
-6.4 ± 4.7 Elo per doubling. The interval still included a small effect, and the rerun was diagnostic rather than a
production throughput configuration.

The tradeoff also depended on the total budget. At 1,000 searches, a terminal fixed-network sweep measured point
losses of 19 Elo with four leaves and 45 Elo with sixteen leaves relative to serial search, while sixteen-way
parallelism was substantially more damaging at 100 searches. The likely explanation is that a deep search will
eventually visit more of the temporarily suboptimal leaves selected from stale state. However, only a few budget and
parallelism combinations were measured, and the terminal strength curve itself uses different parallel counts at
different depths. It is not a pure scaling curve from which a universal safe-parallelism law can be inferred.

The retained design consequently uses bounded parallelism as a serving parameter, not a free algorithmic speedup.
The right setting depends on available independent roots, batch capacity, total search depth, latency requirements,
CPU load, and memory. Older results from the fast/full batching tail do not directly prescribe the topology of the
final all-full workload.

## When a tree became a graph

Chess appears rich in transpositions: different move orders often reach the same board. The project tested whether
Monte Carlo graph search, as in Czech, Korus, and Kersting [6], could turn those transpositions into shared neural evaluations, descendants, and search
statistics. This was a complete implementation, not a cache mislabeled as graph search. Canonical nodes held shared
state-level information; parent/action edges retained local PUCT statistics; correction backups exposed better
shared values to incoming edges; and the system handled trajectory reservations, cycles, rerooting, pruning, and
capacity reclamation. An audit against both the paper and its reference implementation found and corrected a
missing first-link backup behavior.

The limiting fact was chess-state identity. Pieces and side to move are not enough: castling rights, en-passant
state, the halfmove clock, and repetition-relevant history can change the legal result. Merging positions that differ
on those fields would create an approximate algorithm with different game semantics. Under exact equality, most
apparent board transpositions disappeared.

At ordinary budgets, verified links were absent or negligible. After the first-link correction, only 0.0249% and
0.1769% of neural evaluations were avoided at 1,000 and 10,000 searches, while the graph was 8.63% and 8.28% slower.
At 30,000 and 60,000 searches, table hits rose to 2.37% and 3.46%, but avoided evaluations remained approximately
0.0001% and 0.0348%; throughput was still 7.06% and 5.76% lower. Structural counters confirmed that shared
descendants and statistics were active. A final strength match was unnecessary to answer the deployment question:
exact reuse was orders of magnitude too sparse to repay the bookkeeping cost. This is an implementation and
throughput rejection, not evidence that graph search is universally ineffective.

## Why inference caching found little to reuse

Inference caching asked a narrower question: if an identical encoded neural input reappears under the same model,
can its policy and value outputs be reused without sharing search state? Two investigations answered complementary
parts of it.

First, a real bounded, sharded cache was shared by the search threads within each self-play process. With eight
processes per GPU, three search threads per process, and capacity for 1.5 million entries per process, its hit rate
was 0.970%. Disabling the cache made game updates about 0.88% faster in the short stochastic comparison and reduced
summed worker peak memory by about 7.12%. The implemented cache consumed memory without demonstrating a speedup.

A later audit measured the upper bound available to a wider cache before building one. An unbounded tracker shared
across one search executor observed exact encoded inputs but deliberately evaluated every position. With diverse
starts and production-style progression, repeats were 1.33% at 150 searches, about 4.24% at 800, and about 3.5% in
the mixed workload; same-batch duplicates were essentially absent. Following retained trees for several moves
raised the rate only to roughly 4%. The tracker itself cost about 3.65% throughput and grew without bound. A real
cache would additionally pay for output storage, synchronization, eviction, and device transfers while retaining
fewer entries.

The wider design was declined before implementation; it was an opportunity audit, not a production cache benchmark.
Together with the implemented per-process cache, it bounds the likely benefit under this workload.

## The retained search design

The selected system returned to fixed visits, but not to a bare or naive search. It uses PUCT, reduced-parent-value
first-play urgency, root noise and temperature for self-play exploration, forced root playouts followed by pruning
of visits that should not become policy supervision, batched native inference, a 0.99 per-ply value discount, and
tree retention with 60% of visits carried across a played move. Every recorded move receives the current staged
visit cap. The current settings are summarized in Chapter 7 and defined by the public configuration [10].

Those ingredients do not form an additive ablation table. Search depth has direct strength and target-fidelity
evidence. Evaluation and self-play clearly need consistent first-play-urgency semantics. Parallel leaves have a
measured, workload-dependent quality/latency tradeoff. Root retention, forced playouts, noise, discount, and the
exact PUCT constants are retained parts of a coherent AlphaZero-style recipe, but most lack isolated final-workload
strength estimates.

The overall conclusion is correspondingly narrow. For this diverse chess self-play workload, fixed budgets were
strong, legible, and operationally robust. The adaptive systems either lacked a safe observable, optimized the wrong
proxy, or removed work that did not control elapsed training time. Exact graph and cache reuse were too sparse to
cover their overhead. Different games, a repetitive analysis service, or a non-overlapped learner could reverse
some of those economics. They do not reverse the decision supported here: under the measured workload, the simplest
auditable allocation was the most dependable one.
