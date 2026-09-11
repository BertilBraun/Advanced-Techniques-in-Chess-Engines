# What the v34 Elo numbers mean

Date: 2026-09-11. This note audits the rating scale behind the v34 terminal results and fixes language for public
reporting. The conclusion is simple: **3037 and 3174 are valid results on this project's Stockfish-node ladder, but
they are not FIDE ratings.** The scale was designed as a historical human-strength proxy, so it supports a qualified
"superhuman" claim. It does not support a precise conversion to the modern human pool.

## Recommended public statement

> At the saved three-day checkpoint, v34 reached **3037 benchmark Elo** at 10,000 searches per move and **3174
> benchmark Elo** at 80,000 searches per move on our Stockfish 13 fixed-node ladder. The 95% paired-bootstrap
> sampling intervals are **3012--3061** and **3144--3206**, conditional on the ladder's historical SSDF-derived
> anchors. This is strong evidence of superhuman chess strength under that calibration, but the figures are not FIDE
> ratings and should not be compared point-for-point with human or other engine lists.

For a short headline, use **"superhuman strength for about $50 of training compute"**, followed nearby by the full
benchmark sentence above. Use **"strong-engine territory"** only with the qualification that current full-strength
engines remain far ahead. Avoid "3037 FIDE Elo", "3174 human Elo", and "approaching state-of-the-art engine level".

## What was measured

Both final matches used v34 generation 1465, model SHA-256
`402efb61146b5a0f569c960e1f7a7e7714ba1bb28982fcdfb7f10e8ec5a98ad6`, experiment-configuration SHA-256
`5a1194975e225d4c5ebc329ca7641b205e7cd17b88d5b4aa24679d9e6a939c9a`, source revision `a87a186e`, and 200 paired
openings (400 games). The opening manifest SHA-256 is
`40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39`. Stockfish 13 used one thread and fixed nodes
per move.

| v34 budget | Stockfish 13 opponent | W/D/L | Score | Score 95% CI | Ladder Elo | Conditional 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 searches | 100,000 nodes, anchor 3100 | 67/194/139 | 0.4100 | 0.3763--0.4438 | 3037 | 3012--3061 |
| 80,000 searches | 50,000 nodes, anchor 2960 | 252/115/33 | 0.7738 | 0.7425--0.8050 | 3174 | 3144--3206 |

The result uses the ordinary Elo performance transform

`rating = opponent_anchor + 400 * log10(score / (1 - score))`.

The intervals resample paired openings and then apply the same transform. They describe game-sampling uncertainty
against a fixed opponent anchor. They do **not** include uncertainty in the anchor scale. The 80,000-search opponent
was also weaker than ideal for precision: a 77.4% score is farther from the most informative 50% region.

At the time of this audit, the terminal JSON files are still on the evaluation node under
`/workspace/postrun/v34-terminal-g1465/final/`. They must be fetched and hash-verified before the result satisfies
the repository's archival evidence rule.

## Where the ladder's zero point comes from

The ladder anchors in [`py/src/evaluation/ladder.py`](../../py/src/evaluation/ladder.py) are readings from Marco
Meloni's 2021 Stockfish 13 fixed-node experiment. The two anchors used here are 50,000 nodes = 2960 and 100,000
nodes = 3100. These are plotted-point readings, not ratings independently measured by this project.

Meloni ran Stockfish 13 from one to 256 million nodes per move, using Cute Chess, Ordo, and the TopGM six-move
opening suite. He reports more than 110,000 games. To assign an absolute level, he matched node-limited Stockfish
against Fruit 2.2.1 and connected other node levels through asymmetric Stockfish matches. Fruit was fixed near 2830
from the Swedish Chess Computer Association (SSDF) list. Meloni explicitly says that the purpose was to compare with
the human FIDE scale, "or, at least, ... try to do so." This wording matters: the source presents a proxy rather than
a direct human rating ([Meloni's method and caveat](https://www.melonimarco.it/en/2021/03/08/stockfish-and-lc0-test-at-different-number-of-nodes/)).

The chain is therefore:

`v34 match score -> fixed-node Stockfish 13 anchor -> Meloni node curve -> Fruit 2.2.1 -> historical SSDF level`.

Every link after the match score adds assumptions. In particular, this project uses different openings, game
adjudication, hardware, and candidate search. The node-limited Stockfish opponent is reproducible, which makes the
ladder useful for comparisons within this project; reproducibility does not make its absolute zero point exact.

## SSDF and human Elo

SSDF is the most relevant external list because it is the source of Meloni's anchor. Its final published list used
tournament time control, 40 moves in two hours followed by 20 moves per subsequent hour, and joined old and new
program/hardware combinations through a connected game pool. The list asks that every quotation retain games and
error margins. In the 2023-12-31 list, Stockfish 13 on a Ryzen 7 1800X was 3569 (+35/-32 over 480 games), and the
leader was Lc0 at 3586 (+30/-28 over 600 games) ([official SSDF list](https://ssdf.bosjo.net/list.htm)). These are
ratings for those exact program, hardware, and time-control combinations.

SSDF did have a bridge to humans, unlike most engine-only lists, but it was old and limited. SSDF chairman Thoralf
Karlsson explained that the 1999 level rested on 337 serious tournament games against Swedish players from
1987--1991. His own strict interpretation was that the list measured relative program strength under SSDF's test
method; he said that correspondence to the rating each program would earn in hundreds of human games was unknown
([contemporaneous SSDF commentary](https://www.oocities.org/marochess/ssdf/1999/ssdf9902.htm)).

In 2000, SSDF revisited the offset using more recent human-computer results and lowered the whole list by 100 points.
Karlsson wrote that the aim was for top programs *as a group* to correlate better with human Elo, while the
individual correlation would probably never be established. He also identified possible pool-spreading, human
adaptation, selection bias, and Swedish-versus-international rating differences
([Karlsson's 2000 statement, reproduced here](https://groups.google.com/g/rec.games.chess.computer/c/mJ7s2zuK0LA)).

This history makes SSDF better motivated as a human proxy than an arbitrarily shifted engine list. It does not make
the top of its 2023 list a continuation of the modern FIDE pool with known error. The human bridge was made with much
weaker engines decades earlier, and the list then extrapolated through engine-versus-engine games.

## Why FIDE, CCRL, CEGT, and SSDF numbers differ

Elo determines expected score from a **rating difference**. A connected pool still needs an arbitrary additive
origin. Two disconnected pools can use the same 400-point logistic slope and display different numbers for the same
playing strength. Time control, hardware, openings, tablebases, pondering, adjudication, opponent selection, and
rating software can also change measured differences rather than merely shift the origin.

| List | Population and protocol | Relevant published number | What it can establish |
| --- | --- | ---: | --- |
| This project | v34 against one-thread Stockfish 13 at fixed nodes per move; paired four-move openings | 3037 at 10k; 3174 at 80k | v34's performance against the specified node-limited opponents, on Meloni's offset |
| SSDF | Engine/hardware pairs; 40 moves/120 min; mixed historical hardware | Stockfish 13: 3569; leader: 3586 | Relative strength under SSDF conditions, with a historical and approximate human bridge |
| CCRL 40/15 | Engines; equivalent to 40 moves/15 min on i7-4770K; generic book; tablebases; BayesElo | Stockfish 13: 3572 (1 CPU) or 3613 (4 CPU); Stockfish 18: 3649 | Relative engine strength under CCRL conditions |
| CEGT 40/20 | Engines; current list described by CEGT as 10 min + 5 sec; one-CPU entry cited here | Stockfish 18: 3612 in February 2026 | Relative engine strength under CEGT conditions |
| FIDE standard | Rated humans in approved over-the-board standard events | Current listed leader: 2823 | Relative human tournament performance in the FIDE pool |

CCRL describes its purpose as comparing chess programs and normalises time to an i7-4770K. It permits generic
books up to 12 moves and three-to-six-piece tablebases, and uses Stockfish 10 to calibrate time across testers
([CCRL conditions](https://computerchess.org.uk/4040/about.html)). Its 2026 list contains more than 2.4 million
games, yet Stockfish 13 itself differs by 41 points between the one-CPU and four-CPU entries
([CCRL complete list](https://computerchess.org.uk/4040/rating_list_all.html)). Its faster list places Stockfish 13
higher again. These are protocol-dependent measurements, not contradictory estimates of one universal Elo.

The CEGT team's February 2026 update reports Stockfish 18 at 3612 (+11/-11 over 2900 games) in its 40/20 list
([CEGT team update](https://talkchess.com/viewtopic.php?t=86036)). CCRL 40/15 currently gives the same release 3649
(+12/-12). The 37-point difference is small here, but it has no fixed conversion value: list composition and testing
conditions evolve independently.

FIDE maintains a separate human standard, rapid, and blitz rating for each player; its database currently lists
Magnus Carlsen first in standard chess at 2823, rapid at 2803, and blitz at 2860
([FIDE ratings database](https://ratings.fide.com/)). FIDE's own expected-score table also uses a 400-point Elo-like
scale, but sharing the formula does not connect its origin to an engine pool
([FIDE rating regulations](https://handbook.fide.com/chapter/B02RBRegulations2024)). Online chess-site ratings are
disconnected pools as well and need their own empirical bridge.

## Can 3037 or 3174 be converted to human Elo?

There is no defensible statistical conversion interval from the available evidence. A valid conversion would need
enough games between representative modern humans and the evaluated v34 configuration, at a defined human time
control and without selection effects. This project has none. The old SSDF bridge supplies an approximate origin,
but no current estimate of its systematic error at 3000+ and no basis for turning that error into a confidence
interval.

The central values are 214 and 351 points above the current top FIDE standard rating. That arithmetic describes the
two displayed scales; it is not an expected-score prediction against Carlsen. Even subtracting SSDF's historical
100-point correction would leave both central estimates above 2823, but that correction was an offset decision in
2000, not a bound on present error.

The responsible human comparison is therefore qualitative:

- **Defensible:** "above the top-human region on a historical SSDF-derived calibration"; "strong evidence of
  superhuman strength under this benchmark's calibration."
- **Too strong without human games:** "3037/3174 FIDE Elo"; "would score as a 3037/3174 human"; "proved stronger
  than every human."

If a numeric human claim becomes important, the measurement should be designed directly: pre-register a time
control, hardware and interface; recruit a connected set of titled human opponents; use balanced openings and
colours; and report scores and opponents before fitting a human-pool performance rating. A handful of exhibition
games would demonstrate play, not calibrate 3000-level Elo.

## What “engine level” can mean here

Every evaluated participant is already an engine, so "engine level" has no technical threshold. The SSDF-derived
numbers place v34 near historical engines in the low-3000 region, and its direct results show competitiveness with a
severely node-limited Stockfish 13. That supports **"strong-engine territory."**

It does not support **"approaching current top-engine strength."** The last SSDF leader is about 400 points above
v34's higher-search estimate, and current CCRL/CEGT leaders are around 3600 on their own scales. Those gaps cannot be
converted exactly across protocols, but they are too large to present v34 as near Stockfish 18. A direct match at a
defined equal wall-clock budget would be the clean way to make that comparison.

## Reporting rules for this repository

1. Call the measure **"Stockfish-13 ladder Elo"**, **"SSDF-derived benchmark Elo"**, or **"benchmark Elo"** on
   first use. Never label it simply "FIDE Elo."
2. Keep the model search budget, `parallel_searches`, Stockfish version and node count beside every headline result.
3. Report W/D/L, game count, paired-opening score interval, and transformed Elo interval. State that the Elo interval
   is conditional on fixed anchors.
4. Use exact values in benchmark tables. Round prose to about 3040 at 10k and 3170 at 80k so the writing does not
   imply that anchor error is smaller than one Elo point.
5. Compare generations and recipes on this project's unchanged ladder. Compare humans or external engine lists only
   through explicit bridge evidence.
6. Describe the three-day checkpoint as the cost/strength result even if later training produces a slightly stronger
   checkpoint; state which checkpoint each evaluation used.

