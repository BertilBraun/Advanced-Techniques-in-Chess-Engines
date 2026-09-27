# Evidence language for the report

The report will describe evidence in ordinary language. Readers should not have to memorize a code system before
they can judge a claim.

## Preferred descriptions

Use the narrowest accurate phrase:

- **Paired strength match:** two artifacts played under the same openings, colors, opponent, search budget, and
  adjudication protocol, with uncertainty reported.
- **Shared-state online comparison:** alternatives began from the same checkpoint, optimizer, replay, and elapsed
  state, then generated their own subsequent data.
- **Independent online observation:** learning behavior observed during a run without a matched counterfactual.
- **Frozen-replay training comparison:** alternatives trained against the same immutable data and sample identities;
  useful for optimization and target fit, but not a direct self-play strength result.
- **Policy-fidelity measurement:** policy cross-entropy, KL divergence, top-action agreement, or agreement with a
  deeper search on fixed positions.
- **Model-core throughput benchmark:** isolated forward-pass rate for a fixed shape and backend.
- **Search throughput benchmark:** completed simulations or searched positions under a specified concurrency and
  batch regime.
- **Live-pipeline throughput measurement:** completed games, admitted replay rows, training steps, or generations per
  wall-clock time with the production pipeline active.
- **Correctness or mechanics test:** evidence that a component behaves as specified, not that it improves learning or
  playing strength.
- **Implementation inspection:** a claim derived from configuration, code, or artifact structure.
- **External rationale:** a published result or practitioner explanation from another game, engine, or compute
  regime; its transfer assumptions must be stated.
- **Uncontrolled historical observation:** potentially useful diagnostic evidence with coupled changes or incomplete
  provenance; never present it as a causal effect.

## Required qualifiers

Every quantitative claim should make the following evident from the sentence or nearby context:

- what object was measured;
- on which data, checkpoint, hardware, and workload;
- what the comparison was;
- whether the outcome was strength, learning, fidelity, throughput, mechanics, or rationale;
- whether important variables were matched;
- where the underlying evidence is preserved.

## Internal identifiers

Internal run labels do not explain a result. Replace them with a descriptive subject, for example:

- “the completed predecessor model”;
- “the fixed-budget shared-state arm”;
- “the convolutional model with a direct policy-plane head”;
- “the final continued training lineage”;
- “the checkpoint immediately before the serving failure.”

Exact run names, generation numbers, commits, and artifact hashes belong in linked provenance or the reproducibility
manifest when needed to locate evidence. They should not be used as conceptual labels in the report.
