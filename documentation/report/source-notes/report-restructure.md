# Topic-first report restructure

The existing draft is an evidence map, not the publication outline. It overuses internal run labels, chronology, and
compact evidence codes. The publication report will be rebuilt around technical questions and decisions after the
source dossiers pass review.

## Proposed report structure

1. **Problem, constraints, and contributions**
   - compute regime, research goals, claimed contributions, and explicit non-claims;
   - no internal run chronology.
2. **System overview**
   - publication-quality interaction SVG;
   - Python/C++ ownership, processes, data products, synchronization, and artifact lifecycle.
3. **Training and evaluation method**
   - state/action encoding, model outputs, search target construction, replay, optimization, and evaluation protocols.
4. **Search investigations**
   - fixed and mixed budgets, adaptive allocation and stopping, tree mechanics, graph search, caching, parallelism,
     and retained search design;
   - each substantial investigation receives its own subsection.
5. **Model architecture and policy representation**
   - every policy encoding/head, trunk family, context mechanism, value head, quantization-compatible block, and
     progressive-sizing decision.
6. **Data, replay, and curriculum**
   - data creation, eligibility, replay growth/reuse, selection versus weighting, starts, resignation, cuts,
     auxiliaries, and reanalysis.
7. **Inference and systems engineering**
   - native ownership, batching, topology, compiler/backend alternatives, quantization, replay/DDP, and end-to-end
     throughput accounting.
8. **Failure studies and transferable pitfalls**
   - bounded chronological narratives only where the sequence of failure, diagnosis, and repair is itself useful.
9. **Final integrated recipe**
   - the canonical living configuration explained by mechanism, without development-version identifiers.
10. **Final evaluation and learning dynamics**
    - terminal evidence, figures, matched predecessor comparisons, uncertainty, cost, and reproducibility identity.
11. **Limitations, reproducibility, and conclusions**
    - plain-language evidence boundaries and remaining uncertainty.

## Material to remove or redistribute

- Delete the lineage/version chapter as a reader-facing chapter.
- Move useful causal lessons from that chapter into the relevant technical topic or failure study.
- Keep exact source revisions, checkpoints, resumes, and discarded intervals in the reproducibility artifact manifest,
  not in the explanatory narrative.
- Replace `S`, `P`, `T`, `M`, and `O` codes with direct phrases such as “paired strength match,” “frozen-replay
  fidelity measurement,” “live-pipeline throughput benchmark,” or “implementation inspection.”
- Replace development-version comparisons with descriptive model/checkpoint labels. Internal identifiers may remain
  inside machine-readable provenance and linked source paths, but not as the report's explanatory vocabulary.
- Promote graph search, inference caching, compiler alternatives, policy representations, and other substantial
  investigations from clauses or decision-table rows to full subsections.

## Writing gate

No chapter should be rewritten into final prose until its source dossier:

1. covers every applicable item in the topic inventory;
2. has been checked against repository evidence and preserved branches;
3. identifies known evidence gaps without filling them by inference; and
4. has received project-owner feedback on missing tacit rationale.

This makes project-owner review a completeness and correctness review rather than an archaeological exercise.
