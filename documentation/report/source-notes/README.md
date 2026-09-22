# Technical-report source dossiers

These dossiers are the factual substrate for the technical report. They are deliberately not publication prose.
Their purpose is to make missing concepts, weak causal claims, and undocumented decisions visible before narrative
editing makes them harder to find.

## Organization

The dossiers are organized by research question and system boundary, not by internal run label or calendar order.
Internal run identifiers are provenance metadata, not explanatory concepts, and must not become the report's
navigation or narrative vocabulary. When a source filename contains an internal identifier, link it under a
descriptive label.

Chronology belongs only in a bounded incident account where order is necessary to understand a failure, diagnosis,
and repair. Examples include late-game target poisoning, invalid inference conversion, and recovery after corrupted
self-play data.

## Dossiers ready for factual review

- [Search and inference](search-and-inference.md)
- [Network architecture and policy](network-architecture-and-policy.md)
- [Training, data generation, and replay](training-data-and-replay.md)
- [Runtime architecture alternatives](runtime-architecture-alternatives.md)
- [Current system architecture and figure specification](system-architecture-and-figure-dossier.md)
- [Evaluation methodology and transferable pitfalls](evaluation-and-pitfalls.md)

The independent [completeness audit](completeness-audit.md) records the evidence search, resolved omissions, remaining
questions, and terminal-archive-only work. The [topic inventory](topic-inventory.md) is the compact coverage ledger.

## Required structure for each topic

Each dossier must record:

1. **Question** — the concrete problem the work attempted to solve.
2. **Mechanism** — how every materially different approach worked.
3. **Alternatives investigated** — including implemented, measured, rejected, and superseded variants.
4. **Measurements** — the actual strength, learning, fidelity, throughput, or correctness observations, written out
   in words rather than encoded as letter grades.
5. **Decision rationale** — why an approach was retained, rejected, or left unresolved.
6. **Interactions** — what other subsystem could change the interpretation.
7. **Pitfalls and corrections** — misleading measurements, implementation defects, and lessons useful to readers.
8. **Evidence** — direct links to benchmarks, analyses, code, configurations, release tags, and archived artifacts.
9. **Unknowns** — missing raw evidence, uncontrolled comparisons, and claims that cannot currently be made.

The evidence description must say what was measured. Shorthand such as `S`, `P`, `T`, `M`, or `O` is prohibited in
the report and in these dossiers.

## Completeness gate

A dossier is ready for project-owner review only when:

- every concept in the [topic inventory](topic-inventory.md) has a substantive entry or an explicit explanation of
  why it is out of scope;
- repository documentation, benchmarks, analyses, relevant code, Git history, and preserved release branches have
  been searched;
- large investigations are not compressed into an “also tried” sentence;
- distinct implementations are separated even when they pursued the same objective;
- retained choices are distinguished from choices with isolated causal evidence;
- known evidence gaps are stated next to the affected claim;
- all local links resolve.

Only after this gate passes should the dossier be turned into report prose.

## Review workflow

The project owner reviews one dossier at a time for missing tacit knowledge and incorrect causal explanations. The
review is not a prose-editing pass. Corrections are incorporated into the dossier first; the corresponding report
chapter is written or revised only after the dossier is accepted as a complete factual map.
