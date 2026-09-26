# Editorial rules for the technical report

These rules guide editing; they are not part of the published paper.

1. Explain the question and intuition before naming the mechanism. Explain why a choice matters before its settings or implementation.
2. Follow a connected argument: problem, proposed approach, experiment, observation, interpretation, decision. Use this as a reasoning check, not a repeated set of headings or sentence templates.
3. Introduce project-specific mechanisms before relying on them. Assume familiarity with standard machine-learning tools such as DDP; give their relevant settings without textbook explanations.
4. Give each paragraph one purpose. Each sentence should develop that purpose or bridge to the next paragraph; remove detached qualifications and inventory-like lists.
5. State results directly with enough experimental context to understand them. Put uncertainty beside the inference it limits, not as a reflexive disclaimer after every result.
6. Keep essential results and explanations in the paper. Cross-references provide detail; they must not substitute for explaining the point locally.
7. Put replication settings, archival gaps, test inventories, and operational safeguards in the relevant appendix or limitations discussion unless they explain the experiment or a failure.
8. Distinguish observations from hypotheses without legalistic language. Do not claim unmeasured gains, universal optimality, or guaranteed improvement; do not repeat the lack of complete ablations in every subsection.
9. Explain systems work through its contribution to affordable learning. Native execution, batching, data loading, and deployment matter because they supply and consume useful games efficiently.
10. Preserve approved prose unless a specific correction or consistency issue requires changing it. No run-version shorthand, arbitrary word limits, forced terseness, or fixed figure counts.
11. Review a whole chapter for prerequisites and flow, then its transitions to adjacent chapters. A polished opening does not pass a chapter whose later subsections remain unexplained.
12. Validate the rendered report: readable figures, working internal references, ordinary top/bottom floats, and no forced half-empty pages.
13. Begin each chapter with a short introduction establishing its question and scope before the first subsection. Preserve existing openings that already do this; do not add redundant roadmaps.
