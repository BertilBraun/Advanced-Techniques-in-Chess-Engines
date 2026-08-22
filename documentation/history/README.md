# Historical implementation notes

Everything below this directory is archival and non-normative. Files may name deleted modules, obsolete commands,
superseded data formats, or earlier training/evaluation architectures. Use the
[current architecture index](../architecture/README.md) for implementation decisions.

This archive retains selected evidence-bearing material:

- the R5 contract inventory;
- the V10 implementation note;
- implementation and optimization notes from earlier architectures (the pre-C++-port era lives under `pre-cpp-port/`);
- experience-based recommendations tied to those revisions.

The former trainer roadmap, clean-training-run plan, and duplicate future-work list were intentionally deleted
rather than archived because their detailed instructions could be mistaken for current authority.

`media/` holds era plots and raw measurement text referenced by the optimization notes, plus unlinked legacy
plots kept for the record: `self-play-problem.png`, `inference-server-problem.png`,
`fuse-compile-inference-speed.png`, `inference-speed-by-gpu-and-dtype.txt`, `mcts-speed-test-results.txt`.
