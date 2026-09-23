# Final chess evidence index, 2026-09-23

This directory is the compact, repository-tracked index for the final chess result. The large evidence archives stay
outside Git at `C:\Projects\AZ\.codex-diagnostics\final-2026-09-23\`. The hashes below identify the local copies
that were inspected for this documentation pass.

The compact tables intentionally expose evidence status rather than making every supplied value look equally frozen:

- [`evaluation-results.csv`](evaluation-results.csv) contains the teacher matrix and parallel-search sweep;
- [`student-results.csv`](student-results.csv) separates the completed first student from the unfinished longer run;
- [`training-summary.csv`](training-summary.csv) distinguishes archive-backed checkpoint facts, derived values, and
  ladder facts that still require reconciliation; and
- [`ladder-elo-export.json`](ladder-elo-export.json) is the delivered cross-campaign ladder input, SHA-256
  `136ded703b8b81fec915839577e5f82fb859b7e19aabb15d8af9f55d29aa58dc`.

## Evidence freeze

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `evidence-small.tgz` | 175,856,846 | `4fb8a5941e002d6d8f3186b998ae4a38f3eddd885ef81da3092558c9f556460c` |
| `evidence-tensorboard.tgz` | 143,576,346 | `72936c566026c3065cc07c9ce5d234a89dcb4cefd0efd035cea0db9e44e1c910` |
| `evidence-logs.tgz` | 2,813,105 | `7af5705a92a7bfbaabe5016e9830c68a8b80cd9feaa8d1a84985435c705f8f82` |
| `evidence-provenance.tgz` | 3,172,422 | `06a6e071fa41c840407e498b130e868e3aa68620d942450625d6aefe8474ffd7` |

The archive pull predates several evaluations reported in the operator recap. Those completed results are retained in
[`evaluation-results.csv`](evaluation-results.csv) with `awaiting_refetch` status. They are usable as reported results,
but publication remains gated on fetching their result directories and checking their manifests and checksums. The
second, longer student training and its queued evaluations were still running and are not reported as completed.

The later ladder export is now preserved, but it does not exactly match the recap's summary count. Its final
multi-rung series contains 229 observations through 274,800 stitched seconds (76.33 hours), while the recap reports
216 observations over 3.0 days. The selected checkpoint remains at 60 accepted-lineage hours and the 2,407.6 peak is
present in the export. The count and duration difference must be explained before the figure caption is finalized;
neither source is silently substituted for the other.

## Selected checkpoint

| Item | Value |
| --- | --- |
| Generation | 1026 |
| Architecture | 14 residual blocks, 160 channels, scaled post-activation, global pooling every second block |
| Policy | chess from-to attention, key size 128 |
| Parameters | 6,315,378 |
| Deployment | pre-fold INT8 QAT through TensorRT, batch 64 |
| Checkpoint manifest | `final-model/checkpoint_1026.json`, SHA-256 `351ac840a4622295c97aec44ab498764aceca519b4eccf6d39d42170a772cb99` |
| Training model | SHA-256 `c92a363b041a18d0ef93b852ac1c6d58716ae9a22b4e62d543de297c4ec5f904` |
| INT8 ONNX | SHA-256 `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` |
| QAT state | SHA-256 `c41c955f8d306844201b9cb1ffae7916963cb1f1c96a0c69c4c2f779227cfda9` |
| TensorRT engine | SHA-256 `357652b119b4e4570127587bf0a04757ece6c01dd03abe3a0a905254249d84a5` |
| Completed optimizer steps in QAT state | 408,500 |

The stitched multi-rung ladder reached its strongest window at approximately generations 1020--1080. Generation
1026 was selected because it was the last checkpoint in that window retained with a complete model, optimizer, QAT,
ONNX, and TensorRT evidence set. The later, larger network reached parity but did not establish a stronger plateau.

## Evaluation protocol

All terminal matrix matches used Stockfish 13 with one thread and 1,024 MiB hash, 50 opening pairs (100 games), and
the same 8-move opening manifest (`490425ed0f466f55d6f4470d915ef99ce87f2a20be73648790fe1cfdcac7fd0b`).
Each opening was played once from each colour. Ratings use the established fixed-node Stockfish anchor curve. For each
model budget the headline is the opponent rung whose score was closest to 0.500; both rungs remain in the CSV so the
easy-rung bias is visible.

The terminal curve mixes search parallelism: one parallel search at 100 and 1,000 searches, four at 10,000, and 16
at 100,000. Consequently it is a measured operating curve, not a pure comparison of search budget with parallelism
held constant. The separate 1,000-search sweep quantifies this confound.

The archived result manifests directly verify eleven matches. Their evaluation source revisions are
`ea80f09919290c613011893797413268673cbe0e` for the two 10,000-search matches and
`51a7843722f4d477cb63f4f6d5a76a3b25c07f17` for the later captured sweep. Exact internal run identifiers remain
provenance metadata and are not
reader-facing labels.

## Cost boundary

The reported `$43.20` is `60 h x $0.72/h`: effective accepted-lineage time through the selected checkpoint on the
training node. It is not total rental spend and excludes discarded branches, later capacity-growth work,
distillation, terminal evaluation, and other project compute. Actual end-to-end spend has not yet been reconciled.

## Required follow-up

- Re-fetch the final evaluation directory and checksum the policy-only, 100,000-versus-200,000, and student result
  directories created after the evidence pull.
- Record the completed longer-student result separately; do not overwrite the first student experiment.
- Reconcile final games, admitted positions, replay occupancy, discarded work, stage time, and actual total spend
  from the provenance and logs archives.
- Reconcile the ladder export's 229 points and 76.33-hour endpoint with the recap's 216 points and 3.0-day summary,
  then generate the cross-campaign and training-dynamics figures.
