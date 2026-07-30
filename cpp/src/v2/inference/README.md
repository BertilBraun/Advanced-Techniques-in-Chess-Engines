# Inference contract

Every request carries its runtime action count and a request ID. A batch is
nonempty, all action counts are positive, and request IDs are unique within that
batch. Results preserve request order: result `i` must have the same request ID
as request `i`. Validation checks cardinality, association, policy shape, policy
mass entries, and value range before a consumer uses the batch.

The fixed search currently consumes the synchronous evaluator concept. These
same request and result types define the batch boundary so a later native
batcher can aggregate calls without changing policy masking, value backup, or
simulation accounting.
