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

The current Go pybind adapter implements that evaluator as `PythonGoEvaluator`.
C++ retains ownership of the search and calls a Python callback for each leaf;
the runtime's single inference broker per device batches callbacks arriving from
concurrent searches. Unlike the legacy `DirectInferencePipeline`, it does not
run a LibTorch model or batching thread inside C++. This keeps the v2 search
independent of model and game-specific tensor implementations, but adds one
pybind/GIL crossing per inference request. Measure that overhead before replacing
the adapter. A future native queue should preserve these typed requests, ordered
results, validation rules, and single model owner.
