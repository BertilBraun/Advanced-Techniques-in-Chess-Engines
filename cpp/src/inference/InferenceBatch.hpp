#pragma once

#include "common.hpp"
#include "inference/InferenceTypes.hpp"

#include <concepts>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace az::inference {

template <typename Encoding> struct InferenceBatch {
    std::vector<InferenceRequest<Encoding>> requests;
};

struct InferenceBatchResult {
    std::vector<InferenceResult> results;
};

template <typename Evaluator, typename Encoding>
concept BatchEvaluator = requires(Evaluator evaluator, const InferenceBatch<Encoding> &batch) {
    { evaluator.evaluateBatch(batch) } -> std::same_as<InferenceBatchResult>;
};

template <typename Encoding> void validateBatch(const InferenceBatch<Encoding> &batch) {
    if (batch.requests.empty()) {
        throw std::invalid_argument("inference batch must not be empty");
    }
    std::unordered_set<uint64> requestIds;
    requestIds.reserve(batch.requests.size());
    for (const InferenceRequest<Encoding> &request : batch.requests) {
        if (request.actionCount <= 0) {
            throw std::invalid_argument("batch request actionCount must be positive");
        }
        if (!requestIds.insert(request.requestId).second) {
            throw std::invalid_argument("inference batch request IDs must be unique");
        }
    }
}

template <typename Encoding>
void validateBatchResult(const InferenceBatch<Encoding> &batch,
                         const InferenceBatchResult &batch_result) {
    validateBatch(batch);
    if (batch_result.results.size() != batch.requests.size()) {
        throw std::invalid_argument(
            "inference batch result cardinality must match request cardinality");
    }
    for (std::size_t index = 0; index < batch.requests.size(); ++index) {
        const InferenceRequest<Encoding> &request = batch.requests[index];
        const InferenceResult &result = batch_result.results[index];
        validateResult(result, request.requestId, request.actionCount);
    }
}

} // namespace az::inference
