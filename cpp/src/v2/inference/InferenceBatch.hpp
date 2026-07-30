#pragma once

#include "inference/InferenceTypes.hpp"

#include <concepts>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace az::v2::inference {

template <typename Encoding> struct InferenceBatch {
    std::vector<InferenceRequest<Encoding>> requests;
};

struct InferenceBatchResult {
    std::vector<InferenceResult> results;
};

template <typename Evaluator, typename Encoding>
concept BatchEvaluator = requires(Evaluator evaluator, const InferenceBatch<Encoding> &batch) {
    { evaluator.evaluate_batch(batch) } -> std::same_as<InferenceBatchResult>;
};

template <typename Encoding> void validate_batch(const InferenceBatch<Encoding> &batch) {
    if (batch.requests.empty()) {
        throw std::invalid_argument("inference batch must not be empty");
    }
    std::unordered_set<std::uint64_t> request_ids;
    request_ids.reserve(batch.requests.size());
    for (const InferenceRequest<Encoding> &request : batch.requests) {
        if (request.action_count <= 0) {
            throw std::invalid_argument("batch request action_count must be positive");
        }
        if (!request_ids.insert(request.request_id).second) {
            throw std::invalid_argument("inference batch request IDs must be unique");
        }
    }
}

template <typename Encoding>
void validate_batch_result(const InferenceBatch<Encoding> &batch,
                           const InferenceBatchResult &batch_result) {
    validate_batch(batch);
    if (batch_result.results.size() != batch.requests.size()) {
        throw std::invalid_argument(
            "inference batch result cardinality must match request cardinality");
    }
    for (std::size_t index = 0; index < batch.requests.size(); ++index) {
        const InferenceRequest<Encoding> &request = batch.requests[index];
        const InferenceResult &result = batch_result.results[index];
        validate_result(result, request.request_id, request.action_count);
    }
}

} // namespace az::v2::inference
