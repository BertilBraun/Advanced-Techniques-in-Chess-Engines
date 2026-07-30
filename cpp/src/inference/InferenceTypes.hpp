#pragma once

#include "common.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace az::inference {

template <typename Encoding> struct InferenceRequest {
    uint64 requestId;
    Encoding encoding;
    int32 actionCount;
};

struct InferenceResult {
    uint64 requestId;
    std::vector<double> policy;
    double value;
};

template <typename Evaluator, typename Encoding>
concept SynchronousEvaluator =
    requires(Evaluator evaluator, const InferenceRequest<Encoding> &request) {
        { evaluator.evaluate(request) } -> std::same_as<InferenceResult>;
    };

inline void validateResult(const InferenceResult &result, uint64 expected_request_id,
                           int32 actionCount) {
    if (result.requestId != expected_request_id) {
        throw std::invalid_argument("inference result requestId does not match its request");
    }
    if (actionCount <= 0 || result.policy.size() != static_cast<std::size_t>(actionCount)) {
        throw std::invalid_argument("inference policy size does not match actionCount");
    }
    if (!std::isfinite(result.value) || result.value < -1.0 || result.value > 1.0) {
        throw std::invalid_argument("inference value must be finite and in [-1, 1]");
    }
    for (const double probability : result.policy) {
        if (!std::isfinite(probability) || probability < 0.0) {
            throw std::invalid_argument("inference policy entries must be finite and non-negative");
        }
    }
}

} // namespace az::inference
