#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace az::v2::inference {

template <typename Encoding> struct InferenceRequest {
    std::uint64_t request_id;
    Encoding encoding;
    std::int32_t action_count;
};

struct InferenceResult {
    std::uint64_t request_id;
    std::vector<double> policy;
    double value;
};

template <typename Evaluator, typename Encoding>
concept SynchronousEvaluator =
    requires(Evaluator evaluator, const InferenceRequest<Encoding> &request) {
        { evaluator.evaluate(request) } -> std::same_as<InferenceResult>;
    };

inline void validate_result(const InferenceResult &result, std::uint64_t expected_request_id,
                            std::int32_t action_count) {
    if (result.request_id != expected_request_id) {
        throw std::invalid_argument("inference result request_id does not match its request");
    }
    if (action_count <= 0 || result.policy.size() != static_cast<std::size_t>(action_count)) {
        throw std::invalid_argument("inference policy size does not match action_count");
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

} // namespace az::v2::inference
