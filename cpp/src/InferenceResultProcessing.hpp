#pragma once

#include "InferenceClientTypes.hpp"

[[nodiscard]] InferenceResult processInferenceResult(const float *policyData,
                                                     const float *outcomeData, const Board &board);
