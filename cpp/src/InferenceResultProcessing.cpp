#include "InferenceResultProcessing.hpp"

#include "MoveEncoding.hpp"

InferenceResult processInferenceResult(const float *policyData, const float *outcomeData,
                                       const Board &board) {
    const float win = outcomeData[static_cast<size_t>(WdlIndex::Win)];
    const float draw = outcomeData[static_cast<size_t>(WdlIndex::Draw)];
    const float loss = outcomeData[static_cast<size_t>(WdlIndex::Loss)];
    if (!std::isfinite(win) || !std::isfinite(draw) || !std::isfinite(loss) || win < 0.0F ||
        draw < 0.0F || loss < 0.0F || std::abs(win + draw + loss - 1.0F) > 1e-2F) {
        throw std::runtime_error("Inference model WDL output must be three probabilities");
    }

    return {filterPolicyThenGetMoveScores(policyData, &board), {win, draw, loss}};
}
