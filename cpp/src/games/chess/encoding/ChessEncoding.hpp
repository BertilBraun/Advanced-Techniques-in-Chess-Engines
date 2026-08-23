#pragma once

#include "games/chess/implementation/ChessAction.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "search/InferenceTypes.hpp"
#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

#include <cstddef>
#include <cstdint>
#include <torch/torch.h>

enum class ChessActionEncoding {
    Reduced,
    PolicyPlane,
};

inline constexpr ChessActionEncoding chessActionEncoding = ChessActionEncoding::Reduced;

struct ChessRepresentationDimensions {
    static constexpr int boardLength = 8;
    static constexpr int channelCount = 29;
    static constexpr int binaryChannelCount = 22;
    static constexpr int scalarChannelCount = 7;
    static constexpr int policyPlaneCount = 76;
    static constexpr int policyPlaneActionCount = policyPlaneCount * boardLength * boardLength;
    static constexpr int reducedActionCount = 1880;
    static constexpr int actionCount = chessActionEncoding == ChessActionEncoding::Reduced
                                           ? reducedActionCount
                                           : policyPlaneActionCount;
};

static_assert(ChessRepresentationDimensions::channelCount ==
              ChessRepresentationDimensions::binaryChannelCount +
                  ChessRepresentationDimensions::scalarChannelCount);

struct ChessEncoding {
    static constexpr int actionCount = ChessRepresentationDimensions::actionCount;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = ChessRepresentationDimensions::channelCount,
            .rows = ChessRepresentationDimensions::boardLength,
            .columns = ChessRepresentationDimensions::boardLength,
            .actions = ChessRepresentationDimensions::actionCount,
            .outcomes = WDL_OUTPUT_SIZE,
        };
    }

    [[nodiscard]] static int actionId(ChessAction action, const Board &state);
    [[nodiscard]] static ChessAction decodeAction(int actionId, const Board &state);
    [[nodiscard]] static int mirrorActionId(int actionId);
    static void encodeInputInto(const Board &state, std::int8_t *destination);
};

using CompressedEncodedBoard = EncodedPlanes<ChessRepresentationDimensions::boardLength,
                                             ChessRepresentationDimensions::binaryChannelCount,
                                             ChessRepresentationDimensions::scalarChannelCount>;

[[nodiscard]] CompressedEncodedBoard encodeBoard(const Board &board);
[[nodiscard]] torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed);
