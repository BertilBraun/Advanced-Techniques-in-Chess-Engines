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

// The Lc0 teacher branch encodes Lc0's own 112-plane input instead of this project's 52 planes:
// 8 history positions x 13 planes, then castling, side to move, rule 50 and the two constant planes.
// The project's own networks cannot run against this layout; that is deliberate and this branch is
// not merged.
struct ChessRepresentationDimensions {
    static constexpr int boardLength = 8;
    static constexpr int historyPositionCount = 8;
    static constexpr int planesPerHistoryPosition = 13;
    static constexpr int channelCount = 112;
    static constexpr int binaryChannelCount =
        historyPositionCount * planesPerHistoryPosition + 5;
    static constexpr int scalarChannelCount = 3;
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
