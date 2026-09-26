#pragma once

#include "games/chess/implementation/ChessAction.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "search/InferenceTypes.hpp"
#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <torch/torch.h>

enum class ChessActionEncoding {
    Reduced,
    PolicyPlane,
};

inline constexpr ChessActionEncoding chessActionEncoding = ChessActionEncoding::Reduced;

// Two input layouts exist side by side. The project's own 52 planes feed its networks; Lc0's
// INPUT_CLASSICAL_112_PLANE layout feeds an Lc0 teacher. Both encoders are always compiled, so a
// teacher can be queried while a project network is trained. CHESS_LC0_INPUT only chooses which one
// the search and the inference pipeline use.
#ifdef CHESS_LC0_INPUT
inline constexpr bool chessSearchUsesLc0Input = true;
#else
inline constexpr bool chessSearchUsesLc0Input = false;
#endif

struct ProjectInputDimensions {
    static constexpr int binaryChannelCount = 40;
    static constexpr int scalarChannelCount = 12;
    static constexpr int channelCount = binaryChannelCount + scalarChannelCount;
};

struct Lc0InputDimensions {
    static constexpr int historyPositionCount = 8;
    static constexpr int planesPerHistoryPosition = 13;
    // 8 x 13 history, 4 castling, side to move; the scalars are rule 50, zeros and ones.
    static constexpr int binaryChannelCount = historyPositionCount * planesPerHistoryPosition + 5;
    static constexpr int scalarChannelCount = 3;
    static constexpr int channelCount = binaryChannelCount + scalarChannelCount;
};
static_assert(Lc0InputDimensions::channelCount == 112);

using SearchInputDimensions =
    std::conditional_t<chessSearchUsesLc0Input, Lc0InputDimensions, ProjectInputDimensions>;

struct ChessRepresentationDimensions {
    static constexpr int boardLength = 8;
    static constexpr int channelCount = SearchInputDimensions::channelCount;
    static constexpr int binaryChannelCount = SearchInputDimensions::binaryChannelCount;
    static constexpr int scalarChannelCount = SearchInputDimensions::scalarChannelCount;
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

using ProjectEncodedBoard = EncodedPlanes<ChessRepresentationDimensions::boardLength,
                                          ProjectInputDimensions::binaryChannelCount,
                                          ProjectInputDimensions::scalarChannelCount>;
using Lc0EncodedBoard = EncodedPlanes<ChessRepresentationDimensions::boardLength,
                                      Lc0InputDimensions::binaryChannelCount,
                                      Lc0InputDimensions::scalarChannelCount>;
using CompressedEncodedBoard = EncodedPlanes<ChessRepresentationDimensions::boardLength,
                                             ChessRepresentationDimensions::binaryChannelCount,
                                             ChessRepresentationDimensions::scalarChannelCount>;

[[nodiscard]] ProjectEncodedBoard encodeProjectBoard(const Board &board);
[[nodiscard]] Lc0EncodedBoard encodeLc0Board(const Board &board);
[[nodiscard]] CompressedEncodedBoard encodeBoard(const Board &board);
[[nodiscard]] torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed);
