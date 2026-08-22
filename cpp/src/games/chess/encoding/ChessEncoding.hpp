#pragma once

#include "games/chess/implementation/ChessAction.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "search/InferenceTypes.hpp"
#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

#include <cstddef>
#include <cstdint>
#include <torch/torch.h>

struct ChessRepresentationDimensions {
    static constexpr int board_length = 8;
    static constexpr int channel_count = 29;
    static constexpr int binary_channel_count = 22;
    static constexpr int scalar_channel_count = 7;
    static constexpr int policy_plane_count = 76;
    static constexpr int action_count = policy_plane_count * board_length * board_length;
};

static_assert(ChessRepresentationDimensions::channel_count ==
              ChessRepresentationDimensions::binary_channel_count +
                  ChessRepresentationDimensions::scalar_channel_count);

struct ChessEncoding {
    static constexpr int action_count = ChessRepresentationDimensions::action_count;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = ChessRepresentationDimensions::channel_count,
            .rows = ChessRepresentationDimensions::board_length,
            .columns = ChessRepresentationDimensions::board_length,
            .actions = ChessRepresentationDimensions::action_count,
            .outcomes = WDL_OUTPUT_SIZE,
        };
    }

    [[nodiscard]] static int actionId(ChessAction action, const Board &state);
    [[nodiscard]] static ChessAction decodeAction(int actionId, const Board &state);
    [[nodiscard]] static int mirrorActionId(int actionId);
    static void encodeInputInto(const Board &state, std::int8_t *destination);
};

using CompressedEncodedBoard = EncodedPlanes<ChessRepresentationDimensions::board_length,
                                             ChessRepresentationDimensions::binary_channel_count,
                                             ChessRepresentationDimensions::scalar_channel_count>;

[[nodiscard]] CompressedEncodedBoard encodeBoard(const Board &board);
[[nodiscard]] torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed);
