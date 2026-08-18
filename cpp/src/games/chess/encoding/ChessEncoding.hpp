#pragma once

#include "games/chess/implementation/ChessAction.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "search/InferenceTypes.hpp"
#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

#include <cstddef>
#include <cstdint>
#include <array>

#include <torch/torch.h>

struct ChessRepresentationDimensions {
    static constexpr int board_length = 8;
    static constexpr int channel_count = 29;
    static constexpr int binary_channel_count = 22;
    static constexpr int scalar_channel_count = 7;
    static constexpr int action_count = 1880;
};

static_assert(ChessRepresentationDimensions::channel_count ==
              ChessRepresentationDimensions::binary_channel_count +
                  ChessRepresentationDimensions::scalar_channel_count);

struct ChessEncoding {
    static constexpr int action_count = ChessRepresentationDimensions::action_count;
    static constexpr int policy_plane_count = 76;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = ChessRepresentationDimensions::channel_count,
            .rows = ChessRepresentationDimensions::board_length,
            .columns = ChessRepresentationDimensions::board_length,
            .actions = ChessRepresentationDimensions::action_count,
            .outcomes = 3,
        };
    }

    [[nodiscard]] static int actionId(ChessAction action, const Board &state);
    [[nodiscard]] static ChessAction decodeAction(int actionId, const Board &state);
    [[nodiscard]] static int mirrorActionId(int actionId);
    [[nodiscard]] static const std::array<int, action_count> &policyPlaneIndices();
    static void encodeInputInto(const Board &state, std::int8_t *destination);
};

using CompressedEncodedBoard = EncodedPlanes<ChessRepresentationDimensions::board_length,
                                             ChessRepresentationDimensions::binary_channel_count,
                                             ChessRepresentationDimensions::scalar_channel_count>;

struct BoardFingerprint {
    std::uint64_t first;
    std::uint64_t second;

    [[nodiscard]] bool operator==(const BoardFingerprint &other) const noexcept {
        return first == other.first && second == other.second;
    }
};

struct BoardFingerprintHash {
    [[nodiscard]] std::size_t operator()(const BoardFingerprint &fingerprint) const noexcept;
};

[[nodiscard]] CompressedEncodedBoard encodeBoard(const Board &board);
[[nodiscard]] BoardFingerprint fingerprintBoard(const CompressedEncodedBoard &compressed);
[[nodiscard]] torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed);
