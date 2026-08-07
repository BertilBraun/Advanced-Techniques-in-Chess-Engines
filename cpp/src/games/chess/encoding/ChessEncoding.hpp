#pragma once

#include "games/chess/implementation/ChessAction.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "search/InferenceTypes.hpp"
#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include <torch/torch.h>

struct ChessEncoding {
    static constexpr int action_count = 1880;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = 29,
            .rows = 8,
            .columns = 8,
            .actions = action_count,
            .outcomes = 3,
        };
    }

    [[nodiscard]] static int actionId(ChessAction action, const Board &state);
    [[nodiscard]] static ChessAction decodeAction(int actionId, const Board &state);
    static void encodeInputInto(const Board &state, std::int8_t *destination);
};

struct ChessRepresentationDimensions {
    static constexpr int board_length = 8;
    static constexpr int channel_count = 29;
    static constexpr int binary_channel_count = 22;
    static constexpr int scalar_channel_count = 7;
    static constexpr int action_count = ChessEncoding::action_count;
};

static_assert(ChessRepresentationDimensions::channel_count ==
              ChessRepresentationDimensions::binary_channel_count +
                  ChessRepresentationDimensions::scalar_channel_count);

struct CompressedEncodedBoard {
    static constexpr std::size_t packed_binary_bytes =
        packed_binary_plane_bytes<ChessRepresentationDimensions::board_length,
                                  ChessRepresentationDimensions::binary_channel_count>;
    static constexpr std::size_t packed_bytes =
        packed_binary_bytes + ChessRepresentationDimensions::scalar_channel_count;

    std::array<BitBoard<ChessRepresentationDimensions::board_length>,
               ChessRepresentationDimensions::binary_channel_count>
        bits;
    std::array<std::int8_t, ChessRepresentationDimensions::scalar_channel_count> scal;

    [[nodiscard]] bool operator==(const CompressedEncodedBoard &other) const noexcept {
        return bits == other.bits && scal == other.scal;
    }
};

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

void writeTensorEncoding(const CompressedEncodedBoard &compressed, std::int8_t *destination);
void writePackedPlaneEncoding(const CompressedEncodedBoard &compressed, std::int8_t *destination);
void encodeBoardInto(const Board &board, std::int8_t *destination);
