#pragma once

#include "common.hpp"
#include "games/chess/ChessPolicy.hpp"

#include <array>
#include <optional>
#include <position.h>
#include <string>
#include <vector>

namespace az::games::chess {

struct ChessEncoding;

enum class Player : int8 { White = 1, Black = -1 };

enum class TerminationReason : int8 {
    Ongoing = 0,
    Checkmate = 1,
    Stalemate = 2,
    ThreefoldRepetition = 3,
    HalfmoveRule = 4,
    InsufficientMaterial = 5,
    SafetyPlyCap = 6,
};

struct ChessRules {
    std::string startingFen;
    int32 halfmoveDrawPlyCount;
    int32 safetyPlyCap;

    [[nodiscard]] bool operator==(const ChessRules &) const = default;
};

struct TerminalResult {
    TerminationReason reason;
    std::optional<Player> winner;

    [[nodiscard]] bool operator==(const TerminalResult &) const = default;
};

class ChessState {
public:
    using action_type = int32;
    using player_type = Player;
    using termination_reason_type = TerminationReason;
    using terminal_result_type = TerminalResult;
    using encoding_type = ChessEncoding;

    explicit ChessState(ChessRules rules);
    ChessState(const ChessState &other);
    ChessState &operator=(const ChessState &other);
    ChessState(ChessState &&) noexcept = default;
    ChessState &operator=(ChessState &&) noexcept = default;

    [[nodiscard]] const ChessRules &rules() const;
    [[nodiscard]] int32 actionCount() const;
    [[nodiscard]] std::vector<int32> legalActions() const;
    [[nodiscard]] bool isLegal(int32 action) const;
    void apply(int32 action);
    [[nodiscard]] Player currentPlayer() const;
    [[nodiscard]] int32 ply() const;
    [[nodiscard]] int32 repetitionCount() const;
    [[nodiscard]] TerminationReason terminationReason() const;
    [[nodiscard]] bool isTerminal() const;
    [[nodiscard]] TerminalResult terminalResult() const;
    [[nodiscard]] ChessEncoding canonicalEncoding() const;
    [[nodiscard]] uint64 stateHash() const;
    [[nodiscard]] const Stockfish::Position &position() const;
    [[nodiscard]] bool operator==(const ChessState &other) const;

private:
    inline static constexpr uint16 MAXIMUM_HISTORY_POSITIONS = 151;

    static void initializeStockfish();
    static void validateRules(const ChessRules &rules);
    [[nodiscard]] std::vector<Stockfish::Move> legalMoves() const;
    [[nodiscard]] bool hasInsufficientMaterial() const;
    [[nodiscard]] uint8 castlingRightsMask() const;
    void appendHistory(bool resetHistory);

    ChessRules _rules;
    Stockfish::Position _position;
    std::array<uint64, MAXIMUM_HISTORY_POSITIONS> _repetitionKeys{};
    uint16 _historyCount = 0;
    int32 _ply = 0;
};

} // namespace az::games::chess
