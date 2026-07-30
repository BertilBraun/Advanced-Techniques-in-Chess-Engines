#include "games/go/GoState.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace az::v2::games::go {
namespace {

constexpr uint64 FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64 FNV_PRIME = 1099511628211ULL;

void hashByte(uint64 &hash, uint8 value) {
    hash ^= value;
    hash *= FNV_PRIME;
}

void hashInt(uint64 &hash, int32 value) {
    const auto unsignedValue = static_cast<uint32>(value);
    for (int32 shift = 0; shift < 32; shift += 8) {
        hashByte(hash, static_cast<uint8>((unsignedValue >> shift) & 0xffU));
    }
}

} // namespace

std::optional<Player> AreaScore::winner() const {
    if (blackTwice > whiteTwice) {
        return Player::Black;
    }
    if (whiteTwice > blackTwice) {
        return Player::White;
    }
    return std::nullopt;
}

GoState::GoState(GoRules rules)
    : _rules(rules), _board(), _currentPlayer(Player::Black), _ply(0), _consecutivePasses(0),
      _terminationReason(TerminationReason::Ongoing), _positionHistory() {
    validateRules(rules);
    const auto pointCount =
        static_cast<std::size_t>(rules.boardSize) * static_cast<std::size_t>(rules.boardSize);
    _board.assign(pointCount, Stone::Empty);
    _positionHistory.push_back(_board);
}

GoState::GoState(GoRules rules, std::vector<Stone> board, Player currentPlayer, int32 ply,
                 int32 consecutivePasses, TerminationReason terminationReason,
                 std::vector<std::vector<Stone>> positionHistory)
    : _rules(rules), _board(std::move(board)), _currentPlayer(currentPlayer), _ply(ply),
      _consecutivePasses(consecutivePasses), _terminationReason(terminationReason),
      _positionHistory(std::move(positionHistory)) {}

void GoState::validateRules(const GoRules &rules) {
    if (rules.boardSize < 3) {
        throw std::invalid_argument("Go board size must be at least 3");
    }
    const int64 pointCount = static_cast<int64>(rules.boardSize) * rules.boardSize;
    if (pointCount >= std::numeric_limits<int32>::max()) {
        throw std::invalid_argument("Go board area and pass action must fit in int32");
    }
    if (rules.safetyPlyCap < pointCount) {
        throw std::invalid_argument("Go safety ply cap must be at least the board area");
    }
    if (rules.historyLength < 1) {
        throw std::invalid_argument("Go history length must be positive");
    }
    if (rules.historyLength > MAXIMUM_HISTORY_LENGTH) {
        throw std::invalid_argument("Go history length exceeds the supported maximum");
    }
}

GoState GoState::restore(GoRules rules, std::vector<Stone> board, Player currentPlayer, int32 ply,
                         int32 consecutivePasses, std::vector<std::vector<Stone>> positionHistory) {
    validateRules(rules);
    const auto pointCount =
        static_cast<std::size_t>(rules.boardSize) * static_cast<std::size_t>(rules.boardSize);
    if (board.size() != pointCount || positionHistory.empty() || positionHistory.back() != board) {
        throw std::invalid_argument("Restored Go board history is inconsistent");
    }
    if (std::any_of(positionHistory.begin(), positionHistory.end(),
                    [pointCount](const std::vector<Stone> &historic_board) {
                        return historic_board.size() != pointCount ||
                               std::any_of(
                                   historic_board.begin(), historic_board.end(), [](Stone stone) {
                                       return stone != Stone::Empty && stone != Stone::Black &&
                                              stone != Stone::White;
                                   });
                    })) {
        throw std::invalid_argument("Restored Go board history contains invalid data");
    }
    if (ply < 0 || static_cast<std::size_t>(ply) + 1U != positionHistory.size()) {
        throw std::invalid_argument("Restored Go ply does not match its position history");
    }
    if (ply > rules.safetyPlyCap) {
        throw std::invalid_argument("Restored Go ply exceeds the safety ply cap");
    }
    const Player expectedPlayer = ply % 2 == 0 ? Player::Black : Player::White;
    if (currentPlayer != expectedPlayer) {
        throw std::invalid_argument("Restored Go player does not match its ply");
    }
    if (consecutivePasses < 0 || consecutivePasses > 2 || consecutivePasses > ply) {
        throw std::invalid_argument("Restored Go consecutive pass count is invalid");
    }
    if (ply > 0) {
        const auto &latest = positionHistory[positionHistory.size() - 1U];
        const auto &previous = positionHistory[positionHistory.size() - 2U];
        if ((consecutivePasses == 0) != (latest != previous)) {
            throw std::invalid_argument("Restored Go pass count does not match its history");
        }
        if (consecutivePasses == 2) {
            if (positionHistory.size() < 3U ||
                previous != positionHistory[positionHistory.size() - 3U]) {
                throw std::invalid_argument("Restored Go double pass does not match its history");
            }
        } else if (consecutivePasses == 1 && positionHistory.size() >= 3U &&
                   previous == positionHistory[positionHistory.size() - 3U]) {
            throw std::invalid_argument(
                "Restored Go single pass follows another pass in its history");
        }
    }
    TerminationReason reason = TerminationReason::Ongoing;
    if (consecutivePasses == 2) {
        reason = TerminationReason::TwoPasses;
    } else if (ply >= rules.safetyPlyCap) {
        reason = TerminationReason::SafetyPlyCap;
    }
    return GoState(rules, std::move(board), currentPlayer, ply, consecutivePasses, reason,
                   std::move(positionHistory));
}

const GoRules &GoState::rules() const { return _rules; }
int32 GoState::boardSize() const { return _rules.boardSize; }
int32 GoState::actionCount() const { return _rules.boardSize * _rules.boardSize + 1; }
int32 GoState::passAction() const { return actionCount() - 1; }
Player GoState::currentPlayer() const { return _currentPlayer; }
int32 GoState::ply() const { return _ply; }
int32 GoState::consecutivePasses() const { return _consecutivePasses; }
const std::vector<Stone> &GoState::board() const { return _board; }
const std::vector<std::vector<Stone>> &GoState::positionHistory() const { return _positionHistory; }

std::vector<int32> GoState::neighbors(int32 point) const {
    const int32 row = point / _rules.boardSize;
    const int32 column = point % _rules.boardSize;
    std::vector<int32> result;
    result.reserve(4);
    if (row > 0) {
        result.push_back(point - _rules.boardSize);
    }
    if (column > 0) {
        result.push_back(point - 1);
    }
    if (column + 1 < _rules.boardSize) {
        result.push_back(point + 1);
    }
    if (row + 1 < _rules.boardSize) {
        result.push_back(point + _rules.boardSize);
    }
    return result;
}

std::vector<int32> GoState::groupAt(const std::vector<Stone> &board, int32 origin) const {
    const Stone color = board[static_cast<std::size_t>(origin)];
    std::vector<int32> group;
    std::vector<int32> pending{origin};
    std::vector<bool> seen(board.size(), false);
    seen[static_cast<std::size_t>(origin)] = true;
    while (!pending.empty()) {
        const int32 point = pending.back();
        pending.pop_back();
        group.push_back(point);
        for (const int32 neighbor : neighbors(point)) {
            const auto index = static_cast<std::size_t>(neighbor);
            if (!seen[index] && board[index] == color) {
                seen[index] = true;
                pending.push_back(neighbor);
            }
        }
    }
    return group;
}

bool GoState::groupHasLiberty(const std::vector<Stone> &board,
                              const std::vector<int32> &group) const {
    for (const int32 point : group) {
        for (const int32 neighbor : neighbors(point)) {
            if (board[static_cast<std::size_t>(neighbor)] == Stone::Empty) {
                return true;
            }
        }
    }
    return false;
}

std::vector<Stone> GoState::boardAfterPlacement(int32 action) const {
    std::vector<Stone> candidate = _board;
    const Stone ownStone = stoneFor(_currentPlayer);
    const Stone opponentStone = stoneFor(opponent(_currentPlayer));
    candidate[static_cast<std::size_t>(action)] = ownStone;

    for (const int32 neighbor : neighbors(action)) {
        if (candidate[static_cast<std::size_t>(neighbor)] != opponentStone) {
            continue;
        }
        const std::vector<int32> group = groupAt(candidate, neighbor);
        if (!groupHasLiberty(candidate, group)) {
            for (const int32 point : group) {
                candidate[static_cast<std::size_t>(point)] = Stone::Empty;
            }
        }
    }
    if (!groupHasLiberty(candidate, groupAt(candidate, action))) {
        throw std::invalid_argument("Go placement is suicide");
    }
    return candidate;
}

bool GoState::repeatsPosition(const std::vector<Stone> &board) const {
    return std::find(_positionHistory.begin(), _positionHistory.end(), board) !=
           _positionHistory.end();
}

bool GoState::isLegal(int32 action) const {
    if (isTerminal() || action < 0 || action >= actionCount()) {
        return false;
    }
    if (action == passAction()) {
        return true;
    }
    if (_board[static_cast<std::size_t>(action)] != Stone::Empty) {
        return false;
    }
    try {
        return !repeatsPosition(boardAfterPlacement(action));
    } catch (const std::invalid_argument &) {
        return false;
    }
}

std::vector<int32> GoState::legalActions() const {
    std::vector<int32> result;
    if (isTerminal()) {
        return result;
    }
    for (int32 action = 0; action < actionCount(); ++action) {
        if (isLegal(action)) {
            result.push_back(action);
        }
    }
    return result;
}

void GoState::apply(int32 action) {
    if (!isLegal(action)) {
        throw std::invalid_argument("Illegal Go action");
    }
    if (action == passAction()) {
        ++_consecutivePasses;
    } else {
        _board = boardAfterPlacement(action);
        _consecutivePasses = 0;
    }
    ++_ply;
    _currentPlayer = opponent(_currentPlayer);
    _positionHistory.push_back(_board);
    if (_consecutivePasses == 2) {
        _terminationReason = TerminationReason::TwoPasses;
    } else if (_ply >= _rules.safetyPlyCap) {
        _terminationReason = TerminationReason::SafetyPlyCap;
    }
}

TerminationReason GoState::terminationReason() const { return _terminationReason; }
bool GoState::isTerminal() const { return _terminationReason != TerminationReason::Ongoing; }

AreaScore GoState::areaScore() const {
    int64 blackArea = 0;
    int64 whiteArea = 0;
    std::vector<bool> seen(_board.size(), false);
    for (int32 point = 0; point < passAction(); ++point) {
        const auto index = static_cast<std::size_t>(point);
        if (_board[index] == Stone::Black) {
            ++blackArea;
            continue;
        }
        if (_board[index] == Stone::White) {
            ++whiteArea;
            continue;
        }
        if (seen[index]) {
            continue;
        }
        std::vector<int32> region;
        std::vector<int32> pending{point};
        seen[index] = true;
        bool touchesBlack = false;
        bool touchesWhite = false;
        while (!pending.empty()) {
            const int32 emptyPoint = pending.back();
            pending.pop_back();
            region.push_back(emptyPoint);
            for (const int32 neighbor : neighbors(emptyPoint)) {
                const Stone stone = _board[static_cast<std::size_t>(neighbor)];
                if (stone == Stone::Black) {
                    touchesBlack = true;
                } else if (stone == Stone::White) {
                    touchesWhite = true;
                } else if (!seen[static_cast<std::size_t>(neighbor)]) {
                    seen[static_cast<std::size_t>(neighbor)] = true;
                    pending.push_back(neighbor);
                }
            }
        }
        if (touchesBlack && !touchesWhite) {
            blackArea += static_cast<int64>(region.size());
        } else if (touchesWhite && !touchesBlack) {
            whiteArea += static_cast<int64>(region.size());
        }
    }
    return AreaScore{
        .blackTwice = blackArea * 2,
        .whiteTwice = whiteArea * 2 + static_cast<int64>(_rules.komiHalfPoints),
    };
}

TerminalResult GoState::terminalResult() const {
    if (_terminationReason == TerminationReason::Ongoing) {
        return TerminalResult{
            .reason = _terminationReason, .score = std::nullopt, .winner = std::nullopt};
    }
    if (_terminationReason == TerminationReason::SafetyPlyCap) {
        return TerminalResult{
            .reason = _terminationReason, .score = std::nullopt, .winner = std::nullopt};
    }
    const AreaScore score = areaScore();
    return TerminalResult{.reason = _terminationReason, .score = score, .winner = score.winner()};
}

uint64 GoState::stateHash() const {
    uint64 hash = FNV_OFFSET;
    hashInt(hash, _rules.boardSize);
    hashInt(hash, _rules.komiHalfPoints);
    hashInt(hash, _rules.safetyPlyCap);
    hashInt(hash, _rules.historyLength);
    hashByte(hash, static_cast<uint8>(_currentPlayer));
    hashInt(hash, _ply);
    hashInt(hash, _consecutivePasses);
    hashByte(hash, static_cast<uint8>(_terminationReason));
    for (const auto &historicBoard : _positionHistory) {
        for (const Stone stone : historicBoard) {
            hashByte(hash, static_cast<uint8>(stone));
        }
        hashByte(hash, std::numeric_limits<uint8>::max());
    }
    return hash;
}

Stone GoState::stoneFor(Player player) { return static_cast<Stone>(player); }
Player GoState::opponent(Player player) {
    return player == Player::Black ? Player::White : Player::Black;
}

} // namespace az::v2::games::go
