#pragma once

#include "MCTS/GameSearch.hpp"
#include "games/chess/ChessGameContract.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

using NodeIndex = std::size_t;
inline constexpr NodeIndex INVALID_NODE_INDEX = std::numeric_limits<NodeIndex>::max();

using ChessSearchTree = GameSearchTree<ChessGameContract>;
using ChessSearchNode = GameSearchNode<ChessGameContract>;
using ChessSearchEdge = GameSearchEdge<ChessAction>;
using ChessSearchRoot = GameSearchRoot<ChessGameContract>;

struct MCTSChild {
    std::string move;
    int encoded_move;
    float raw_policy;
    float policy;
    std::uint32_t visits;
    float result_sum;
    float virtual_loss;
    bool is_materialized;
};

class MCTSRoot {
public:
    static MCTSRoot create(Board board, std::uint32_t arenaCapacity);
    static MCTSRoot create(const std::string &fen, std::uint32_t arenaCapacity);

    explicit MCTSRoot(ChessSearchRoot root) : m_root(std::move(root)) {}

    [[nodiscard]] const Board &board() const { return m_root.position(); }
    [[nodiscard]] bool isTerminal() const { return m_root.isTerminal(); }
    [[nodiscard]] bool isExpanded() const { return tree().root().expanded(); }
    [[nodiscard]] std::uint32_t visits() const { return m_root.visits(); }
    [[nodiscard]] float virtualLoss() const { return tree().root().virtual_loss; }
    [[nodiscard]] float resultSum() const { return tree().root().value_sum; }
    [[nodiscard]] int maxDepth() const { return tree().maximumDepth(); }
    [[nodiscard]] std::uint32_t liveNodeCount() const {
        return static_cast<std::uint32_t>(tree().liveNodeCount());
    }
    [[nodiscard]] std::uint64_t totalChildCount() const { return tree().totalChildCount(); }
    [[nodiscard]] std::uint32_t arenaCapacity() const {
        return static_cast<std::uint32_t>(tree().capacity());
    }
    [[nodiscard]] std::vector<MCTSChild> children() const;
    [[nodiscard]] std::string move() const;
    [[nodiscard]] std::string repr() const;

    [[nodiscard]] MCTSRoot makeNewRoot(std::uint32_t childIndex);
    void reset();
    void discount(float percentageOfNodeVisitsToKeep) {
        tree().discount(percentageOfNodeVisitsToKeep);
    }

    [[nodiscard]] ChessSearchTree &tree() { return m_root.tree(); }
    [[nodiscard]] const ChessSearchTree &tree() const { return m_root.tree(); }
    [[nodiscard]] NodeIndex rootIndex() const { return tree().rootIndex(); }
    [[nodiscard]] const ChessSearchRoot &gameRoot() const { return m_root; }

private:
    ChessSearchRoot m_root;
    std::optional<ChessAction> m_move;
};
