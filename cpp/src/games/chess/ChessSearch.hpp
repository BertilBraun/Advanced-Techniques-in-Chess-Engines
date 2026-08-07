#pragma once

#include "SearchInference.hpp"
#include "search/BatchedSearch.hpp"
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
using ChessGameSearchRoot = GameSearchRoot<ChessGameContract>;
using ChessInferenceResult = SearchInferenceResult<ChessGameContract>;

struct ChessSearchChild {
    std::string move;
    int encoded_move;
    float raw_policy;
    float policy;
    std::uint32_t visits;
    float result_sum;
    float virtual_loss;
    bool is_materialized;
};

class ChessSearchRoot {
public:
    static ChessSearchRoot create(Board board, std::uint32_t arenaCapacity);
    static ChessSearchRoot create(const std::string &fen, std::uint32_t arenaCapacity);

    explicit ChessSearchRoot(ChessGameSearchRoot root) : m_root(std::move(root)) {}

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
    [[nodiscard]] std::vector<ChessSearchChild> children() const;
    [[nodiscard]] std::string move() const;
    [[nodiscard]] std::string repr() const;

    [[nodiscard]] ChessSearchRoot makeNewRoot(std::uint32_t childIndex);
    void reset();
    void discount(float percentageOfNodeVisitsToKeep) {
        tree().discount(percentageOfNodeVisitsToKeep);
    }

    [[nodiscard]] ChessSearchTree &tree() { return m_root.tree(); }
    [[nodiscard]] const ChessSearchTree &tree() const { return m_root.tree(); }
    [[nodiscard]] NodeIndex rootIndex() const { return tree().rootIndex(); }
    [[nodiscard]] const ChessGameSearchRoot &gameRoot() const { return m_root; }

private:
    ChessGameSearchRoot m_root;
    std::optional<ChessAction> m_move;
};
