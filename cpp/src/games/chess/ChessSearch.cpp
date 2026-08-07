#include "games/chess/ChessSearch.hpp"

#include "games/chess/ChessAction.hpp"

ChessSearchRoot ChessSearchRoot::create(Board board, const std::uint32_t arenaCapacity) {
    return ChessSearchRoot(ChessGameSearchRoot(std::move(board), arenaCapacity, 0, 0.99F));
}

ChessSearchRoot ChessSearchRoot::create(const std::string &fen,
                                        const std::uint32_t arenaCapacity) {
    return create(Board(fen), arenaCapacity);
}

std::vector<ChessSearchChild> ChessSearchRoot::children() const {
    const ChessSearchNode &rootNode = tree().root();
    std::vector<ChessSearchChild> children;
    children.reserve(rootNode.children.size());
    for (const ChessSearchEdge &child : rootNode.children) {
        children.push_back({toString(child.action),
                            ChessGameContract::actionId(child.action, rootNode.position),
                            child.raw_prior,
                            child.prior,
                            child.visits,
                            child.value_sum,
                            child.virtual_loss,
                            child.child_index.has_value()});
    }
    return children;
}

std::string ChessSearchRoot::move() const {
    return m_move.has_value() ? toString(*m_move) : toString(Move::null());
}

std::string ChessSearchRoot::repr() const {
    std::stringstream output;
    output << "ChessSearchRoot(" << board().fen() << ", Move: " << move()
           << ", Visits: " << visits()
           << ", Score: " << resultSum() << ", Virtual Loss: " << virtualLoss()
           << ", Live Nodes: " << liveNodeCount() << "/" << arenaCapacity() << ")";
    return output.str();
}

ChessSearchRoot ChessSearchRoot::makeNewRoot(const std::uint32_t childIndex) {
    const ChessSearchNode &rootNode = tree().root();
    if (childIndex >= rootNode.children.size()) {
        throw std::out_of_range("Cannot reroot to a missing chess action");
    }
    m_move = rootNode.children[childIndex].action;
    tree().rerootEdge(childIndex);
    return *this;
}

void ChessSearchRoot::reset() {
    tree().reset();
    m_move.reset();
}
