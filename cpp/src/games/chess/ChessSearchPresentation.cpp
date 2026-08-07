#include "games/chess/ChessSearchPresentation.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

ChessSearchRoot createChessSearchRoot(Board board, const std::uint32_t arenaCapacity) {
    return ChessSearchRoot(std::move(board), arenaCapacity, 0,
                           ChessGameContract::searchTurnDiscount());
}

std::vector<ChessSearchChild> chessSearchChildren(const ChessSearchRoot &root) {
    const GameSearchNode<ChessGameContract> &rootNode = root.tree().root();
    std::vector<ChessSearchChild> children;
    children.reserve(rootNode.children.size());
    for (const GameSearchEdge<ChessAction> &child : rootNode.children) {
        children.push_back({ChessActionCodec::toUci(child.action),
                            ChessGameContract::actionId(child.action, rootNode.position),
                            child.raw_prior, child.prior, child.visits, child.value_sum,
                            child.virtual_loss, child.child_index.has_value()});
    }
    return children;
}

ChessSearchRoot rerootChessSearch(ChessSearchRoot root, const std::uint32_t childIndex) {
    const GameSearchNode<ChessGameContract> &rootNode = root.tree().root();
    if (childIndex >= rootNode.children.size()) {
        throw std::out_of_range("Cannot reroot to a missing chess action");
    }
    root.play(ChessGameContract::actionId(rootNode.children[childIndex].action, root.position()));
    return root;
}

std::string describeChessSearchRoot(const ChessSearchRoot &root) {
    std::stringstream output;
    output << "ChessSearchRoot(" << root.position().fen() << ", Visits: " << root.visits()
           << ", Score: " << root.tree().root().value_sum
           << ", Virtual Loss: " << root.tree().root().virtual_loss
           << ", Live Nodes: " << root.liveNodeCount() << "/" << root.tree().capacity() << ")";
    return output.str();
}
