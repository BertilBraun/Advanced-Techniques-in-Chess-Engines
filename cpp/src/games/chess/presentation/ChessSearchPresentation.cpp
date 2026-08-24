#include "games/chess/presentation/ChessSearchPresentation.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

ChessSearchRoot createChessSearchRoot(Board board, const std::uint32_t arenaCapacity) {
    return ChessSearchRoot(std::move(board), arenaCapacity);
}

std::vector<ChessSearchChild> chessSearchChildren(const ChessSearchRoot &root) {
    const GameSearchNode<ChessGame> &rootNode = root.tree().root();
    std::vector<ChessSearchChild> children;
    children.reserve(rootNode.children.size());
    for (const GameSearchEdge<ChessAction> &child : rootNode.children) {
        children.push_back({
            .move = child.action.toUci(),
            .encoded_move = child.action_id,
            .raw_policy = child.raw_prior,
            .policy = child.prior,
            .visits = child.visits,
            .result_sum = child.value_sum,
            .virtual_loss = child.virtual_loss,
            .is_materialized = child.child_index.has_value(),
        });
    }
    return children;
}

ChessSearchRoot rerootChessSearch(ChessSearchRoot root, const std::uint32_t childIndex) {
    const GameSearchNode<ChessGame> &rootNode = root.tree().root();
    if (childIndex >= rootNode.children.size()) {
        throw std::out_of_range("Cannot reroot to a missing chess action");
    }
    root.play(rootNode.children[childIndex].action_id);
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
