#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"

#include "MCTS.hpp"

/**
 * Tree‑parallel friendly variant of `MCTSNode`.
 *   • `visit`, `result`, and `virtual_loss` are lock‑free atomics.
 *   • Only the *first* thread that discovers an unexpanded leaf builds the
 *     child list.  The finished vector is published with a single
 *     `std::atomic_store`.  Readers therefore see **all‑or‑nothing** – they
 *     never iterate over a partially filled container, and they never take
 *     a lock.
 *   • Expansion is serialized with a 1‑byte spin‑lock (`atomic_flag`).  That
 *     lock is **never** taken on the critical select/backup hot‑path.
 *
 * The public API mirrors the original `MCTSNode` so the search engine can
 * switch between node types via a simple alias.
 */
class EvalMCTSNode : public std::enable_shared_from_this<EvalMCTSNode> {
public:
    using ChildVector = std::vector<std::shared_ptr<EvalMCTSNode>>;

    /* ──────────────────────────  factory ───────────────────────────── */
    static std::shared_ptr<EvalMCTSNode> createRoot(const std::string &fen) {
        return std::shared_ptr<EvalMCTSNode>(
            new EvalMCTSNode{fen, 1.0f, Move::null(), std::weak_ptr<EvalMCTSNode>()});
    }

    /* ──────────────────  status helpers (thread‑safe) ───────────────── */
    [[nodiscard]] bool isTerminal() const { return board.isGameOver(); }
    [[nodiscard]] bool isExpanded() const { return !children().empty(); }

    /* ──────────────────────  core search helpers  ───────────────────── */
    [[nodiscard]] float ucb(float uCommon) const;

    /**
     * Create all children.  Only one thread may enter; everybody else will
     * return immediately.  The fully‑built vector is *atomically published*
     * so readers never observe an incompletely initialised child.
     */
    void expand(const std::vector<MoveScore> &moves);

    void addVirtualLoss();
    void backPropagateAndRemoveVirtualLoss(float result);

    void backPropagate(float result);

    [[nodiscard]] std::shared_ptr<EvalMCTSNode> bestChild(float cParam) const;

    /* API parity with original node – unchanged semantics */
    std::shared_ptr<EvalMCTSNode> makeNewRoot(std::size_t childIdx);
    [[nodiscard]] int maxDepth() const;

    [[nodiscard]] VisitCounts gatherVisitCounts() const;

    /* ──────────────────────  public data  ──────────────────────────── */
    Board board;
    Move moveToGetHere = Move::null();

    std::atomic<int> number_of_visits{0};
    std::atomic<int> virtual_loss{0};
    std::atomic<float> result_sum{0.f};
    float policy = 0.f; // immutable

    std::weak_ptr<EvalMCTSNode> parent;

    ChildVector children() const {
        const auto cvec = childrenRaw.load(std::memory_order_acquire);
        return cvec ? *cvec : ChildVector{};
    }

private:
    /*
     * Children are stored in a heap‑allocated vector.  `childrenPtr` acts as a
     * once‑initialised atomic pointer: null → not expanded, non‑null → fully
     * expanded.  After publication the vector is *never* mutated again, so
     * readers require no synchronisation at all.
     */
    /* 1️⃣ owning pointer (written once under lock, then never touched)   */
    std::shared_ptr<ChildVector> childrenStrong;
    /* 2️⃣ raw pointer published atomically for readers                   */
    std::atomic<ChildVector *> childrenRaw{nullptr};

    /* 1‑byte spin‑lock used only while *building* the child vector */
    mutable std::atomic_flag expand_lock = ATOMIC_FLAG_INIT;
    struct SpinGuard {
        explicit SpinGuard(std::atomic_flag &f) : flag{f} {
            while (flag.test_and_set(std::memory_order_acquire)) { /* busy wait */
            }
        }
        ~SpinGuard() { flag.clear(std::memory_order_release); }
        std::atomic_flag &flag;
    };

    /* ────────────────────  construction only via factory  ──────────── */
    EvalMCTSNode(const std::string &fen, const float policy, const Move move,
                 std::weak_ptr<EvalMCTSNode> parent)
        : board{fen}, moveToGetHere{move}, policy{policy}, parent{std::move(parent)} {}
};

/* =====================================================================
 *  Inline implementations
 * =================================================================== */
// NOTE: more virtual loss, to avoid the same node being selected multiple times?
// (i.e. multiply delta by 2-5?)
static inline constexpr int VIRTUAL_LOSS_DELTA =
    1; // How much to increase the virtual loss by each time

static inline constexpr float TURN_DISCOUNT =
    0.99f; // Discount factor for the result score when backpropagating
// this makes the result score decay over time, simulating the fact that very long searches add
// more uncertainty to the result.

inline float EvalMCTSNode::ucb(const float uCommon) const {
    const int v = number_of_visits.load(std::memory_order_relaxed);
    const float uScore = policy * uCommon / static_cast<float>(1 + v);

    float qScore = 0.f;
    if (v > 0) {
        const float sum = result_sum.load(std::memory_order_relaxed) +
                          static_cast<float>(virtual_loss.load(std::memory_order_relaxed));
        qScore = -sum / static_cast<float>(v);
    }
    return uScore + qScore;
}

inline void EvalMCTSNode::expand(const std::vector<MoveScore> &moves) {
    if (isExpanded() || moves.empty())
        return;

    SpinGuard g{expand_lock}; // serialize builders
    if (isExpanded())
        return; // another thread won the race

    const auto newChildren = std::make_shared<ChildVector>();
    newChildren->reserve(moves.size());
    for (const auto &[move, policy] : moves) {
        Board next = board;
        next.makeMove(move);
        newChildren->emplace_back(std::shared_ptr<EvalMCTSNode>(
            new EvalMCTSNode{next.fen(), policy, move, weak_from_this()}));
    }
    /* Publish fully‑built vector – release makes sure all contents are visible */
    childrenStrong = newChildren;                                    // own the memory
    childrenRaw.store(newChildren.get(), std::memory_order_release); // publish
}

inline void EvalMCTSNode::addVirtualLoss() {
    for (auto n = shared_from_this(); n; n = n->parent.lock()) {
        n->virtual_loss.fetch_add(VIRTUAL_LOSS_DELTA, std::memory_order_relaxed);
        n->number_of_visits.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void EvalMCTSNode::backPropagateAndRemoveVirtualLoss(float result) {
    for (auto n = shared_from_this(); n; n = n->parent.lock()) {
        n->result_sum.fetch_add(result, std::memory_order_relaxed);
        n->virtual_loss.fetch_sub(VIRTUAL_LOSS_DELTA, std::memory_order_relaxed);
        result = -result * TURN_DISCOUNT;
    }
}

inline void EvalMCTSNode::backPropagate(float result) {
    for (auto n = shared_from_this(); n; n = n->parent.lock()) {
        n->result_sum.fetch_add(result, std::memory_order_relaxed);
        n->number_of_visits.fetch_add(1, std::memory_order_relaxed);
        result = -result * TURN_DISCOUNT;
    }
}

inline std::shared_ptr<EvalMCTSNode> EvalMCTSNode::bestChild(const float cParam) const {
    const auto childVec = children();
    if (childVec.empty())
        return nullptr; // caller treats as leaf

    const int visits = number_of_visits.load(std::memory_order_relaxed);
    const float uCommon = cParam * std::sqrt(static_cast<float>(visits));

    auto best = childVec[0];
    float bestV = -std::numeric_limits<float>::infinity();

    for (const auto &child : childVec) {
        const float v = child->ucb(uCommon);
        if (v > bestV) {
            bestV = v;
            best = child;
        }
    }
    return best;
}

inline std::shared_ptr<EvalMCTSNode> EvalMCTSNode::makeNewRoot(const size_t childIdx) {
    const auto childVec = children();
    assert(childIdx < childVec.size());

    auto newRoot = childVec[childIdx];
    newRoot->parent.reset();

    /* Drop every other subtree (only we hold shared_ptr copies here) */
    childrenRaw.store(nullptr, std::memory_order_release); // release the old pointer
    childrenStrong.reset();                                // release the old vector
    return newRoot;
}

inline int EvalMCTSNode::maxDepth() const {
    int depth = 0;
    for (const auto &child : children()) {
        const int childDepth = child->maxDepth();
        if (childDepth > depth)
            depth = childDepth;
    }
    return depth + 1; // +1 for the current node
}

inline VisitCounts EvalMCTSNode::gatherVisitCounts() const {
    const auto childVec = children();
    VisitCounts visitCounts;
    visitCounts.reserve(childVec.size());
    for (const auto &child : childVec) {
        int encodedMove = encodeMove(child->moveToGetHere, &board);
        visitCounts.emplace_back(encodedMove,
                                 child->number_of_visits.load(std::memory_order_relaxed));
    }
    return visitCounts;
}