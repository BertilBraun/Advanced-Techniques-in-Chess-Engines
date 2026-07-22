#pragma once

#include <memory>

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
    using ChildSnapshot = std::shared_ptr<const ChildVector>;

    /* ──────────────────────────  factory ───────────────────────────── */
    static std::shared_ptr<EvalMCTSNode> createRoot(const std::string &fen) {
        return createRoot(Board(fen));
    }

    static std::shared_ptr<EvalMCTSNode> createRoot(Board board) {
        return std::shared_ptr<EvalMCTSNode>(new EvalMCTSNode{
            std::make_unique<Board>(std::move(board)), 1.0f, Move::null(),
            std::weak_ptr<EvalMCTSNode>()});
    }

    /* ──────────────────  status helpers (thread‑safe) ───────────────── */
    [[nodiscard]] bool isTerminal() const { return board().isGameOver(); }
    [[nodiscard]] bool isExpanded() const { return children() != nullptr; }
    [[nodiscard]] bool hasMaterializedBoard() const { return m_board != nullptr; }
    [[nodiscard]] const Board &board() const {
        assert(m_board != nullptr);
        return *m_board;
    }
    void materializeBoard();

    /* ──────────────────────  core search helpers  ───────────────────── */
    [[nodiscard]] float ucb(float uCommon) const;

    /**
     * Create all children.  Only one thread may enter; everybody else will
     * return immediately.  The fully‑built vector is *atomically published*
     * so readers never observe an incompletely initialised child.
     */
    void expand(const std::vector<MoveScore> &moves,
                std::optional<WdlPrediction> outcome = std::nullopt);

    [[nodiscard]] const std::optional<WdlPrediction> &outcomePrediction() const {
        return networkOutcome;
    }

    [[nodiscard]] bool tryBeginEvaluation();
    void endEvaluation();

    void addVirtualLoss();
    void cancelVirtualLoss();
    void backPropagateAndRemoveVirtualLoss(float result);

    void backPropagate(float result);

    [[nodiscard]] std::shared_ptr<EvalMCTSNode> bestChild(float cParam) const;

    /* API parity with original node – unchanged semantics */
    std::shared_ptr<EvalMCTSNode> makeNewRoot(std::size_t childIdx);
    [[nodiscard]] int maxDepth() const;

    [[nodiscard]] VisitCounts gatherVisitCounts() const;

    /* ──────────────────────  public data  ──────────────────────────── */
    Move moveToGetHere = Move::null();

    std::atomic<int> number_of_visits{0};
    std::atomic<int> virtual_loss{0};
    std::atomic<float> result_sum{0.f};
    float policy = 0.f; // immutable
    std::optional<WdlPrediction> networkOutcome;

    std::weak_ptr<EvalMCTSNode> parent;

    [[nodiscard]] ChildSnapshot children() const {
        return std::atomic_load_explicit(&childrenPublished, std::memory_order_acquire);
    }

private:
    // Readers own an immutable snapshot while traversing concurrently.
    ChildSnapshot childrenPublished;
    std::atomic_flag evaluationInProgress = ATOMIC_FLAG_INIT;

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
    EvalMCTSNode(std::unique_ptr<Board> board, const float policy, const Move move,
                 std::weak_ptr<EvalMCTSNode> parent)
        : m_board{std::move(board)}, moveToGetHere{move}, policy{policy},
          parent{std::move(parent)} {}

    std::unique_ptr<Board> m_board;
};

/* =====================================================================
 *  Inline implementations
 * =================================================================== */
// NOTE: more virtual loss, to avoid the same node being selected multiple times?
// (i.e. multiply delta by 2-5?)
static inline constexpr int VIRTUAL_LOSS_DELTA =
    1; // How much to increase the virtual loss by each time

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

inline void EvalMCTSNode::expand(const std::vector<MoveScore> &moves,
                                 const std::optional<WdlPrediction> outcome) {
    if (isExpanded() || moves.empty())
        return;

    SpinGuard g{expand_lock}; // serialize builders
    if (isExpanded())
        return; // another thread won the race

    networkOutcome = outcome;

    const auto newChildren = std::make_shared<ChildVector>();
    newChildren->reserve(moves.size());
    for (const auto &[move, policy] : moves) {
        newChildren->emplace_back(std::shared_ptr<EvalMCTSNode>(
            new EvalMCTSNode{nullptr, policy, move, weak_from_this()}));
    }
    /* Publish fully‑built vector – release makes sure all contents are visible */
    std::atomic_store_explicit(&childrenPublished, ChildSnapshot{std::move(newChildren)},
                               std::memory_order_release);
}

inline void EvalMCTSNode::materializeBoard() {
    if (m_board != nullptr) {
        return;
    }

    SpinGuard guard{expand_lock};
    if (m_board != nullptr) {
        return;
    }
    const std::shared_ptr<EvalMCTSNode> parentNode = parent.lock();
    assert(parentNode != nullptr);
    m_board = std::make_unique<Board>(parentNode->board());
    m_board->makeMove(moveToGetHere);
}

inline bool EvalMCTSNode::tryBeginEvaluation() {
    return !evaluationInProgress.test_and_set(std::memory_order_acquire);
}

inline void EvalMCTSNode::endEvaluation() {
    evaluationInProgress.clear(std::memory_order_release);
}

inline void EvalMCTSNode::addVirtualLoss() {
    for (auto n = shared_from_this(); n; n = n->parent.lock()) {
        n->virtual_loss.fetch_add(VIRTUAL_LOSS_DELTA, std::memory_order_relaxed);
        n->number_of_visits.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void EvalMCTSNode::cancelVirtualLoss() {
    for (auto node = shared_from_this(); node; node = node->parent.lock()) {
        node->virtual_loss.fetch_sub(VIRTUAL_LOSS_DELTA, std::memory_order_relaxed);
        node->number_of_visits.fetch_sub(1, std::memory_order_relaxed);
    }
}

inline void EvalMCTSNode::backPropagateAndRemoveVirtualLoss(float result) {
    for (auto n = shared_from_this(); n; n = n->parent.lock()) {
        n->result_sum.fetch_add(result, std::memory_order_relaxed);
        n->virtual_loss.fetch_sub(VIRTUAL_LOSS_DELTA, std::memory_order_relaxed);
        result = -result;
    }
}

inline void EvalMCTSNode::backPropagate(float result) {
    for (auto n = shared_from_this(); n; n = n->parent.lock()) {
        n->result_sum.fetch_add(result, std::memory_order_relaxed);
        n->number_of_visits.fetch_add(1, std::memory_order_relaxed);
        result = -result;
    }
}

inline std::shared_ptr<EvalMCTSNode> EvalMCTSNode::bestChild(const float cParam) const {
    const auto childVec = children();
    if (childVec == nullptr || childVec->empty())
        return nullptr; // caller treats as leaf

    const int visits = number_of_visits.load(std::memory_order_relaxed);
    const float uCommon = cParam * std::sqrt(static_cast<float>(visits));

    auto best = childVec->front();
    float bestV = -std::numeric_limits<float>::infinity();

    for (const auto &child : *childVec) {
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
    assert(childVec != nullptr && childIdx < childVec->size());

    auto newRoot = (*childVec)[childIdx];
    newRoot->materializeBoard();
    newRoot->parent.reset();

    /* Drop every other subtree (only we hold shared_ptr copies here) */
    std::atomic_store_explicit(&childrenPublished, ChildSnapshot{}, std::memory_order_release);
    return newRoot;
}

inline int EvalMCTSNode::maxDepth() const {
    int depth = 0;
    const auto childVec = children();
    if (childVec == nullptr)
        return 1;

    for (const auto &child : *childVec) {
        const int childDepth = child->maxDepth();
        if (childDepth > depth)
            depth = childDepth;
    }
    return depth + 1; // +1 for the current node
}

inline VisitCounts EvalMCTSNode::gatherVisitCounts() const {
    const auto childVec = children();
    VisitCounts visitCounts;
    if (childVec == nullptr)
        return visitCounts;

    visitCounts.reserve(childVec->size());
    for (const auto &child : *childVec) {
        int encodedMove = encodeMove(child->moveToGetHere, &board());
        visitCounts.emplace_back(encodedMove,
                                 child->number_of_visits.load(std::memory_order_relaxed));
    }
    return visitCounts;
}
