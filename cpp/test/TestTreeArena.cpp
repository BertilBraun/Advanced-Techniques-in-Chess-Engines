#include "search/TreeArena.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace {

bool allocationTracking = false;
std::size_t allocationCount = 0;

struct TestNodeData {
    int32 state;
    int32 visits;
    double valueSum;
};

struct TestEdgeData {
    int32 action;
    double prior;
    int32 visits;
};

using Arena = az::search::TreeArena<TestNodeData, TestEdgeData>;

template <typename Exception, typename Operation> void expectException(Operation operation) {
    bool threw = false;
    try {
        operation();
    } catch (const Exception &) {
        threw = true;
    }
    assert(threw);
}

void representativeArenaCycle(Arena &arena, az::search::SearchScratch<int32> &scratch,
                              const std::array<TestEdgeData, 2> &edgeData) {
    const az::search::NodeIndex root =
        arena.reset(TestNodeData{.state = 0, .visits = 0, .valueSum = 0.0});
    const std::optional<az::search::EdgeSpan> rootEdges = arena.createEdges(root, edgeData);
    assert(rootEdges.has_value());

    const auto firstChild =
        arena.createNode(TestNodeData{.state = 1, .visits = 0, .valueSum = 0.0}, root);
    const auto secondChild =
        arena.createNode(TestNodeData{.state = 2, .visits = 0, .valueSum = 0.0}, root);
    assert(firstChild.has_value() && secondChild.has_value());
    arena.setChild(arena.edgeIndex(*rootEdges, 0), *firstChild);
    arena.setChild(arena.edgeIndex(*rootEdges, 1), *secondChild);

    scratch.reset();
    scratch.pushPath(root);
    scratch.pushPath(*firstChild);
    scratch.pushAction(0, 0.75);
    scratch.pushAction(1, 0.25);

    arena.node(*firstChild).data.visits += 1;
    arena.node(*firstChild).data.valueSum += 0.5;
    arena.edge(arena.edgeIndex(*rootEdges, 0)).data.visits += 1;
    arena.node(root).data.visits += 1;
    arena.node(root).data.valueSum -= 0.5;
    arena.reroot(*firstChild);
    assert(arena.rootIndex() == *firstChild);
}

void testNoAllocationsAfterWarmup() {
    Arena arena(az::search::TreeArenaConfiguration{
        .nodeCapacity = 8,
        .edgeCapacity = 16,
        .capacityPolicy = az::search::ArenaCapacityPolicy::StopSearch,
    });
    az::search::SearchScratch<int32> scratch(8, 8);
    const std::array<TestEdgeData, 2> edges{
        TestEdgeData{.action = 0, .prior = 0.75, .visits = 0},
        TestEdgeData{.action = 1, .prior = 0.25, .visits = 0},
    };
    representativeArenaCycle(arena, scratch, edges);

    allocationCount = 0;
    allocationTracking = true;
    for (int32 cycle = 0; cycle < 100; ++cycle) {
        representativeArenaCycle(arena, scratch, edges);
    }
    allocationTracking = false;
    assert(allocationCount == 0);
}

void testGenerationAndCapacityBehavior() {
    Arena arena(az::search::TreeArenaConfiguration{
        .nodeCapacity = 2,
        .edgeCapacity = 2,
        .capacityPolicy = az::search::ArenaCapacityPolicy::StopSearch,
    });
    const az::search::NodeIndex staleRoot =
        arena.reset(TestNodeData{.state = 0, .visits = 0, .valueSum = 0.0});
    const auto child =
        arena.createNode(TestNodeData{.state = 1, .visits = 0, .valueSum = 0.0}, staleRoot);
    assert(child.has_value());
    assert(!arena.createNode(TestNodeData{.state = 2, .visits = 0, .valueSum = 0.0}, staleRoot)
                .has_value());
    assert(arena.telemetry().nodeCapacityExhaustions == 1);

    const std::array<TestEdgeData, 3> tooManyEdges{
        TestEdgeData{.action = 0, .prior = 0.5, .visits = 0},
        TestEdgeData{.action = 1, .prior = 0.3, .visits = 0},
        TestEdgeData{.action = 2, .prior = 0.2, .visits = 0},
    };
    assert(!arena.createEdges(staleRoot, tooManyEdges).has_value());
    assert(arena.telemetry().edgeCapacityExhaustions == 1);

    static_cast<void>(arena.reset(TestNodeData{.state = 3, .visits = 0, .valueSum = 0.0}));
    expectException<std::logic_error>(
        [&arena, staleRoot]() { static_cast<void>(arena.node(staleRoot)); });
}

void testFailSessionCapacityPolicy() {
    Arena arena(az::search::TreeArenaConfiguration{
        .nodeCapacity = 1,
        .edgeCapacity = 1,
        .capacityPolicy = az::search::ArenaCapacityPolicy::FailSession,
    });
    const az::search::NodeIndex root =
        arena.reset(TestNodeData{.state = 0, .visits = 0, .valueSum = 0.0});
    expectException<std::overflow_error>([&arena, root]() {
        static_cast<void>(
            arena.createNode(TestNodeData{.state = 1, .visits = 0, .valueSum = 0.0}, root));
    });
}

} // namespace

void *operator new(std::size_t size) {
    if (allocationTracking) {
        ++allocationCount;
    }
    if (void *allocation = std::malloc(size)) {
        return allocation;
    }
    throw std::bad_alloc();
}

void operator delete(void *allocation) noexcept { std::free(allocation); }

void operator delete(void *allocation, std::size_t) noexcept { std::free(allocation); }

int main() {
    testNoAllocationsAfterWarmup();
    testGenerationAndCapacityBehavior();
    testFailSessionCapacityPolicy();
    return 0;
}
