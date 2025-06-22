# MCTS Optimizations in AlphaZero Implementation

This document describes the high-level algorithmic optimizations implemented in the Monte Carlo Tree Search (MCTS) component of this AlphaZero system. These optimizations focus on improving search efficiency, parallelization, and training data quality rather than low-level implementation details.

## 1. Fast vs Full Playouts (Playout-Cap Randomization)

### Implementation

The system implements a dual-mode search strategy inspired by KataGo's "Randomized Playout Cap" (RPC):

- **Full Searches**: Use the complete search budget (`num_full_searches`)
- **Fast Searches**: Use a reduced search budget (`num_fast_searches`)

### Selection Criteria

Full searches are performed when:

- All moves are being stored for training (`!only_store_sampled_moves`)
- The game is in early phase (`< num_moves_after_which_to_play_greedy`)
- Many pieces remain on the board (`> 8 pieces` for chess)
- The game hasn't been resigned
- Random chance based on `playout_cap_randomization` parameter

### Core Purpose: Value Target Generation

The primary goal of RPC is to **maximize the number of completed games** to generate more value targets for training:

- **Value Learning Challenge**: Value targets are only available at game completion, making them much harder to learn than policy targets
- **Fast Playouts**: Allow games to finish more quickly by using shallow searches and selecting decent (but not necessarily optimal) moves
- **Training Data Strategy**: Only moves with full searches contribute policy targets to avoid suboptimal policy training
- **Value Target Diversity**: Fast playouts generate diverse game endings, providing rich value training data

### Benefits

- **Training Efficiency**: Maximizes value target generation per unit of computation
- **Data Quality**: Separates high-quality policy data (full searches) from diverse value data (all games)
- **Game Completion Rate**: Enables finishing ~4x more games by using fast searches strategically

## 2. Virtual Loss Mechanism

### Implementation

Virtual losses are applied during parallel search to prevent multiple threads from exploring the same path simultaneously:

```cpp
constexpr int VIRTUAL_LOSS_DELTA = 1;
```

### Process

1. **Selection Phase**: Add virtual loss to selected path (`addVirtualLoss()`)
2. **Expansion/Evaluation**: Perform neural network inference
3. **Backpropagation**: Remove virtual loss and add real result (`backPropagateAndRemoveVirtualLoss()`)

### Benefits

- **Parallelization**: Enables effective parallel MCTS by discouraging redundant exploration
- **Search Diversity**: Forces different threads to explore different parts of the tree
- **Load Balancing**: Distributes computational work more evenly across threads

## 3. Q-Value Initialization Strategies

### Current Implementation

```cpp
float qScore = 0.0f; // parentQ - FPU_REDUCTION;
```

### Considered Alternatives

The code shows awareness of multiple initialization strategies:

- **Zero Initialization**: `0.0f` (currently used)
- **Parent Q with FPU Reduction**: `parentQ - FPU_REDUCTION` (commented out)
- **Negative Initialization**: `-1.0f` (mentioned as CrazyAra's approach)
- **Parent Q**: `parentQ` (mentioned as LeelaZero's approach)

### First Play Urgency (FPU)

```cpp
constexpr float FPU_REDUCTION = 0.10f; // ≈-10 centipawns
```

### Trade-offs

- **Zero Init**: Neutral starting point, relies purely on UCB exploration term
- **Parent Q**: Inherits parent's value estimate, potentially more informed
- **FPU Reduction**: Encourages exploration of unvisited nodes by making them slightly less attractive

## 4. Node Reuse and Tree Persistence

### Implementation

The system maintains MCTS trees between moves through:

- **Root Transition**: `makeNewRoot(childIdx)` creates new root from selected child
- **Tree Pruning**: Automatically discards unused subtrees
- **Node Expansion Tracking**: `already_expanded_node` maintains tree state

### Benefits

- **Computational Efficiency**: Reuses previous search results
- **Memory Optimization**: Prevents memory leaks from abandoned subtrees
- **Search Continuity**: Maintains deep analysis across game moves

## 5. Progressive Search Scaling

### Implementation

Search budget scales with training progress:

```python
self.num_searches_per_turn = int(lerp(
    self.args.mcts.num_searches_per_turn / 2,
    self.args.mcts.num_searches_per_turn,
    clamp(iteration * 20 / TRAINING_ARGS.num_iterations, 0.0, 1.0)
))
```

### Benefits

- **Early Training**: Faster games with reduced search depth when neural network is weak
- **Late Training**: Full search depth when neural network provides better guidance
- **Curriculum Learning**: Gradually increases task difficulty

## 6. Node Discounting for Randomization

### Implementation

```cpp
void MCTSNode::discount(float percentage_of_node_visits_to_keep) {
    number_of_visits *= percentage_of_node_visits_to_keep;
    result_sum *= percentage_of_node_visits_to_keep;
}
```

### Purpose

Part of the Playout-Cap Randomization strategy:

- Reduces confidence in previous search results
- Encourages fresh exploration in full searches
- Prevents over-reliance on cached evaluations

## 7. Dirichlet Noise for Exploration

### Implementation

```cpp
std::vector<float> dirichlet(const float alpha, const size_t n) {
    std::gamma_distribution<float> gamma(alpha, 1.0);
    // ... sampling logic
}
```

### Application

- **Root Node Only**: Noise applied only to root node's children
- **Full Searches Only**: Only applied during full search mode
- **Policy Mixing**: `child->policy = lerp(child->policy, noise[i], dirichlet_epsilon)`

### Benefits

- **Exploration**: Ensures all legal moves have some probability of being explored
- **Training Robustness**: Prevents overfitting to deterministic search results
- **Opening Diversity**: Encourages varied game openings

## 8. Turn-Based Result Discounting

### Implementation

```cpp
constexpr float TURN_DISCOUNT = 0.99f;
```

Applied during backpropagation:

```cpp
result = -1.0f * result * TURN_DISCOUNT;
```

### Purpose

- **Uncertainty Modeling**: Accounts for increased uncertainty in longer searches
- **Temporal Preference**: Slightly favors shorter winning sequences
- **Noise Reduction**: Reduces impact of very deep, potentially noisy evaluations

## 9. Minimum Visit Count Thresholding

### Implementation

```cpp
uint8 min_visit_count; // Minimum visit count for a child of a root node
```

### Selection Logic

```cpp
for (const auto &child : root->children) {
    if (child->number_of_visits < m_args.min_visit_count) {
        root = child;
        break;
    }
}
```

### Benefits

- **Balanced Exploration**: Ensures all children receive minimum attention before deep exploration
- **Search Quality**: Prevents premature convergence to single lines
- **Statistical Reliability**: Ensures visit counts are statistically meaningful

## 10. Temperature-Based Move Selection

### Implementation

```python
def _sample_from_probabilities(action_probabilities: np.ndarray, temperature: float = 1.0) -> int:
    temperature_action_probabilities = action_probabilities ** (1 / temperature)
    temperature_action_probabilities /= np.sum(temperature_action_probabilities)
    return np.random.choice(len(action_probabilities), p=temperature_action_probabilities)
```

### Progressive Temperature Scaling

```python
temperature = lerp(
    self.args.starting_temperature, 
    self.args.final_temperature, 
    game_progress
)
```

### Benefits

- **Early Game**: High temperature maintains exploration and training diversity
- **Late Game**: Low temperature focuses on best moves for game outcome
- **Smooth Transition**: Gradual temperature reduction prevents abrupt strategy changes

## 11. Batch Processing and Parallelization

### Multi-Level Parallelization

1. **Game-Level**: Multiple games processed simultaneously
2. **Search-Level**: Parallel search strands within each game
3. **Inference-Level**: Batched neural network evaluation

### Implementation Strategy

```cpp
std::vector<std::shared_ptr<MCTSNode>> batch;
for (auto &[root, limit] : active)
    batch.emplace_back(root);
parallelIterate(batch);
```

### Benefits

- **Hardware Utilization**: Maximizes GPU and CPU usage
- **Throughput**: Significantly increases games per second
- **Scalability**: Adapts to available computational resources

## 12. UCB Exploration Term Optimization

### Implementation

```cpp
float ucb(const MCTSNode &node, float exploration_weight) {
    float uScore = node.policy * sqrt(node.parent->number_of_visits) / (1 + node.number_of_visits);
    // TODO which is the best initializer for qScore?
    // most seem to init to 0.0
    // CrazyAra inits to -1.0
    // LeelaZero inits to -parentScore
    // some init to parentScore - FPU_REDUCTION
    float qScore = 0.0f;
    if (node.number_of_visits > 0) {
        qScore = -1 * (node.result_sum + node.virtual_loss) / node.number_of_visits;
    }
    return qScore + exploration_weight * uScore;
}
```

### Considerations

- **Exploration Weight**: Tunable parameter to balance exploration vs exploitation (1.0 - 1.8 commonly used)
- **Q-Value Initialization**: Various strategies considered, currently using zero initialization. Since values that are good for the parent, are bad for the current node, I figured inverting the value of the parent is the best approximation of the q score that I have. But this emperically didn't work well, so I reverted to zero initialization.
- **UCB Formula**: Combines policy probability and visit counts to guide search.

## 13. Optimized 1-Board Evaluation

Optimized Parallel MCTS search using atomics and spinlocks and virtual losses in the tree nodes to parallelize with batching Network requests over the parallel running search threads for inference.

This implementation is designed for scenarios where only a single board is evaluated, such as in tournaments or evaluation games. It allows for efficient parallelization of the MCTS search process while maintaining high throughput. This implementation will be slower than the batch processing implementation, if more than one board is evaluated, but it is optimized for single-board evaluation.

This achieves about 6-10k searches per second on a single board on a single A10 GPU with 16 CPUs.

## 14. Implementation Performance Analysis

### Python to C++ Migration Impact

The MCTS implementation was initially developed in Python but migrated to C++ to unlock significant parallelization potential. Performance testing revealed two distinct optimization scenarios, each favoring different implementation approaches.

### Performance Comparison Results

![Performance Analysis Overview](../images/MCTS%20Search%20comparisons.png)

Figure: Average per-board search times for different batch sizes and thread counts. Each cell shows the time per board when processing N boards simultaneously (64B) versus a single board (1B) across Python, C++, and C++ Eval implementations.

Observation: Batching 64 boards drastically reduces per-board search time compared to single-board runs. Although neural inference scales mildly with batch size, the amortized overhead of inference calls and improved parallel utilization makes large batches far more efficient.

#### Batch Processing Scenario (Self-Play Training)

For self-play training where multiple game boards are processed simultaneously:

- **C++ Implementation**: 15-20× speedup over Python baseline
- **Optimal Configuration**: 64+ boards with high thread counts (96T configuration)
- **Peak Throughput**: Over 21,000 neural network inferences per second
- **Primary Bottleneck**: Game state management and MCTS tree traversal, not neural network inference
- **Scaling Behavior**: Performance scales dramatically with batch size due to amortized overhead

#### Single Board Scenario (Tournament/Evaluation Play)

For tournament or evaluation scenarios processing individual game states:

- **C++ Eval Implementation**: 4× speedup over both Python and regular C++ implementations  
- **Unique Characteristic**: Effective utilization of multiple CPU cores for single-board search
- **Peak Performance**: Nearly 5,000 inferences per second on single boards
- **Key Innovation**: Superior intra-board parallelization of search processes
- **Thread Utilization**: Benefits from high thread counts even without batch processing

### Implementation Strategy Recommendations

**For Training/Self-Play Workloads:**

- Deploy regular C++ implementation with maximum available boards per batch
- Prioritize high thread counts (96T configuration provides optimal throughput)
- Focus optimization efforts on batch size scaling over single-board performance
- Expect linear scaling benefits with increased parallelism

**For Tournament/Evaluation Workloads:**

- Deploy C++ Eval implementation for single-board processing scenarios
- Maintain high thread counts to leverage intra-board parallelization advantages
- Expect significant performance gains even without batch processing benefits
- Optimize for search depth and quality over raw throughput

### Performance Architecture Insights

The analysis demonstrates that optimal MCTS implementation strategy depends critically on the deployment scenario:

- **Batch Processing**: Benefits from reducing per-operation overhead through parallelization across multiple game states
- **Single Board Processing**: Benefits from parallelizing the search process within individual game states
- **Thread Utilization**: Both scenarios benefit from high thread counts but through different parallelization mechanisms
- **Bottleneck Analysis**: Training workloads are CPU-bound (tree traversal), while evaluation workloads benefit from deeper search parallelization

## Summary

These optimizations work together to create an efficient, scalable MCTS implementation that balances:

- **Computational Efficiency**: Through fast/full search modes, progressive scaling, and language-specific optimizations
- **Search Quality**: Through proper initialization, virtual losses, and minimum visit thresholds  
- **Training Effectiveness**: Through temperature control, noise injection, and balanced exploration
- **Parallelization**: Through virtual losses, batch processing, multi-level parallelism, and implementation-specific optimizations
- **Memory Management**: Through node reuse, tree pruning, and proper cleanup
- **Performance Scaling**: Through C++ implementation providing 4-20× speedup depending on use case

The combination of these techniques, particularly the strategic choice between C++ implementations based on deployment scenario, enables the system to generate high-quality training data efficiently while maintaining the theoretical guarantees of Monte Carlo Tree Search. The performance analysis reveals that implementation language and parallelization strategy have profound impacts on system throughput, with C++ providing substantial advantages in both training and evaluation contexts.
