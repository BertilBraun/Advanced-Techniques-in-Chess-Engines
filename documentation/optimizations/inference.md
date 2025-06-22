# Neural Network Inference Optimizations

This document details our current strategies and benchmarks for optimizing neural network inference within our MCTS self-play framework.

---

## 1. Hardware & Precision Tuning

![NN Speed Comparison](../images/Network%20Inference%20Speed%20comparison.png)

*Inference latency per board across GPU types (A10, A100, V100) and precision modes (FP32, bfloat16).*
Despite newer GPUs, the **A10** achieved the best real-world throughput for our workload:

* **Per-board latency (bfloat16)**: \~157 µs
* **Batch of 100 boards**: \~15.7 ms total

**Techniques**:

* Mixed precision (`bfloat16`) inference
* Model compilation (`torch.compile`) and operator fusion (Conv2d + BatchNorm2d + ReLU)

---

## 2. C++ Inference Pipeline

* **Full C++ Port**: All inference moved to a dedicated C++ server process, eliminating Python overhead.
* **Worker Configuration**:

  * **8 self-play workers per GPU**, each with its own model instance and in-memory cache.
  * **4–6 MCTS threads per worker** enqueue inference requests concurrently.
  * **\~192 parallel boards per worker** (32 games × 6 threads), boosted by virtual-loss scheduling.
* **Throughput**: \~20 000 inference calls/sec per worker → ∼160 000 calls/sec per GPU.

---

## 3. Caching Strategies

* **In-Memory Shared Cache**: One mutex‑guarded hash table in C++ shared across threads.
* **Keying**: Canonical board states via symmetry-aware Zobrist hashing.
* **Hit Rate**: \~10–15% under current workloads—still reduces redundant work.

---

## 4. Local Batching

* **Batch Size**: Inference requests are grouped into batches of \~100 boards within each worker.
* **Latency Amortization**: Larger batches lower per-board overhead and maximize GPU utilization.

---

## 5. Torch Optimizations & Baseline Comparison

### 5.1 Torch/Libraries

* `torch.compile` speedups
* Quantization/fusion with `torch.quantization.fuse_modules`
* Use of `bfloat16` for reduced memory and faster compute
* Numba (`njit`) acceleration for any remaining Python data munging (\~30× speedups)

### 5.2 Baseline ([Crazyhouse, Czech et al. 2019](https://ml-research.github.io/papers/czech2019deep.pdf))

> • 45 self-play games/min per GPU → \~1 000 samples/min
> • 800 MCTS rollouts/sample, batch size 8

**Our Results**:

* \~160 000 inference calls/sec per GPU → saturates modern GPUs with 7–8 GPUs for self-play+training.

---

*Through these measures, inference is now the dominant cost in MCTS, accounting for >95% of runtime. Ongoing work focuses on further reducing per-call latency and increasing cache efficiency.*
