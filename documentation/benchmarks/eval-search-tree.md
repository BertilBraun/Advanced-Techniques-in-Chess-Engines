# Evaluation search-tree microbenchmark

`EvalSearchTreeBenchmark` isolates serial MCTS tree mechanics from neural inference. It
uses the same deterministic legal-move policy and board-derived value for both variants:

- `arena-new-delete`: indexed nodes with actual `std::vector` child blocks;
- `arena-pool`: the same indexed nodes with child blocks allocated from a tree-owned
  `std::pmr::unsynchronized_pool_resource`.

The benchmark includes selection, lazy board materialization, expansion, and backup.
It deliberately excludes encoding and inference. Run each variant in a fresh process
when comparing resident memory because standard and PMR allocators can retain released
pages:

```powershell
.\EvalSearchTreeBenchmark arena-new-delete 100000
.\EvalSearchTreeBenchmark arena-pool 100000
```

The output records searches per second, live nodes and edges, resident bytes where the
operating system exposes them, compiler identity, C++ standard, hardware thread count,
and relevant object sizes. Build in Release mode and pin to one CPU when comparing
machines or repeated runs.

The PMR design does not introduce a second arena. Each node owns a conventional
vector-shaped contiguous child block. The resource is declared before node storage, so
all vectors are destroyed before the resource. Re-rooting destroys discarded nodes and
returns their child blocks to the pool for reuse. The pool may retain its high-water
allocation for later searches, but repeated re-rooting does not accumulate live nodes
or request new blocks indefinitely for a stable workload.

The indexed representation follows one child-block pointer per visited node. Scanning
candidate edges is contiguous. A separate edge-span arena is only justified if profiling
shows that the remaining one-block-per-expanded-node lookup is material.

## Local prototype result (2026-07-22)

Hardware and software:

- Intel Core i7-11370H, WSL2 Linux 6.6.87.2;
- GCC 13.3.0, C++20, Release `-O3 -march=native -flto`;
- one benchmark process pinned to logical CPU 0;
- 100,000 searches per process, five interleaved repetitions per variant.

| Variant | Median searches/s | Observed range | RSS | Live nodes | Live edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pointer | 164,603 | 97,074–204,254 | 524.8 MB | 1,956,673 | 1,956,672 |
| Arena `std::vector` | 289,868 | 231,796–366,290 | 309.7 MB | 99,521 | 1,956,672 |
| Arena PMR pool | 229,794 | 134,743–383,056 | 323.1 MB | 99,521 | 1,956,672 |

The local machine showed substantial scheduler/frequency variance, so medians are more
useful than individual runs. The ordinary indexed variant was 76% faster than the
pointer baseline at the median and reduced RSS by 41%. PMR did not provide a consistent
throughput advantage and used about 13 MB more RSS; the production prototype therefore
defaults to ordinary `std::vector` blocks. Re-run the harness on controlled hardware
before treating the exact percentage as portable.

The pointer tree reports one node object per legal edge because expansion constructs all
child nodes, even though their boards remain lazy. The arena reports one full node per
traversed edge and stores untraversed moves only as compact edges. This is the principal
memory and locality improvement.

The historical pointer-tree comparison remains recorded above, but its superseded
implementation is no longer built. Arena growth preserves indices but can relocate node
objects, so callers must encode selected positions before any operation that can grow the
arena.
