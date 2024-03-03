# Training Parallelization Strategy

This document details the training parallelization strategy for our chess engine project, with a focus on our experience and findings from implementing multiple GPUs to enhance the efficiency and scalability of the training process. Initially, the strategy aimed to evolve from our single-GPU training method to address performance bottlenecks and improve resource utilization. However, subsequent evaluations revealed significant challenges, particularly regarding communication overhead and synchronization times.

## Overview of Data Parallel Approach

The data parallel approach divides the dataset among `n` workers, where one or more workers correspond to a GPU. This setup theoretically allows for parallel execution of forward and backward passes across multiple models, followed by a synchronization step to accumulate and redistribute gradients. Despite its potential, our findings indicate that the increase in communication overhead substantially impacts the efficiency of this model.

## Process Workflow and Performance Findings

1. **Data Distribution:** Training data is split among available workers, each associated with a GPU. Each worker processes a unique subset of the data.

2. **Parallel Forward and Backward Passes:** While each worker performs forward and backward passes independently, only 25% of the total training time is dedicated to these operations due to significant overheads elsewhere.

3. **Synchronization Barrier and Gradient Aggregation:** These stages now consume about 75% of the total training time, with 50% for gradient aggregation across devices and 25% for synchronization and parameter updates.

4. **Performance Bottleneck:** The primary bottleneck has shifted from the inefficiencies in a single-GPU setup to communication overhead and synchronization in the multi-GPU configuration.

## Performance Evaluation

A comparative analysis of different training setups showed:

- **Single-threaded (1 GPU, 2000 Samples):** Completed in 31 seconds, with GPU usage fluctuating between 10% and 90%.
- **Multi-threaded (1 GPU, 4 Threads, 8000 Total Samples):** Completed in 141 seconds, averaging 35.3 seconds per 2000 Samples, with GPU usage at 70-80%.
- **Multi-threaded (2 GPUs, 5 Threads, 20000 Total Samples):** Completed in 364 seconds, averaging 36.4 seconds per 2000 Samples, with each GPU's usage at 40-50%.
- **Multi-threaded (2 GPUs, 3 Threads, 6000 Total Samples):** Completed in 123 seconds, averaging 41 seconds per 2000 Samples, with GPU usage ranging from 30% to 60%.

## Conclusion and Strategy Update

The evaluation demonstrates that the increased communication overhead and synchronization requirements of the multi-GPU setup lead to a significant performance degradation compared to a single-GPU approach. As a result, the project will revert to a single-threaded (single-GPU) model for the foreseeable future. This decision is grounded in the superior time efficiency and more predictable GPU usage of the single-GPU model.

Future directions will include optimizing data parallelism techniques and investigating alternative parallelization strategies that could offer a better balance between computational efficiency and communication overhead.
