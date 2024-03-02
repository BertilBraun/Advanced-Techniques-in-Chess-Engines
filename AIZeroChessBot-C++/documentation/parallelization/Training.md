# Training Parallelization Strategy

This document details the training parallelization strategy implemented in our chess engine project, focusing on the utilization of multiple GPUs to enhance the efficiency and scalability of the training process. This approach represents a significant evolution from our initial single-GPU training method, aiming to address performance bottlenecks and improve resource utilization.

## Overview of Data Parallel Approach

The core of our training parallelization strategy is the data parallel approach, where the dataset is divided amongst `n` workers corresponding to the number of GPUs available. This setup allows us to execute forward and backward passes in parallel across multiple models, followed by a synchronization step to accumulate gradients across all models. Once gradients are aggregated, the updated model parameters are redistributed to all models, and the process repeats for the entirety of the training data.

## Process Workflow

1. **Data Distribution:** Training data is split amongst the available workers, each associated with a GPU, ensuring that each worker processes a unique subset of the data.
2. **Parallel Forward and Backward Passes:** Each worker independently performs forward and backward passes on its subset of data, calculating the necessary gradients for learning.
3. **Synchronization Barrier:** Workers reach a synchronization barrier where gradients are accumulated across all models, ensuring consistent learning and model updates.
4. **Gradient Aggregation and Parameter Redistribution:** After gradient accumulation, an optimization step is performed, and the updated model parameters are redistributed to all workers.
5. **Iteration:** The cycle repeats until all training data have been processed, maintaining synchronization to ensure model consistency.

## Transition from Single-GPU to Multi-GPU Training

### Original Single-GPU Implementation

The initial training implementation utilized a single GPU, achieving up to 90% GPU usage. However, this efficiency was periodically disrupted, with drops to about 10% usage, likely due to data loading or gradient updates. This led to significant performance inconsistencies and was identified as a bottleneck in our training process.

### Multi-GPU Implementation

The shift to a multi-GPU approach with distributed models addresses the limitations of the single-GPU setup. Each GPU (or trainer) processes a portion of the training data, leading to more consistent workload distribution. While this introduced more communication overhead, resulting in about 20% GPU usage per trainer, the overall system achieves consistent and scalable GPU usage. With four or more trainers, we maintain a constant total GPU usage of about 80%, demonstrating excellent scalability.

Currently, we are running the system with four T100 GPUs and 16 trainers in parallel, efficiently utilizing the available resources and significantly improving upon the single-GPU model's performance.

## Conclusion

The transition to a multi-GPU, data parallel training approach has markedly enhanced the training efficiency and scalability of our chess engine project. By addressing the limitations of the single-GPU setup, we have achieved a more stable and scalable training environment, paving the way for further advancements in our engine's capabilities.
