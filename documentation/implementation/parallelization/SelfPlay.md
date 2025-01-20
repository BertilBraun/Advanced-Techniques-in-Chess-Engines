# Self-Play Parallelization Strategy

This document outlines the self-play parallelization strategy implemented in our chess engine project, designed to address the imbalance between data generation (self-play) and data consumption (training) rates. By employing a master-worker architecture, we ensure a continuous supply of training data, enhancing the efficiency and effectiveness of the training process.

## Master-Worker Approach

### Master (Trainer)

The master in this architecture is the trainer, responsible for continually training and updating the model with the available data. It acts as the central node that integrates the generated data into the training cycle, refining the model iteratively.

### Workers (Self-Play Instances)

Workers are dedicated self-play instances, each autonomously playing games to generate training data. These workers operate independently, periodically checking for and reloading new model versions to ensure that the self-play data reflects the most current understanding of the game. After generating a predefined number of games (e.g., 32 games), a worker checks for a new model version. If available, the worker reloads the model and resumes self-play, maintaining a cycle of continuous improvement.

## Data Management

All self-play workers write their generated game data as training batches to a shared dataset folder. The trainer periodically fetches new data batches from this folder for training purposes. Once the data has been sufficiently utilized, it is deleted to make room for fresher, more relevant data samples. This process ensures that the training loop focuses on the most current and strategically valuable game instances, optimizing learning efficiency.

## Dynamic Scalability

One of the key advantages of this approach is its inherent scalability. The number of self-play workers can be adjusted dynamically based on the current data generation needs and the computational resources available. This flexibility allows for an efficient balance between data generation and consumption rates, ensuring the trainer always has access to sufficient training material.

## Balancing Generation and Training Times

Despite the efficiency of this parallelization strategy, a significant difference remains between the time required to generate and the time needed to train on the data.

### Training Time Estimation Formula

To achieve a balance where $T_{gen} = T_{train}$, we'll need to determine the number of workers required for sample generation and training processes so that their times are equal. This balance ensures that the time spent generating samples through self-play is equal to the time spent training on those samples, optimizing resource utilization.

Given:

- $T_{gen}$ is the time to generate one sample.
- $W_{gen}$ is the number of workers dedicated to sample generation.
- $T_{batch}$ is the time to process one training batch.
- $E$ is the number of epochs.
- $D$ is the desired dataset size.
- $B$ is the batch size.

### Definitions for Parallel Processing

For parallel processing, the effective time to generate the dataset and the time to train on it are influenced by the number of workers in each process:

1. **Effective Time for Sample Generation with $W_{gen}$ Workers**:

    The total time to generate $D$ samples with $W_{gen}$ workers is:
    $$T_{total\_gen} = \frac{D \times T_{gen}}{W_{gen}}$$

2. **Time for Training**:

    $$T_{total\_train} = E \times \frac{D}{B} \times T_{batch}$$

### Balancing $T_{gen}$ and $T_{train}$

To balance $T_{gen}$ and $T_{train}$, we set $T_{total\_gen} = T_{total\_train}$ and solve for $W_{gen}$, the number of workers needed for sample generation:

$$\frac{D \times T_{gen}}{W_{gen}} = E \times \frac{D}{B} \times T_{batch}$$

Solving for $W_{gen}$:

$$W_{gen} = \frac{D \times T_{gen}}{E \times \frac{D}{B} \times T_{batch}} = \frac{T_{gen} \times B}{E \times T_{batch}}$$

This formula gives you the number of workers for sample generation needed to match the training time, assuming optimal parallelization and no significant overhead for increasing workers.

### Considerations

- **Parallelization Efficiency**: In practice, the efficiency of adding more workers may decrease due to overhead and communication costs. The actual number of workers needed could differ.

### Our Setups' Calculation

Our current setup has the following parameters:

- `1` Worker with `800` Iterations per Move with `64` Games in parallel and a Batch size of `64` generated `9806` samples in `80.4` min
- Training took `13` min for `654912` samples

With these parameters, we can calculate the number of workers needed for sample generation to balance the generation and training times:

- $T_{gen} = 0.492$ seconds
- $T_{batch} = 0.0019$ second
- $E = 40$ epochs
- $B = 64$ batch size

This also lets us calculate the amount of iterations we're able to calculate per second per TPU:

$$\frac{Samples \times Iter\_per\_move}{T_{gen}seconds} = \frac{9806 \times 800}{80.4 \times 60} = 1626.2$$

This is about half the amount of iteration that Google achieved with AlphaZero which is still a good result.

Find $W_{gen}$:

$$W_{gen} = \frac{0.492sec \times 64}{40 \times 0.0019sec} = \frac{31.488}{0.076} = 414.32$$

Rounding up, you would need about 415 workers dedicated to sample generation to balance the generation and training times under these conditions.

Comparing our worker requirements with those of Google's AlphaZero, we can see that AlphaZero required 5000 TPUs Self-Play and 64 TPUs for training. This means they had a ratio of 78.125 workers for sample generation to 1 worker for training. Our ratio is 415 workers for sample generation to 1 worker for training. This is a significant difference and shows that we are not utilizing our resources as efficiently as AlphaZero. Though it is possible that AlphaZero's ratio is not optimal as their trainers might be bottlenecked by the number of workers generating samples.

## Conclusion

The self-play parallelization strategy is a cornerstone of our project, enabling continuous model improvement through a sustainable cycle of data generation and consumption. By leveraging a master-worker architecture and dynamic scalability, we efficiently address the challenges of training time imbalance, ensuring steady progress in the development of our chess engine.
