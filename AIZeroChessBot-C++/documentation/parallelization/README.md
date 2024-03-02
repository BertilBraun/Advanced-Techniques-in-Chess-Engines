# Parallelization Strategies in Chess Engine Training

This README provides an overview of the parallelization strategies employed in our chess engine project to enhance the efficiency and speed of training. We adopt two main approaches to parallelization: self-play parallelization and training parallelization. Each strategy is designed to maximize the use of computational resources, reduce training time, and improve the overall performance of the chess engine.

## Self-Play Parallelization

The self-play parallelization approach focuses on distributing the self-play phase of the algorithm across multiple workers. By doing so, we can generate a larger dataset of games in a shorter period, which is crucial for the iterative improvement of the chess engine. This method leverages the capability of modern computing systems to handle multiple instances of self-play games concurrently, thus accelerating the data generation process.

For more details on how self-play parallelization is implemented and optimized, please refer to the [self-play documentation](/AIZeroChessBot-C++/documentation/parallelization/SelfPlay.md).

## Training Parallelization

Training parallelization takes advantage of multiple GPUs to expedite the neural network training process. This approach involves distributing the training workload across several GPUs and synchronizing the gradients among the models to ensure consistency and convergence. By parallelizing the training process, we can significantly reduce training time and enhance the learning efficiency of the neural network.

For a comprehensive explanation of the training parallelization strategy, including gradient synchronization and workload distribution, see the [training documentation](/AIZeroChessBot-C++/documentation/parallelization/Training.md).

## Conclusion

The adoption of parallelization strategies in both self-play and training phases is critical to the success of our chess engine project. These approaches allow us to leverage modern hardware capabilities fully, resulting in faster iteration times and more efficient learning processes. For detailed insights into each parallelization method, please consult the respective documentation files linked above.
