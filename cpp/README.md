# C++ Port of Alpha-Zero Chess Bot

// TODO update the readme

As described in the [Future Works](../documentation/future.md) Section of the Python implementation, a lot of performance is lost due to the limited parallelization capabilities of Python regarding the GIL. This project aims to port the Alpha-Zero Chess Bot to C++ to take advantage of the better parallelization capabilities of C++ as well as the better performance of C++ in general.

## Documentation

The project documentation is organized into several categories, each focusing on a specific aspect of the project. The documentation provides more detailed information about specific components, development processes, and performance analysis. The documentation categories are as follows:

- **[Chess Encoding for Neural Networks](documentation/encodings/README.md)**: Describes the board encoding scheme used to represent chess board states as inputs to the neural network.
- **[Chess Framework](documentation/chess/README.md)**: Details the implementation of the chess framework in C++ and the performance improvements achieved through the translation of the framework from Python to C++.
- **[Pre-Training System](documentation/pretraining/README.md)**: Discusses the pre-training system used to generate training data for the neural network using grandmaster games and stockfish evaluations.
- **[Parallelization](documentation/parallelization/README.md)**: Explains the parallelization strategy used to distribute the training data generation and training processes across multiple nodes and GPUs on the cluster.

## Technologies

- **Programming Language**: C++20
- **Machine Learning Frameworks**: LibTorch (PyTorch C++ API)
- **Chess Library**: Python-Chess (self ported to C++)

## Getting Started

To run the `cpp` project after cloning the repository, follow these steps:

### Step 1: Build the Project

1. **Run the Setup Script** (if you haven't already) to download LibTorch and generate the CMake build system. In the terminal, navigate to your project's root directory and run:

    ```cmd
    setup_build.bat
    ./setup_build.sh
    ```

   This script prepares the build environment, including downloading LibTorch if necessary and generating build files.

2. **Build the Project** using CMake.

    ```cmd
    cd build
    cmake --build . --config Release
    ```

    Or alternatively:

    ```cmd
    cd build
    make
    ```

### Step 2: Running the Executable

After building the project, an executable file named `AIZeroChessBot` (or `AIZeroChessBot.exe` on Windows) will be created in the `build` directory, inside a `Release` or `Debug` subdirectory, depending on your build configuration.

To run your project:

1. **Navigate to the Executable Directory** in the terminal using `cd`:

    ```cmd
    cd Release  # or Debug, depending on your build config
    ```

2. **Run the Executable** by typing its name in the terminal:

    ```cmd
    AIZeroChessBot <train|generate> <num_workers>
    ```

   This command executes your program.

   - The first argument specifies the mode (`train` or `generate`).
     - `train` mode trains the neural network while simultaneously generating self-play data.
     - `generate` mode generates data for training the neural network from datasets of grandmaster games and stockfish evaluations. Read [Self-Play Pre-Training System](/AIZeroChessBot-C++/documentation/pretraining/README.md) for more information.
   - The second argument specifies the number of workers to use for the specified mode.

**On the Cluster**: To run the project on the cluster, you can use the provided `train/train.sh` script to submit a job to the cluster. The script will handle the build and execution of the project on the cluster. To submit a job, run the following command:

```bash
cd train
sbatch train.sh
```

### Step 3: Interacting with the Project

\TODO This section is outdated and needs to be updated.

For evaluating the performance of the AI-Zero Chess Bot, we have a jupyter notebook that can be used to interactively evaluate the bot's performance against a baseline chess bot and track its improvement over time. The notebook will also provide visualizations and metrics to assess the bot's learning progress. The notebook will be deployed on the cluster and will have access to the cluster's GPU for running the bot evaluations. In the notebook, we will be able to play against the bot and observe its moves and strategies in real-time.

To run the evaluation notebook, follow these steps:

1. **Setup the eval Build**: The evaluation notebook requires a new compiled build with the evaluation mode enabled. Run the following commands to set up the evaluation build:

    ```bash
    cd eval
    ./setup_eval_build.sh
    ```

    This will create a new build with the evaluation mode enabled.
2. **Open the Evaluation Notebook**: Open the `eval.ipynb` notebook in Jupyter.
3. **[Optional] Download Model Weights**: If you want to use pre-trained model weights for the evaluation, download the model weights from [here](/documentation/) (TODO: Add link) and place them in the `train` directory.
4. **Run the Notebook**: Execute the cells in the notebook to start the evaluation process. The notebook will guide you through the evaluation steps and display the bot's performance metrics and visualizations.
