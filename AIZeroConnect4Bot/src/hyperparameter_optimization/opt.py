import src.environ_setup  # isort:skip # noqa import first to setup environment variables and other configurations

import torch
import optuna
from torch.optim import AdamW, SGD

from src.Network import Network
from src.alpha_zero.AlphaZero import AlphaZero
from src.alpha_zero.train.TrainingArgs import (
    TrainingArgs,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingParams,
    ClusterParams,
    EvaluationParams,
)
from src.settings import SAVE_PATH, learning_rate_scheduler
from src.util.compile import try_compile
from src.util.log import log


def objective(trial: optuna.Trial) -> float:
    # Define hyperparameter sampling
    mcts_num_searches_per_turn = trial.suggest_int('mcts_num_searches_per_turn', 50, 200, step=50)
    mcts_dirichlet_epsilon = trial.suggest_float('mcts_dirichlet_epsilon', 0.0, 1.0, step=0.25)
    mcts_dirichlet_alpha = trial.suggest_float('mcts_dirichlet_alpha', 0.0, 1.0, step=0.25)
    mcts_c_param = trial.suggest_float('mcts_c_param', 1.0, 6.0, step=1)

    network_num_layers = trial.suggest_int('network_num_layers', 2, 8, step=3)
    network_hidden_size_exponent = trial.suggest_int('network_hidden_size', 5, 9, step=2)  # 32 to 512
    network_hidden_size = 2**network_hidden_size_exponent

    if network_num_layers > 2 and network_hidden_size_exponent > 5:
        raise optuna.TrialPruned()  # Skip this trial - too large for the current hardware

    selfplay_temperature = trial.suggest_float('selfplay_temperature', 0.2, 1.8, step=0.8)
    selfplay_num_games_per_iteration = trial.suggest_int(
        'selfplay_num_games_per_iteration', 512 - 128, 512 + 128, step=128
    )

    training_num_epochs = trial.suggest_int('training_num_epochs', 2, 6, step=2)
    training_batch_size_exponent = trial.suggest_int('training_batch_size', 4, 8, step=2)  # 16 to 256
    training_batch_size = 2**training_batch_size_exponent
    training_learning_rate_initial = trial.suggest_float('training_learning_rate_initial', 1e-5, 1e-1, log=True)
    training_decay_rate = trial.suggest_float('learning_rate_decay_rate', 0.85, 0.99)

    cluster_params = ClusterParams(num_train_nodes_on_cluster=0, num_self_play_nodes_on_cluster=1)

    evaluation_params = EvaluationParams(num_searches_per_turn=0, num_games=0, every_n_iterations=100)  # No evaluation

    def learning_rate(current_iteration: int) -> float:
        return training_learning_rate_initial * (training_decay_rate**current_iteration)

    initial_window = trial.suggest_int('sampling_window_initial', 2, 5)
    max_window = trial.suggest_int('sampling_window_max', 10, 25, step=5)

    def sampling_window(current_iteration: int) -> int:
        return min(initial_window + current_iteration, max_window)

    # Configure TrainingArgs
    mcts_params = MCTSParams(
        num_searches_per_turn=mcts_num_searches_per_turn,
        dirichlet_epsilon=mcts_dirichlet_epsilon,
        dirichlet_alpha=lambda _: mcts_dirichlet_alpha,
        c_param=mcts_c_param,
    )

    network_params = NetworkParams(num_layers=network_num_layers, hidden_size=network_hidden_size)

    self_play_params = SelfPlayParams(
        temperature=selfplay_temperature,
        num_parallel_games=128,
        num_games_per_iteration=selfplay_num_games_per_iteration,
    )

    training_params = TrainingParams(
        num_epochs=training_num_epochs,
        batch_size=training_batch_size,
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    )

    training_args = TrainingArgs(
        save_path=f'{SAVE_PATH}/optuna_trial_{trial.number}',
        num_iterations=6,
        mcts=mcts_params,
        network=network_params,
        self_play=self_play_params,
        cluster=cluster_params,
        training=training_params,
        evaluation=evaluation_params,
    )

    # Run the training loop
    model = Network(training_args.network.num_layers, training_args.network.hidden_size)
    torch.set_float32_matmul_precision('high')
    model = try_compile(model)

    optimizer_type = trial.suggest_categorical('optimizer_type', ['AdamW', 'SGD'])

    if optimizer_type == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=training_learning_rate_initial, weight_decay=1e-4, amsgrad=True)
    elif optimizer_type == 'SGD':
        optimizer = SGD(
            model.parameters(), lr=training_learning_rate_initial, weight_decay=1e-4, nesterov=True, momentum=0.9
        )
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')

    log(f'Running trial {trial.number} with the following hyperparameters:')
    log(trial.params, use_pprint=True)

    for iteration, stat in AlphaZero(model, optimizer, training_args).learn():
        trial.report(stat.total_loss / stat.num_batches, step=iteration)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # Extract the final total_loss
    return stat.total_loss / stat.num_batches


if __name__ == '__main__':
    # Set up the Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2, interval_steps=1),
        study_name='alpha_zero_hyperparameter_optimization',
        storage='sqlite:///alpha_zero_hyperparameter_optimization.db',
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=50,
        timeout=600,
    )

    print(study.best_params)
