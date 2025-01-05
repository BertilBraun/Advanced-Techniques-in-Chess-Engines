import torch
import optuna
from typing import Callable
from torch.optim import AdamW

from src.Network import Network
from src.alpha_zero.AlphaZero import AlphaZero
from src.alpha_zero.train.TrainingArgs import TrainingArgs, MCTSParams, NetworkParams, SelfPlayParams, TrainingParams
from src.cluster.ClusterManager import ClusterManager
from src.settings import SAVE_PATH, USE_GPU
from src.util.compile import try_compile
from src.util.log import log


def objective(rank: int) -> Callable[[optuna.Trial], float]:
    def _objective(trial: optuna.Trial) -> float:
        # Define hyperparameter sampling
        mcts_num_searches_per_turn = trial.suggest_int('mcts_num_searches_per_turn', 50, 200, step=50)
        mcts_dirichlet_epsilon = trial.suggest_float('mcts_dirichlet_epsilon', 0.0, 1.0, step=0.25)
        mcts_dirichlet_alpha = trial.suggest_float('mcts_dirichlet_alpha', 0.2, 0.4, step=0.1)
        mcts_c_param = trial.suggest_float('mcts_c_param', 1.0, 6.0, step=1)

        network_num_layers = trial.suggest_int('network_num_layers', 2, 8, step=3)
        network_hidden_size_exponent = trial.suggest_int('network_hidden_size_exponent', 5, 9, step=2)  # 32, 128, 512
        network_hidden_size = 2**network_hidden_size_exponent

        if network_num_layers > 2 and network_hidden_size_exponent > 5:
            raise optuna.TrialPruned()  # Skip this trial - too large for the current hardware

        selfplay_temperature = trial.suggest_float('selfplay_temperature', 0.2, 1.8, step=0.8)
        selfplay_num_games_per_iteration = trial.suggest_int(
            'selfplay_num_games_per_iteration', 512 - 128, 512 + 128, step=128
        )

        training_num_epochs = trial.suggest_int('training_num_epochs', 2, 6, step=2)
        training_batch_size_exponent = trial.suggest_int('training_batch_size_exponent', 4, 8, step=2)  # 16, 64, 256
        training_batch_size = 2**training_batch_size_exponent
        training_learning_rate_initial = trial.suggest_float('training_learning_rate_initial', 5e-3, 1e-1, log=True)
        training_decay_rate = trial.suggest_float('learning_rate_decay_rate', 0.85, 0.99)

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
        )

        training_args = TrainingArgs(
            save_path=f'{SAVE_PATH}/optuna_trial_{trial.number}',
            num_iterations=6,
            mcts=mcts_params,
            network=network_params,
            self_play=self_play_params,
            training=training_params,
        )

        # Run the training loop
        device = torch.device('cuda', rank % torch.cuda.device_count()) if USE_GPU else torch.device('cpu')
        model = Network(training_args.network.num_layers, training_args.network.hidden_size, device=device)
        model = try_compile(model)

        optimizer = AdamW(model.parameters(), lr=training_learning_rate_initial, weight_decay=1e-4, amsgrad=True)

        log(f'Running trial {trial.number} with the following hyperparameters:')
        log(trial.params, use_pprint=True)

        for iteration, stat in AlphaZero(model, optimizer, training_args).learn():
            trial.report(stat.total_loss / stat.num_batches, step=iteration)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # Extract the final total_loss
        return stat.total_loss / stat.num_batches

    return _objective


if __name__ == '__main__':
    SEPERATE_NODES = 4
    NUM_TRIALS = 80
    TIMEOUT = 600  # 10 minutes

    study_name = 'alpha_zero_hyperparameter_optimization'
    storage = f'sqlite:///{study_name}.db'

    cluster_manager = ClusterManager(SEPERATE_NODES)
    cluster_manager.initialize()

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)

    if cluster_manager.is_root_node:
        log('Starting hyperparameter optimization')
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    cluster_manager.barrier('study_setup')

    if not cluster_manager.is_root_node:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

    study.optimize(objective(cluster_manager.rank), n_trials=NUM_TRIALS, timeout=TIMEOUT)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    log('Study statistics:')
    log(f'  Number of finished trials: {len(study.trials)}')
    log(f'  Number of pruned trials: {len(pruned_trials)}')
    log(f'  Number of complete trials: {len(complete_trials)}')

    log(f'Best trial (Trial: {study.best_trial.number}) with loss {round(study.best_value, 3)}:')
    log(study.best_trial.params)

    log()
    log('Most relevant hyperparameters:')
    print(optuna.importance.get_param_importances(study))

"""After Optimizing for TicTacToe:

Study statistics:
  Number of finished trials: 88
  Number of pruned trials: 20
  Number of complete trials: 28

Best trial (Trial: 79) with loss 0.698 (after 6 iterations):
{'mcts_num_searches_per_turn': 100, 'mcts_dirichlet_epsilon': 0.0, 'mcts_dirichlet_alpha': 0.2, 'mcts_c_param': 1.0, 'network_num_layers': 2, 'network_hidden_size_exponent': 5, 'selfplay_temperature': 1.8, 'selfplay_num_games_per_iteration': 512, 'training_num_epochs': 4, 'training_batch_size_exponent': 8, 'training_learning_rate_initial': 0.02310216465004561, 'learning_rate_decay_rate': 0.8624339039875765, 'sampling_window_initial': 3, 'sampling_window_max': 25}

Most relevant hyperparameters:
{'mcts_dirichlet_epsilon': np.float64(0.34450795717547333), 'sampling_window_max': np.float64(0.17085732233001472), 'training_batch_size_exponent': np.float64(0.15131826914727717), 'mcts_c_param': np.float64(0.1443145208637471), 'learning_rate_decay_rate': np.float64(0.09779456325483078), 'selfplay_temperature': np.float64(0.05329892198678179), 'network_hidden_size_exponent': np.float64(0.01277041094967793), 'mcts_num_searches_per_turn': np.float64(0.009384432920882502), 'selfplay_num_games_per_iteration': np.float64(0.005003434178581438), 'sampling_window_initial': np.float64(0.004917520690605678), 'network_num_layers': np.float64(0.004213624212112882), 'training_num_epochs': np.float64(0.0016190222900145524)}
"""
