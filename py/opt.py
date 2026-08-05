import torch
import optuna
from multiprocessing import Process
from time import monotonic

from src.util.log import log
from src.train.TrainingArgs import (
    ClusterParams,
    OptimizerType,
    TrainingArgs,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingParams,
)
from src.cluster.CommanderProcess import CommanderProcess
from src.settings import DEFAULT_RUNTIME_LIMITS, SAVE_PATH, learning_rate_scheduler


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True


SEPERATE_NODES = 4
NUM_TRIALS = 80
TIMEOUT = 600  # 10 minutes

STUDY_NAME = 'alpha_zero_hyperparameter_optimization'
STORAGE = f'sqlite:///{STUDY_NAME}.db'


def objective(trial: optuna.Trial) -> float:
    # Define hyperparameter sampling
    mcts_num_searches_per_turn = trial.suggest_int('mcts_num_searches_per_turn', 50, 200, step=50)
    mcts_dirichlet_alpha = trial.suggest_float('mcts_dirichlet_alpha', 0.2, 0.4, step=0.1)
    mcts_c_param = trial.suggest_float('mcts_c_param', 1.0, 6.0, step=1)

    network_num_layers = trial.suggest_int('network_num_layers', 2, 8, step=3)
    network_hidden_size_exponent = trial.suggest_int('network_hidden_size_exponent', 5, 9, step=2)  # 32, 128, 512
    network_hidden_size = 2**network_hidden_size_exponent

    if network_num_layers > 2 and network_hidden_size_exponent > 5:
        raise optuna.TrialPruned()  # Skip this trial - too large for the current hardware

    num_games_per_iteration = trial.suggest_int('num_games_per_iteration', 512 - 128, 512 + 128, step=128)

    training_optimizer = trial.suggest_categorical('training_optimizer', ['adamw', 'sgd'])
    training_num_epochs = trial.suggest_int('training_num_epochs', 2, 6, step=2)
    training_batch_size_exponent = trial.suggest_int('training_batch_size_exponent', 4, 8, step=2)  # 16, 64, 256
    training_batch_size = 2**training_batch_size_exponent
    training_learning_rate_initial = trial.suggest_float('training_learning_rate_initial', 5e-3, 1e-1, log=True)
    training_decay_rate = trial.suggest_float('learning_rate_decay_rate', 0.85, 0.99)

    def learning_rate(current_iteration: int, optimizer: OptimizerType) -> float:
        # optimizer can be ignored, as the search is exploring lr seperately for each optimizer
        return training_learning_rate_initial * (training_decay_rate**current_iteration)

    initial_window = trial.suggest_int('sampling_window_initial', 2, 5)
    max_window = trial.suggest_int('sampling_window_max', 5, 10, step=5)

    def sampling_window(current_iteration: int) -> int:
        return min(initial_window + current_iteration, max_window)

    # Configure TrainingArgs
    mcts_params = MCTSParams(
        num_searches_per_turn=mcts_num_searches_per_turn,
        num_parallel_searches=8,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=mcts_dirichlet_alpha,
        c_param=mcts_c_param,
        min_visit_count=mcts_num_searches_per_turn // 50,
        percentage_of_node_visits_to_keep=0.6,
        num_threads=8,  # Adjust based on your hardware
    )

    network_params = NetworkParams(num_layers=network_num_layers, hidden_size=network_hidden_size)

    self_play_params = SelfPlayParams(
        num_parallel_games=128,
        inference_cache_capacity=250_000,
        use_inference_cache=True,
        mcts=mcts_params,
        num_moves_after_which_to_play_greedy=10,
    )

    training_params = TrainingParams(
        optimizer=training_optimizer,  # type: ignore
        num_epochs=training_num_epochs,
        global_batch_size=training_batch_size,
        local_batch_size=training_batch_size,
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    )

    training_args = TrainingArgs(
        save_path=f'{SAVE_PATH}/optuna_trial_{trial.number}',
        num_iterations=6,
        num_games_per_iteration=num_games_per_iteration,
        network=network_params,
        self_play=self_play_params,
        training=training_params,
        cluster=ClusterParams(
            trainer_device_type='cuda' if torch.cuda.is_available() else 'cpu',
            trainer_process_group_backend='nccl' if torch.cuda.is_available() else 'gloo',
            trainer_rank_zero_device_id=max(0, torch.cuda.device_count() - 1),
            trainer_ddp_device_ids=(max(0, torch.cuda.device_count() - 1),),
            evaluation_device_cycle=(max(0, torch.cuda.device_count() - 1),),
            self_play_device_ids=(max(0, torch.cuda.device_count() - 1),) * 2,
            self_play_tensorboard_processes=1,
            trainer_cpu_threads=1,
            trainer_interop_threads=1,
            self_play_node_ids_to_pause_during_training=(),
            max_concurrent_evaluations=1,
        ),
        run_limits=DEFAULT_RUNTIME_LIMITS,
        random_seed=trial.number,
    )

    log(f'Running trial {trial.number} with the following hyperparameters:')
    log(trial.params, use_pprint=True)

    commander = CommanderProcess(trial.number, training_args, monotonic())
    for iteration, training_stats in commander.run():
        trial.report(training_stats.total_loss, step=iteration)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # Extract the final total_loss
    return training_stats.total_loss  # type: ignore[return-value]


def optimizer() -> None:
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=NUM_TRIALS, timeout=TIMEOUT)


if __name__ == '__main__':
    log('Starting hyperparameter optimization')
    study = optuna.create_study(
        direction='minimize',
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
    )

    processes: list[Process] = [Process(target=optimizer) for _ in range(SEPERATE_NODES)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

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
    log(optuna.importance.get_param_importances(study))
