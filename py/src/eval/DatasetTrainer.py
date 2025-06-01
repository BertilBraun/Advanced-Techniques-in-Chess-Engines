import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.cluster.TrainerProcess import as_dataloader
from src.settings import TRAINING_ARGS, TensorboardWriter, CurrentGame, get_run_id, learning_rate
from src.Network import Network
from src.eval.ModelEvaluation import ModelEvaluation
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.train.RollingSelfPlayBuffer import RollingSelfPlayBuffer
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingParams
from src.util.log import log
from src.util.save_paths import (
    create_model,
    create_optimizer,
    load_model_and_optimizer,
    model_save_path,
    save_model_and_optimizer,
)

NUM_EPOCHS = 6


def train_model(
    model: Network,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    iteration: int,
) -> None:
    trainer = Trainer(
        model,
        optimizer,
        TrainingParams(
            num_epochs=num_epochs,
            optimizer='adamw',
            batch_size=TRAINING_ARGS.training.batch_size,
            sampling_window=lambda _: 1,
            learning_rate=learning_rate,
            learning_rate_scheduler=lambda _, lr: lr,
            num_workers=2,
        ),
    )

    log(
        'Training with lr:',
        trainer.args.learning_rate(iteration, trainer.args.optimizer),
        'and batch size:',
        trainer.args.batch_size,
    )

    for epoch in range(num_epochs):
        trainer.train(dataloader, test_dataloader, iteration)


def main(dataset_paths: list[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_folder = f'reference/{CurrentGame.__class__.__name__}'

    for dataset_path in dataset_paths:
        assert os.path.exists(dataset_path), f'Dataset path does not exist: {dataset_path}'

    log(f'Starting training with {len(dataset_paths)} datasets!')

    run_id = get_run_id()
    with TensorboardWriter(run_id, 'dataset_trainer'):
        log('Loading datasets...')
        test_dataset = SelfPlayDataset.load(dataset_paths.pop())
        test_dataset = test_dataset.deduplicate()
        test_dataloader = as_dataloader(
            test_dataset,
            batch_size=TRAINING_ARGS.training.batch_size,
            num_workers=1,
        )

        train_dataset = RollingSelfPlayBuffer(max_buffer_samples=4_000_000)
        train_dataset.update(0, 1, [Path(p) for p in dataset_paths])
        train_dataset.log_all_dataset_stats(run_id)
        train_stats = train_dataset.stats
        train_dataloader = as_dataloader(
            train_dataset,
            batch_size=TRAINING_ARGS.training.batch_size,
            num_workers=1,
        )

        log('Creating model...')
        # Instantiate the model
        model = create_model(TRAINING_ARGS.network, device=device)
        optimizer = create_optimizer(model, TRAINING_ARGS.training.optimizer)

        # Train the model
        log('Starting training...')
        model.print_params()
        log('Number of training samples:', train_stats.num_samples, 'on', train_stats.num_games, 'games')
        log('Evaluating on', test_dataset.stats.num_samples, 'samples on', test_dataset.stats.num_games, 'games')
        log('Training for', NUM_EPOCHS, 'epochs')

        os.makedirs(save_folder, exist_ok=True)

        for pre_iter in range(NUM_EPOCHS - 1, -1, -1):
            if model_save_path(pre_iter, save_folder).exists():
                model, optimizer = load_model_and_optimizer(
                    pre_iter,
                    TRAINING_ARGS.network,
                    device,
                    save_folder,
                    TRAINING_ARGS.training.optimizer,
                )
                log(f'Loaded model_{pre_iter}.pt')
                break

        for iter in range(pre_iter, NUM_EPOCHS):
            # Create a DataLoader
            train_model(model, optimizer, train_dataloader, test_dataloader, num_epochs=1, iteration=iter)

            # Evaluate the model
            policy_at_1, policy_at_5, policy_at_10, avg_value_loss = ModelEvaluation._evaluate_model_vs_dataset(
                model, test_dataloader
            )
            log(f'Evaluation results at iteration {iter}:')
            log(f'    Policy accuracy @1: {policy_at_1*100:.2f}%')
            log(f'    Policy accuracy @5: {policy_at_5*100:.2f}%')
            log(f'    Policy accuracy @10: {policy_at_10*100:.2f}%')
            log(f'    Avg value loss: {avg_value_loss}')

            save_model_and_optimizer(model, optimizer, iter, save_folder)

        log('Training finished')


if __name__ == '__main__':
    # read in the dataset_path from sys.argv
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.eval.DatasetTrainer <dataset_path1> <dataset_path2> ...')
        sys.exit(1)
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    dataset_paths = sys.argv[1:]

    main(dataset_paths)
