import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.Network import Network
from src.eval.ModelEvaluation import ModelEvaluation
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayTrainDataset import SelfPlayTrainDataset
from src.settings import TRAINING_ARGS, TensorboardWriter, CurrentGame, get_run_id
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingParams
from src.util.log import log
from src.util.save_paths import create_model, create_optimizer, load_model

NUM_EPOCHS = 50


def train_model(model: Network, dataloader: DataLoader, num_epochs: int, iteration: int) -> None:
    def learning_rate(iteration: int) -> float:
        if iteration < 5:
            return 0.2
        elif iteration < 8:
            return 0.02
        elif iteration < 10:
            return 0.002
        else:
            return 0.0002

    trainer = Trainer(
        model,
        create_optimizer(model),
        TrainingParams(
            num_epochs=num_epochs,
            batch_size=TRAINING_ARGS.training.batch_size,
            sampling_window=lambda _: 1,
            learning_rate=learning_rate,
            learning_rate_scheduler=lambda _, lr: lr,
            num_workers=2,
        ),
    )

    log('Training with lr:', trainer.args.learning_rate(iteration))

    for epoch in range(num_epochs):
        stats = trainer.train(dataloader, iteration)
        log(f'Epoch {epoch+1}/{num_epochs} done: {stats}')


def main(dataset_paths: list[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_folder = f'reference/{CurrentGame.__class__.__name__}'

    for dataset_path in dataset_paths:
        assert os.path.exists(dataset_path), f'Dataset path does not exist: {dataset_path}'

    run_id = get_run_id()
    with TensorboardWriter(run_id, 'dataset_trainer'):
        test_dataset = SelfPlayDataset.load(dataset_paths.pop())
        test_dataset.deduplicate()
        test_dataloader = DataLoader(test_dataset, batch_size=TRAINING_ARGS.training.batch_size, shuffle=False)

        tmp_dataset = SelfPlayTrainDataset(run_id, device=device)
        tmp_dataset.load_from_files(save_folder, [[Path(p)] for p in dataset_paths])
        train_stats = tmp_dataset.stats

        # Instantiate the model
        model = create_model(TRAINING_ARGS.network, device=device)

        # Train the model
        log('Starting training...')
        model.print_params()
        log('Number of training samples:', train_stats.num_samples, 'on', train_stats.num_games, 'games')
        log('Evaluating on', test_dataset.stats.num_samples, 'samples on', test_dataset.stats.num_games, 'games')
        log('Training for', NUM_EPOCHS, 'epochs')

        os.makedirs(save_folder, exist_ok=True)

        for pre_iter in range(NUM_EPOCHS - 1, -1, -1):
            if os.path.exists(f'{save_folder}/model_{pre_iter}.pt'):
                model = load_model(f'{save_folder}/model_{pre_iter}.pt', TRAINING_ARGS.network, device)
                log(f'Loaded model_{pre_iter}.pt')
                break

        for iter in range(pre_iter, NUM_EPOCHS):
            # Instantiate the dataset
            dataset = SelfPlayTrainDataset(run_id, device=device)
            dataset.load_from_files(save_folder, [[Path(p)] for p in dataset_paths])

            # Create a DataLoader
            train_dataloader = dataset.as_dataloader(TRAINING_ARGS.training.batch_size, num_workers=1)

            train_model(model, train_dataloader, num_epochs=1, iteration=iter)

            # Evaluate the model
            policy_at_1, policy_at_5, policy_at_10, avg_value_loss = ModelEvaluation._evaluate_model_vs_dataset(
                model, test_dataloader
            )
            log(f'Evaluation results at iteration {iter}:')
            log(f'    Policy accuracy @1: {policy_at_1*100:.2f}%')
            log(f'    Policy accuracy @5: {policy_at_5*100:.2f}%')
            log(f'    Policy accuracy @10: {policy_at_10*100:.2f}%')
            log(f'    Avg value loss: {avg_value_loss}')

            torch.save(model.state_dict(), f'{save_folder}/model_{iter}.pt')
            dataset.cleanup()

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
