from collections import defaultdict
import os
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

from src.Network import Network
from src.eval.ModelEvaluation import ModelEvaluation
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayTrainDataset import SelfPlayTrainDataset
from src.settings import TRAINING_ARGS, TensorboardWriter, CurrentGame, get_run_id
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingParams
from src.util.log import log
from src.util.save_paths import create_model

NUM_EPOCHS = 10
BATCH_SIZE = 512


def train_model(model: Network, dataloader: DataLoader, num_epochs: int, iteration: int) -> None:
    lr = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.09, weight_decay=1e-4, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model,
        optimizer,
        TrainingParams(
            num_epochs=num_epochs,
            batch_size=BATCH_SIZE,
            sampling_window=lambda _: 1,
            learning_rate=lambda i: lr * 0.9 ** (i + 1),
            learning_rate_scheduler=lambda _, lr: lr,
            num_workers=0,
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

    with TensorboardWriter(get_run_id(), 'dataset_trainer'):
        test_dataset = SelfPlayDataset.load(dataset_paths.pop())

        # group the datasets by the iteration id. These are of the form 'memory_ITERATIONNUMBER_*.hdf5'
        grouped_paths = defaultdict(list)
        for dataset_path in dataset_paths:
            iteration = int(dataset_path.split('/')[-1].split('_')[1])
            grouped_paths[iteration].append(Path(dataset_path))

        # Instantiate the dataset
        dataset = SelfPlayTrainDataset(chunk_size=BATCH_SIZE * 2000, device=device)
        dataset.load_from_files(save_folder, list(grouped_paths.values()))

        # Create a DataLoader
        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Instantiate the model
        model = create_model(TRAINING_ARGS.network, device=device)

        # Train the model
        log('Starting training...')
        model.print_params()
        log('Number of training samples:', dataset.stats.num_samples, 'on', dataset.stats.num_games, 'games')
        log('Evaluating on', test_dataset.stats.num_samples, 'samples on', test_dataset.stats.num_games, 'games')
        log('Training for', NUM_EPOCHS, 'epochs')

        os.makedirs(save_folder, exist_ok=True)

        for pre_iter in range(NUM_EPOCHS - 1, -1, -1):
            if os.path.exists(f'{save_folder}/model_{pre_iter}.pt'):
                model.load_state_dict(
                    torch.load(f'{save_folder}/model_{pre_iter}.pt', weights_only=True, map_location=device)
                )
                log(f'Loaded model_{pre_iter}.pt')
                break

        for iter in range(pre_iter, NUM_EPOCHS):
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


if __name__ == '__main__':
    # read in the dataset_path from sys.argv
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.eval.DatasetTrainer <dataset_path1> <dataset_path2> ...')
        sys.exit(1)

    dataset_paths = sys.argv[1:]

    main(dataset_paths)
