from collections import defaultdict
import os
import sys
import torch
from torch.utils.data import DataLoader

from src.Network import Network
from src.eval.ModelEvaluation import ModelEvaluation
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayTrainDataset import SelfPlayTrainDataset
from src.settings import TRAINING_ARGS, TensorboardWriter, CurrentGame
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingParams

NUM_EPOCHS = 10
BATCH_SIZE = 512


def train_model(model: Network, dataloader: DataLoader, num_epochs: int, iteration: int) -> None:
    lr = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.09, weight_decay=1e-4, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # IDK_DIDNT_FIX_IT other optimizer?
    # IDK_DIDNT_FIX_IT way lower learning rate?
    # Value loss just explodes into oblivion
    # DONE Does it learn on different samples - Yes, all samples are unique and properly loaded
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

    print('Training with lr:', TRAINING_ARGS.training.learning_rate(iteration))

    for epoch in range(num_epochs):
        stats = trainer.train(dataloader, iteration)
        print(f'Epoch {epoch+1}/{num_epochs} done: {stats}')


def main(dataset_paths: list[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_folder = f'reference/{CurrentGame.__class__.__name__}'

    for dataset_path in dataset_paths:
        assert os.path.exists(dataset_path), f'Dataset path does not exist: {dataset_path}'

    with TensorboardWriter('dataset_trainer'):
        test_dataset = SelfPlayDataset.load(dataset_paths.pop())

        # group the datasets by the iteration id. These are of the form 'memory_ITERATIONNUMBER_*.hdf5'
        grouped_paths = defaultdict(list)
        for dataset_path in dataset_paths:
            iteration = int(dataset_path.split('/')[-1].split('_')[1])
            grouped_paths[iteration].append(dataset_path)

        # Instantiate the dataset
        dataset = SelfPlayTrainDataset(chunk_size=BATCH_SIZE * 2000, device=device)
        dataset.load_from_files(save_folder, list(grouped_paths.values()))

        # Create a DataLoader
        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Instantiate the model
        model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size, device=device)

        # Train the model
        print('Starting training...')
        model.print_params()
        print('Number of training samples:', dataset.stats.num_samples, 'on', dataset.stats.num_games, 'games')
        print('Evaluating on', test_dataset.stats.num_samples, 'samples on', test_dataset.stats.num_games, 'games')
        print('Training for', NUM_EPOCHS, 'epochs')

        os.makedirs(save_folder, exist_ok=True)

        for iter in range(NUM_EPOCHS - 1, -1, -1):
            if os.path.exists(f'{save_folder}/model_{iter}.pt'):
                model.load_state_dict(
                    torch.load(f'{save_folder}/model_{iter}.pt', weights_only=True, map_location=device)
                )
                print(f'Loaded model_{iter}.pt')
                break

        for iter in range(NUM_EPOCHS):
            train_model(model, train_dataloader, num_epochs=1, iteration=iter)

            # Evaluate the model
            policy_at_1, policy_at_5, policy_at_10, avg_value_loss = ModelEvaluation._evaluate_model_vs_dataset(
                model, test_dataloader
            )
            print(f'Evaluation results at iteration {iter}:')
            print(f'    Policy accuracy @1: {policy_at_1*100:.2f}%')
            print(f'    Policy accuracy @5: {policy_at_5*100:.2f}%')
            print(f'    Policy accuracy @10: {policy_at_10*100:.2f}%')
            print(f'    Avg value loss: {avg_value_loss}')

            torch.save(model.state_dict(), f'{save_folder}/model_{iter}.pt')


if __name__ == '__main__':
    # read in the dataset_path from sys.argv
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.eval.DatasetTrainer <dataset_path1> <dataset_path2> ...')
        sys.exit(1)

    dataset_paths = sys.argv[1:]

    main(dataset_paths)
