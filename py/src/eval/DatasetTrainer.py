import os
import sys
import torch
from torch.utils.data import DataLoader

from src.Network import Network
from src.eval.ModelEvaluation import ModelEvaluation
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TRAINING_ARGS, TensorboardWriter
from src.train.Trainer import Trainer

NUM_EPOCHS = 20
BATCH_SIZE = 256
DATASET_PERCENTAGE = 0.9


def train_model(model: Network, dataloader: DataLoader, num_epochs: int, iteration: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)
    trainer = Trainer(model, optimizer, TRAINING_ARGS.training)

    print('Training with lr:', TRAINING_ARGS.training.learning_rate(iteration))

    with TensorboardWriter():
        for epoch in range(num_epochs):
            stats = trainer.train(dataloader, iteration)
            print(f'Epoch {epoch+1}/{num_epochs} done: {stats}')


if __name__ == '__main__':
    # read in the dataset_path from sys.argv
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m src.eval.DatasetTrainer <dataset_path1> <dataset_path2> ...')
        sys.exit(1)

    dataset_paths = sys.argv[1:]

    for dataset_path in dataset_paths:
        assert os.path.exists(dataset_path), f'Dataset path does not exist: {dataset_path}'

    # Instantiate the dataset
    dataset = SelfPlayDataset()
    for dataset_path in dataset_paths:
        dataset += SelfPlayDataset.load(dataset_path)
    dataset.deduplicate()

    train, test = torch.utils.data.random_split(
        dataset, [int(DATASET_PERCENTAGE * len(dataset)), len(dataset) - int(DATASET_PERCENTAGE * len(dataset))]
    )

    # Create a DataLoader
    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model
    model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size, device=device)

    # Train the model
    print('Starting training...')
    model.print_params()
    print('Number of training samples:', len(train))

    for iter in range(NUM_EPOCHS - 1, -1, -1):
        if os.path.exists(f'reference/model_{iter}.pt'):
            model.load_state_dict(torch.load(f'reference/model_{iter}.pt', weights_only=True, map_location=device))
            print(f'Loaded model_{iter}.pt')
            break

    for iter in range(NUM_EPOCHS):
        train_model(model, train_dataloader, num_epochs=1, iteration=iter)

        # Evaluate the model
        policy_accuracy, avg_value_loss = ModelEvaluation._evaluate_model_vs_dataset(model, test_dataloader)
        print(f'Policy Accuracy: {policy_accuracy*100:.2f}%, Avg Value Loss: {avg_value_loss}')

        torch.save(model.state_dict(), f'reference/model_{iter}.pt')
