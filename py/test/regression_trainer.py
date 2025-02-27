import os
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
from src.util.save_paths import create_model, create_optimizer

NUM_EPOCHS = 10
BATCH_SIZE = 512


def train_model(model: Network, dataloader: DataLoader, num_epochs: int, iteration: int) -> None:
    def learning_rate(iteration: int) -> float:
        if iteration < 8:
            return 0.2
        return 0.02

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

    run_id = get_run_id()
    with TensorboardWriter(run_id, 'dataset_trainer'):
        # Use the same dataset for training and testing, to test whether the model can overfit
        test_dataset = SelfPlayDataset.load(dataset_paths.pop())
        test_dataset.deduplicate()

        # Instantiate the model
        model = create_model(TRAINING_ARGS.network, device=device)

        # Train the model
        log('Starting training...')
        model.print_params()
        log('Training for', NUM_EPOCHS, 'epochs')

        os.makedirs(save_folder, exist_ok=True)

        for iter in range(NUM_EPOCHS):
            # Instantiate the dataset
            dataset = SelfPlayTrainDataset(run_id, device=device)
            dataset.load_from_files(save_folder, [(0, [Path(p)]) for p in dataset_paths])

            # Create a DataLoader
            train_dataloader = dataset.as_dataloader(TRAINING_ARGS.training.batch_size, num_workers=1)

            train_model(model, train_dataloader, num_epochs=1, iteration=iter)

            # Evaluate the model
            policy_at_1, policy_at_5, policy_at_10, avg_value_loss = ModelEvaluation._evaluate_model_vs_dataset(
                model, DataLoader(test_dataset, batch_size=TRAINING_ARGS.training.batch_size, shuffle=False)
            )
            log(f'Evaluation results at iteration {iter}:')
            log(f'    Policy accuracy @1: {policy_at_1*100:.2f}%')
            log(f'    Policy accuracy @5: {policy_at_5*100:.2f}%')
            log(f'    Policy accuracy @10: {policy_at_10*100:.2f}%')
            log(f'    Avg value loss: {avg_value_loss}')

        log('Training finished')

        assert policy_at_1 > 0.95, 'Policy accuracy @1 is too low'
        assert policy_at_5 > 0.95, 'Policy accuracy @5 is too low'
        assert policy_at_10 > 0.95, 'Policy accuracy @10 is too low'
        assert avg_value_loss < 0.1, 'Avg value loss is too high'


if __name__ == '__main__':
    """This test should train a model on a dataset and evaluate it on the same dataset to check whether it can overfit.
    The model should be able to reach a high policy accuracy and a low value loss.
    This test is useful as a regression test for the training process, to ensure that the training process can train a model on a given dataset."""
    assert TRAINING_ARGS.evaluation is not None, 'Evaluation args not set'
    assert os.path.exists(TRAINING_ARGS.evaluation.dataset_path), 'Dataset path does not exist'

    dataset_paths = [TRAINING_ARGS.evaluation.dataset_path] * 5

    main(dataset_paths)
