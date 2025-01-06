import torch
from os import PathLike
from pathlib import Path


from src.Network import Network
from src.alpha_zero.train.TrainingArgs import NetworkParams
from src.util.compile import try_compile
from src.util.log import log
from src.settings import TRAINING_ARGS


def model_save_path(iteration: int) -> Path:
    return Path(TRAINING_ARGS.save_path) / f'model_{iteration}.pt'


def optimizer_save_path(iteration: int) -> Path:
    return Path(TRAINING_ARGS.save_path) / f'optimizer_{iteration}.pt'


def create_model(args: NetworkParams, device: torch.device) -> Network:
    model = Network(args.num_layers, args.hidden_size, device)
    model = try_compile(model)
    return model


def create_optimizer(model: Network) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=1e-4)


def load_model(path: str | PathLike, args: NetworkParams, device: torch.device) -> Network:
    try:
        model = create_model(args, device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model
    except FileNotFoundError:
        log(f'No model found for: {path}')
        return create_model(args, device)


def load_optimizer(path: str | PathLike, model: Network) -> torch.optim.Optimizer:
    try:
        optimizer = create_optimizer(model)
        optimizer.load_state_dict(torch.load(path, map_location=model.device))
        return optimizer
    except FileNotFoundError:
        log(f'No optimizer found for: {path}')
        raise


def load_model_and_optimizer(
    iteration: int, args: NetworkParams, device: torch.device
) -> tuple[Network, torch.optim.Optimizer]:
    if iteration <= 0:
        model = create_model(args, device)
        optimizer = create_optimizer(model)
    else:
        try:
            model = load_model(model_save_path(iteration), args, device)
            optimizer = load_optimizer(optimizer_save_path(iteration), model)
        except FileNotFoundError:
            return load_model_and_optimizer(iteration - 1, args, device)

    log(f'Model and optimizer loaded from iteration {iteration}')
    return model, optimizer


def save_model_and_optimizer(model: Network, optimizer: torch.optim.Optimizer, iteration: int) -> None:
    torch.save(model.state_dict(), model_save_path(iteration))
    torch.save(optimizer.state_dict(), optimizer_save_path(iteration))


def get_latest_model_iteration(max_iteration: int = TRAINING_ARGS.num_iterations) -> int:
    while max_iteration >= 0 and not model_save_path(max_iteration).exists():
        max_iteration -= 1
    return max(max_iteration, 0)
