import torch
from os import PathLike
from time import sleep
from pathlib import Path


from src.Network import Network
from src.train.TrainingArgs import NetworkParams, OptimizerType
from src.util.compile import try_compile
from src.util.log import LogLevel, log


def model_save_path(iteration: int, save_folder: str | PathLike) -> Path:
    return Path(save_folder) / f'model_{iteration}.pt'


def optimizer_save_path(iteration: int, save_folder: str | PathLike) -> Path:
    return Path(save_folder) / f'optimizer_{iteration}.pt'


def create_model(args: NetworkParams, device: torch.device) -> Network:
    model = Network(args.num_layers, args.hidden_size, device)
    model = try_compile(model)
    return model


def create_optimizer(model: Network, type: OptimizerType) -> torch.optim.Optimizer:
    if type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
    elif type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001, amsgrad=True)
    raise ValueError(f'Optimizer type {type} not supported. Supported types: adamw, sgd')


def load_model(path: str | PathLike, args: NetworkParams, device: torch.device) -> Network:
    model = create_model(args, device)
    try:
        for _ in range(5):
            try:
                data = torch.load(path, map_location=device, weights_only=True)
                break
            except EOFError:
                sleep(1)
        else:
            log(f'Could not load model from: {path}')
            return model

    except FileNotFoundError:
        log(f'No model found for: {path}')
        return model

    try:
        model.load_state_dict(data)
    except RuntimeError:
        # check if any key contains "_orig_mod." if so, try to load without it, else try to load
        contains_org = any('_orig_mod.' in key for key in data.keys())

        if contains_org:
            assert all('_orig_mod.' in key for key in data.keys()), 'Some keys contain "_orig_mod." and some do not'

            log(f'Could not load model from: {path}, trying to load without compilation')
            # replace all key prefixes or "_orig_mod." with ""

            data = {key.replace('_orig_mod.', ''): value for key, value in data.items()}
            model.load_state_dict(data)
        else:
            assert all('_orig_mod.' not in key for key in data.keys()), 'Some keys contain "_orig_mod." and some do not'
            log(f'Could not load model from: {path}, trying to load with compilation')

            data = {f'_orig_mod.{key}': value for key, value in data.items()}

        try:
            model.load_state_dict(data)
        except RuntimeError:
            log(f'Could not load model from: {path}')
            raise
    log(f'Model loaded from: {path}', level=LogLevel.DEBUG)
    return model


def load_optimizer(path: str | PathLike, model: Network, type: OptimizerType) -> torch.optim.Optimizer:
    optimizer = create_optimizer(model, type)
    try:
        for _ in range(5):
            try:
                data = torch.load(path, weights_only=True)
                break
            except EOFError:
                sleep(1)
        else:
            log(f'Could not load optimizer from: {path}')
            raise FileNotFoundError

    except FileNotFoundError:
        log(f'No optimizer found for: {path}')
        raise

    optimizer.load_state_dict(data)
    return optimizer


def load_model_and_optimizer(
    iteration: int, args: NetworkParams, device: torch.device, save_folder: str | PathLike, type: OptimizerType
) -> tuple[Network, torch.optim.Optimizer]:
    if iteration <= 0:
        model = create_model(args, device)
        optimizer = create_optimizer(model, type)
    else:
        try:
            model = load_model(model_save_path(iteration, save_folder), args, device)
            try:
                optimizer = load_optimizer(optimizer_save_path(iteration, save_folder), model, type)
            except:  # noqa: E722
                optimizer = create_optimizer(model, type)
        except FileNotFoundError:
            return load_model_and_optimizer(iteration - 1, args, device, save_folder, type)

    log(f'Model and optimizer loaded from iteration {iteration}')
    return model, optimizer


def save_model_and_optimizer(
    model: Network, optimizer: torch.optim.Optimizer, iteration: int, save_folder: str | PathLike
) -> None:
    torch.save(model.state_dict(), model_save_path(iteration, save_folder))
    torch.save(optimizer.state_dict(), optimizer_save_path(iteration, save_folder))


def get_latest_model_iteration(save_folder: str | PathLike) -> int:
    from src.settings import TRAINING_ARGS

    max_iteration = TRAINING_ARGS.num_iterations
    while max_iteration >= 0 and not model_save_path(max_iteration, save_folder).exists():
        max_iteration -= 1
    return max(max_iteration, 0)
