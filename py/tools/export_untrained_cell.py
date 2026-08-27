from __future__ import annotations

import argparse
from pathlib import Path

import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.checkpoint.persistence import create_model, create_optimizer, save_model_and_optimizer
from src.training.model_cost import format_model_cost, measure_model_cost
from src.util.log import log
from tools.attention_viability_cells import cell_by_name
from tools.distill_train_student import student_architecture


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export an untrained checkpoint of one cell. Throughput depends only on the architecture.'
    )
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output-run-state', required=True, type=Path)
    parser.add_argument('--generation', default=322, type=int)
    parser.add_argument('--random-seed', default=20260827, type=int)
    namespace = parser.parse_args()
    if namespace.generation == 0:
        raise ValueError('Generation 0 would calibrate the bootstrap prior, which an untrained probe must not do.')

    torch.manual_seed(namespace.random_seed)
    architecture = student_architecture(cell_by_name(namespace.cell).arguments)
    model = create_model(architecture, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    namespace.output_run_state.mkdir(parents=True, exist_ok=True)
    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), namespace.generation, namespace.output_run_state)
    log(format_model_cost(namespace.cell, measure_model_cost(model)))
    log(f'Wrote an untrained {namespace.cell} to {namespace.output_run_state}.')


if __name__ == '__main__':
    main()
