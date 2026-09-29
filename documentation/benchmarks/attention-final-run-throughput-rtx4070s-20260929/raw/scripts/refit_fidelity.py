"""Compares a refitted engine with one built directly from the same weights on evaluation positions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from tools.benchmark_tensorrt_inference import BATCH_SIZE, _TensorRtCudaGraphRunner
from tools.build_tensorrt_refit_template import RefitMode, build_template
from tools.measure_inference_precision_agreement import load_positions
from tools.tensorrt_benchmark_metrics import measure_fidelity

perturbed_model, refitted_engine, dataset, output_directory = (Path(argument) for argument in sys.argv[1:5])
device = torch.device('cuda', 0)
torch.cuda.set_device(device)
direct_engine = output_directory / 'perturbed-direct.engine'
build_template(perturbed_model, direct_engine, BATCH_SIZE, 52, 8, 8, 3, None, RefitMode.ALL)
states, legal_action_mask = load_positions(dataset, BATCH_SIZE)
states = states.to(torch.int8)
direct = _TensorRtCudaGraphRunner(direct_engine, states, device, 5).outputs()
refitted = _TensorRtCudaGraphRunner(refitted_engine, states, device, 5).outputs()
print('REFIT_FIDELITY', json.dumps(measure_fidelity(direct, refitted, legal_action_mask).model_dump()))
