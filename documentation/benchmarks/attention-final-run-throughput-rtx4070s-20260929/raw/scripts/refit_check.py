"""Refits a template built from random weights with trained weights and compares it with a direct build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from tools.benchmark_tensorrt_inference import BATCH_SIZE, _TensorRtCudaGraphRunner
from tools.build_tensorrt_refit_template import RefitMode, build_template
from tools.measure_inference_precision_agreement import load_positions
from tools.publish_tensorrt_engine import export_onnx, refit_engine
from tools.tensorrt_benchmark_metrics import ModelOutputs, measure_fidelity

template_model, trained_model, direct_engine, dataset, output_directory = (Path(argument) for argument in sys.argv[1:6])
output_directory.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda', 0)
torch.cuda.set_device(device)

template_engine = output_directory / 'template-b320.engine'
build_template(template_model, template_engine, BATCH_SIZE, 52, 8, 8, 3, None, RefitMode.ALL)
trained_onnx = output_directory / 'trained-b320-fp16.onnx'
export_onnx(trained_model, trained_onnx, (BATCH_SIZE, 52, 8, 8))
refitted_engine = output_directory / 'refitted-b320.engine'
refit_engine(template_engine, trained_onnx, refitted_engine)

states, legal_action_mask = load_positions(dataset, BATCH_SIZE)
states = states.to(torch.int8)


def outputs(engine: Path) -> ModelOutputs:
    return _TensorRtCudaGraphRunner(engine, states, device, 5).outputs()


direct = outputs(direct_engine)
refitted = outputs(refitted_engine)
untrained = outputs(template_engine)
report = {
    'refitted_against_direct_build': measure_fidelity(direct, refitted, legal_action_mask).model_dump(),
    'unrefitted_template_against_direct_build': measure_fidelity(direct, untrained, legal_action_mask).model_dump(),
}
print(json.dumps(report, indent=2))
(output_directory / 'refit-report.json').write_text(json.dumps(report, indent=2) + '\n')
