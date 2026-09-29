"""Refits a template with the same TorchScript graph but perturbed weights, then diagnoses unrefitted weights."""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
import tensorrt as trt
import torch
from tools.publish_tensorrt_engine import export_onnx, refit_engine

model_path, template_engine, output_directory = (Path(argument) for argument in sys.argv[1:4])
output_directory.mkdir(parents=True, exist_ok=True)
model = torch.jit.load(str(model_path), map_location='cpu')
with torch.no_grad():
    for parameter in model.parameters():
        parameter.mul_(1.01).add_(0.001)
perturbed_path = output_directory / 'perturbed.pt'
torch.jit.save(model, str(perturbed_path))
perturbed_onnx = output_directory / 'perturbed-b320-fp16.onnx'
export_onnx(perturbed_path, perturbed_onnx, (320, 52, 8, 8))
try:
    refit_engine(template_engine, perturbed_onnx, output_directory / 'perturbed-refitted.engine')
    print('SAME_STRUCTURE_REFIT ok')
except ValueError as error:
    print('SAME_STRUCTURE_REFIT failed', error)

runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
engine = runtime.deserialize_cuda_engine(template_engine.read_bytes())
refitter = trt.Refitter(engine, trt.Logger(trt.Logger.ERROR))
engine_weights = set(refitter.get_all_weights())
for label, onnx_path in (('perturbed', perturbed_onnx), ('trained', Path(sys.argv[4]))):
    initializers = {initializer.name for initializer in onnx.load(onnx_path, load_external_data=False).graph.initializer}
    missing = sorted(engine_weights - initializers)
    print(f'{label}: engine weights {len(engine_weights)}, onnx initializers {len(initializers)}, '
          f'engine weights missing from onnx {len(missing)}: {missing[:8]}')
