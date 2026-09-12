from __future__ import annotations

import argparse
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import onnx
import tensorrt as trt
import torch
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.hashing import file_sha256

INPUT_NAME = 'states'
POLICY_OUTPUT_NAME = 'policy_logits'
WDL_OUTPUT_NAME = 'wdl_probabilities'
ONNX_OPSET_VERSION = 18


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def engine_input_shape(engine: trt.ICudaEngine) -> tuple[int, int, int, int]:
    shape = engine.get_tensor_shape(INPUT_NAME)
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f'TensorRT template has incompatible input shape: {tuple(shape)}')
    return tuple(shape)


def export_onnx(model_path: Path, output_path: Path, input_shape: tuple[int, int, int, int]) -> None:
    model = torch.jit.load(str(model_path), map_location='cpu').to(dtype=torch.float16).eval()
    example = torch.zeros(input_shape, dtype=torch.float16)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(output_path),
            input_names=(INPUT_NAME,),
            output_names=(POLICY_OUTPUT_NAME, WDL_OUTPUT_NAME),
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,
        )
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported, full_check=True)


def refit_engine(template_path: Path, onnx_path: Path, output_path: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(template_path.read_bytes())
    if engine is None:
        raise ValueError(f'Could not deserialize TensorRT template: {template_path}')
    refitter = trt.Refitter(engine, logger)
    parser_refitter = trt.OnnxParserRefitter(refitter, logger)
    if not parser_refitter.refit_from_file(str(onnx_path)):
        raise ValueError(f'TensorRT could not refit from {onnx_path}')
    missing = tuple(sorted(refitter.get_missing_weights()))
    if missing:
        raise ValueError(f'TensorRT refit is missing weights: {missing}')
    if not refitter.refit_cuda_engine():
        raise ValueError('TensorRT engine refit failed')
    write_bytes_atomically(output_path, bytes(engine.serialize()))


def publish(model_path: Path, template_paths: tuple[Path, ...]) -> dict[str, str | int | bool]:
    if not template_paths:
        raise ValueError('At least one TensorRT template is required.')
    engine_path = model_path.with_suffix('.trt.engine')
    metadata_path = engine_path.with_suffix('.json')
    lock_path = engine_path.with_suffix('.lock')
    source_sha256 = file_sha256(model_path)
    template_sha256s = tuple(file_sha256(template_path) for template_path in template_paths)
    with exclusive_lock(lock_path):
        if engine_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (
                metadata.get('source_sha256') == source_sha256
                and metadata.get('template_sha256') in template_sha256s
                and metadata.get('engine_sha256') == file_sha256(engine_path)
            ):
                return {**metadata, 'cached': True}
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        template = runtime.deserialize_cuda_engine(template_paths[0].read_bytes())
        if template is None:
            raise ValueError(f'Could not deserialize TensorRT template: {template_paths[0]}')
        input_shape = engine_input_shape(template)
        owns_onnx = model_path.suffix != '.onnx'
        onnx_path = engine_path.with_suffix('.temporary.onnx') if owns_onnx else model_path
        if owns_onnx:
            onnx_path.unlink(missing_ok=True)
        try:
            if owns_onnx:
                export_onnx(model_path, onnx_path, input_shape)
            else:
                exported = onnx.load(onnx_path)
                onnx.checker.check_model(exported, full_check=True)
            selected_template_path: Path | None = None
            failures: list[str] = []
            for template_path in template_paths:
                try:
                    refit_engine(template_path, onnx_path, engine_path)
                except (TypeError, ValueError) as error:
                    failures.append(f'{template_path}: {error}')
                    continue
                selected_template_path = template_path
                break
            if selected_template_path is None:
                raise ValueError('No TensorRT template accepted the checkpoint:\n' + '\n'.join(failures))
        finally:
            if owns_onnx:
                onnx_path.unlink(missing_ok=True)
        template_sha256 = file_sha256(selected_template_path)
        metadata = {
            'engine_path': str(engine_path),
            'engine_sha256': file_sha256(engine_path),
            'source_sha256': source_sha256,
            'template_sha256': template_sha256,
            'batch_size': input_shape[0],
            'cached': False,
        }
        write_text_atomically(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Atomically refit a TensorRT template for one checkpoint.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--template-engine', type=Path, required=True, action='append')
    arguments = parser.parse_args()
    print(json.dumps(publish(arguments.model, tuple(arguments.template_engine)), sort_keys=True))


if __name__ == '__main__':
    main()
