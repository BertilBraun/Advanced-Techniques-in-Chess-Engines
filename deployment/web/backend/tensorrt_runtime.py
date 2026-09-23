from __future__ import annotations

from pathlib import Path

from src.training.quantization.runtime import specialize_qat_onnx_batch
from tools.build_tensorrt_refit_template import RefitMode, build_template
from tools.publish_tensorrt_engine import export_onnx, refit_engine, verify_engine


def build_and_verify_tensorrt_engine(
    source_path: Path,
    engine_path: Path,
    batch_size: int,
    channels: int,
    rows: int,
    columns: int,
    builder_optimization_level: int,
) -> None:
    onnx_path = engine_path.with_suffix('.onnx')
    template_path = engine_path.with_suffix('.template.engine')
    onnx_path.unlink(missing_ok=True)
    template_path.unlink(missing_ok=True)
    try:
        if source_path.suffix == '.onnx':
            specialize_qat_onnx_batch(source_path, onnx_path, batch_size)
        else:
            export_onnx(source_path, onnx_path, (batch_size, channels, rows, columns))
        build_template(
            onnx_path,
            template_path,
            batch_size,
            channels,
            rows,
            columns,
            builder_optimization_level,
            None,
            RefitMode.ALL,
        )
        refit_engine(template_path, onnx_path, engine_path)
        verify_engine(
            onnx_path,
            engine_path,
            (batch_size, channels, rows, columns),
            allow_fidelity_deviation=False,
        )
    finally:
        onnx_path.unlink(missing_ok=True)
        template_path.unlink(missing_ok=True)
