from __future__ import annotations

import json
from pathlib import Path

import tensorrt as trt

ONNX_PATH = Path(
    '/workspace/int8-salvage/calibration-sweep/max-4/artifacts/'
    'chess-cnn-14x160-fromto-batch320-int8-qdq.onnx'
)
OUTPUT = Path('/workspace/int8-salvage/full-trunk-engine-inspector-detailed.json')


def main() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(ONNX_PATH)):
        raise ValueError('\n'.join(str(parser.get_error(index)) for index in range(parser.num_errors)))
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    configuration.builder_optimization_level = 3
    configuration.set_flag(trt.BuilderFlag.FP16)
    configuration.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError('TensorRT detailed engine build failed.')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise ValueError('TensorRT detailed engine deserialization failed.')
    inspector = engine.create_engine_inspector()
    report = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    parsed = json.loads(report)
    OUTPUT.write_text(json.dumps(parsed, indent=2) + '\n')
    print(json.dumps(parsed, indent=2))


if __name__ == '__main__':
    main()
