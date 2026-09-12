# TensorRT update-cadence diagnostic

This diagnostic used the two 1,000-step 14x160 scaled-postactivation QAT exports as same-shape,
different-weight checkpoints. It ran on one RTX 4070 SUPER with TensorRT 10.14.1.48.post1 from
branch revision `c3789fb7`. The command was:

```text
PYTHONPATH=. python tools/benchmark_tensorrt_update_cadence.py \
  --source-onnx /workspace/int8-scaled-post-14x160-1k/seed20260913/int8.onnx \
  --updated-onnx /workspace/int8-scaled-post-14x160-1k/seed20260914/int8.onnx \
  --output /workspace/int8-tensorrt-update-cadence-14x160 \
  --device-id 0
```

An editable timing cache reduced the updated-checkpoint build from 67.03 seconds to 6.39 seconds.
A full refittable engine accepted the updated ONNX initializers through `OnnxParserRefitter` in
0.156 seconds. Its 135,311 positions/second matched the fresh cached and uncached engines at 134,904
and 134,840 positions/second. All three engines had comparable errors against the updated ONNX
outputs. The numerical check used a deterministic synthetic binary batch; the replay screen performs
the separate chess-position fidelity measurement.

Refit applies only while the network topology, tensor shapes, inference batch, precision, and Q/DQ
placement stay fixed. Changes to those properties require a new engine build. The serialized timing
cache remains the measured fallback for that build.
