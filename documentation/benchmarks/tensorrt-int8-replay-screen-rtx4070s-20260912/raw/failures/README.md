# Failed or superseded variants

- `post_qat`: the first module-level post-activation QAT formulation quantized convolution outputs before batch normalization. Active-QDQ folded-training smokes became non-finite; earlier zero-QDQ outputs were false positives and were discarded.
- `pre_scaled_qat`: both 100,000-step seeds learned the replay target and agreed with framework fake quantization, but TensorRT execution was catastrophically wrong. A strongly typed engine did not repair it.
- `post_fixed_amax`: forcing all residual boundaries to `amax=6` produced only about 1.38x TorchScript throughput and 83–86% framework/deployment top-1 agreement.
- `post_shared_qat`: distinct Q/DQ boundaries with a numerically shared trunk-output scale were runtime-faithful, but reached only 93,682 positions/s at 14x160, slower than TensorRT FP16.
- `post_scaled_fused`: removing the residual multiply and explicit final clip recovered residual fusions, but the fused convolutions still emitted FP16 and throughput did not materially improve.

These labels are retained here to prevent failed smoke artifacts on the ephemeral node from being mistaken for completed screen arms.
