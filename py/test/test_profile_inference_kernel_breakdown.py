from __future__ import annotations

import pytest
from tools.profile_inference_kernel_breakdown import KernelClass, classify_kernel, interval_union_microseconds


@pytest.mark.parametrize(
    ('name', 'operations', 'expected'),
    (
        ('cudnn::conv', ('aten::conv2d',), KernelClass.CONVOLUTION),
        ('elementwise_kernel', ('aten::relu',), KernelClass.ACTIVATION),
        ('fused_add_relu', (), KernelClass.RESIDUAL_ADD),
        ('elementwise_kernel', ('aten::add',), KernelClass.RESIDUAL_ADD),
        ('reduce_kernel', ('aten::mean',), KernelClass.GLOBAL_POOLING),
        ('reduce_kernel<MeanOps>', (), KernelClass.GLOBAL_POOLING),
        ('cudnn::nchwToNhwcKernel', (), KernelClass.MEMORY_COPY),
        ('Memcpy DtoD', ('aten::copy_',), KernelClass.MEMORY_COPY),
        ('batch_norm_kernel', ('aten::batch_norm',), KernelClass.BATCH_NORM),
        ('softmax_kernel', ('aten::softmax',), KernelClass.OTHER),
    ),
)
def test_classify_kernel(
    name: str,
    operations: tuple[str, ...],
    expected: KernelClass,
) -> None:
    assert classify_kernel(name, operations) is expected


@pytest.mark.parametrize(
    ('intervals', 'expected'),
    (
        ((), 0.0),
        (((0.0, 2.0),), 2.0),
        (((0.0, 2.0), (1.0, 3.0)), 3.0),
        (((0.0, 1.0), (2.0, 4.0)), 3.0),
    ),
)
def test_interval_union_microseconds(
    intervals: tuple[tuple[float, float], ...],
    expected: float,
) -> None:
    assert interval_union_microseconds(intervals) == expected
