# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Testing functionality of the default BOO backend.
"""
import pytest
import torch
from torch import fx
from torch._dynamo.testing import EagerAndRecordGraphs

from iree.turbine.dynamo.backends import boo


def test_custom_boo_conv_used():
    """Test that we're using our custom convolution op"""
    recorder = EagerAndRecordGraphs()
    compiled_conv = torch.compile(
        torch.ops.aten.convolution,
        backend=boo.backend(nested_backend=recorder),
    )

    N, C, H, W = (1, 16, 4, 4)
    F = 32
    K = 3
    input = torch.randn((N, C, H, W))
    weight = torch.randn((F, C, K, K))
    compiled_conv(
        input,
        weight,
        None,  # bias
        [1, 1],  # stride
        [1, 1],  # padding
        [1, 1],  # dilation
        False,  # transposed
        [10, 10],  # output_padding
        1,  # groups
    )

    [compiled_module] = recorder.graphs
    assert isinstance(compiled_module, fx.GraphModule)
    [call_node] = [n for n in compiled_module.graph.nodes if n.op == "call_function"]
    # Make sure we're using 'boo.ops.convolution_replacement'. We have to do a
    # string check unfortunately, as the target is a fused custom op that we
    # can't inspect.
    call_node_target_str = str(call_node.target)
    assert call_node_target_str.startswith("boo.fused_op_convolution_replacement_")


def test_filter_transpose_conv():
    """Test that we don't offload transpose conv to IREE/BOO."""
    recorder = EagerAndRecordGraphs()
    compiled_conv = torch.compile(
        torch.ops.aten.convolution,
        backend=boo.backend(nested_backend=recorder),
    )

    N, C, H, W = (1, 16, 4, 4)
    F = 32
    K = 3
    output_shape = (N, F, H - K + 1, W - K + 1)
    grad_output = torch.randn(output_shape)
    weight = torch.randn((F, C, K, K))
    compiled_conv(
        grad_output,
        weight,
        None,  # bias
        [1, 1],  # stride
        [1, 1],  # padding
        [1, 1],  # dilation
        True,  # transposed
        [0, 0],  # output_padding
        1,  # groups
    )

    [compiled_module] = recorder.graphs
    assert isinstance(compiled_module, fx.GraphModule)
    [call_node] = [n for n in compiled_module.graph.nodes if n.op == "call_function"]
    # Make sure we didn't replace the aten convolution.
    assert call_node.target == torch.ops.aten.convolution.default


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires GPU"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last]
)
def test_output_layout(device: str, memory_format: torch.memory_format):
    """Test that we properly match the layout of pytorch's implementation."""
    with torch.device(device):
        conv = torch.ops.aten.convolution
        boo_conv = torch.compile(conv, backend="iree_boo")
        N, C, H, W = (1, 16, 4, 4)
        F = 32
        K = 3
        input = torch.randn((N, C, H, W)).to(memory_format=memory_format)
        weight = torch.randn((F, C, K, K)).to(memory_format=memory_format)
        args = (
            input,
            weight,
            None,  # bias
            [1, 1],  # stride
            [1, 1],  # padding
            [1, 1],  # dilation
            False,  # transposed
            [10, 10],  # output_padding
            1,  # groups
        )

        actual_result = boo_conv(*args)
        expected_result = conv(*args)
        assert isinstance(actual_result, torch.Tensor)
        assert isinstance(expected_result, torch.Tensor)
        assert actual_result.shape == expected_result.shape
        assert actual_result.stride() == expected_result.stride()


def test_batch_norm_layer():
    """Test that we're using our custom convolution op"""
    # memory_format = torch.channels_last
    memory_format = torch.channels_last
    with torch.device("cuda"):
        N, C, H, W = (128, 384, 24, 48)
        input = torch.randn((N, C, H, W), dtype=torch.bfloat16).to(
            memory_format=memory_format
        )
        model = torch.nn.BatchNorm2d(num_features=C).to(memory_format=memory_format)
        recorder = EagerAndRecordGraphs()
        compiled_model = torch.compile(
            model,
            backend=boo.backend(nested_backend=recorder),
        )

        compiled_model(input)

        [compiled_module] = recorder.graphs
        assert isinstance(compiled_module, fx.GraphModule)


@pytest.mark.parametrize("backend", ["iree_boo", "inductor", "eager"])
def test_batch_norm_bench(backend: str):
    with torch.device("cuda"):
        N, C, H, W = 128, 384, 24, 48
        memory_format = torch.channels_last
        weight = torch.randn((C,), dtype=torch.float32)
        bias = torch.randn((C,), dtype=torch.float32)
        running_mean = torch.randn((C,), dtype=torch.float32)
        running_var = torch.randn((C,), dtype=torch.float32)
        training = True
        exponential_average_factor = 0.1
        epsilon = 1e-5

        def model(input: torch.Tensor):
            return torch.ops.aten._native_batch_norm_legit_functional(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                exponential_average_factor,
                epsilon,
            )

        compiled_model = torch.compile(model, backend=backend)
        input = torch.randn((N, C, H, W), dtype=torch.bfloat16).to(
            memory_format=memory_format
        )
        steps = 100
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=steps),
        ) as prof:
            result = None
            for _ in range(11 + steps):
                result = compiled_model(input)
                torch.cuda.synchronize()
                prof.step()
        print(prof.key_averages().table(sort_by="self_device_time_total"))
        assert isinstance(result, tuple)
        print(f"result={tuple(type(r) for r in result)}")


EXAMPLE_SHAPES = [
    (128, 35, 48, 32),
    (128, 384, 24, 48),
    (128, 384, 48, 32),
    (128, 512, 24, 48),
    (128, 64, 48, 32),
    (128, 96, 48, 32),
    (16, 128, 24, 16),
    (16, 128, 48, 32),
    (16, 144, 24, 16),
    (16, 192, 24, 16),
    (16, 2048, 16, 32),
    (16, 2048, 48, 32),
    (16, 2, 24, 16),
    (16, 288, 1, 48, 32),
    (16, 288, 24, 16),
    (16, 288, 2, 48, 32),
    (16, 288, 4, 48, 32),
    (16, 288, 48, 32),
    (16, 288, 8, 32),
    (16, 288, 8, 48, 32),
    (16, 32, 192, 128),
    (16, 384, 48, 32),
    (16, 384, 8, 32),
    (16, 40, 192, 128),
    (16, 40, 96, 64),
    (16, 48, 24, 16),
    (16, 48, 48, 32),
    (16, 48, 96, 64),
    (16, 576, 48, 32),
    (16, 64, 225, 225),
    (16, 64, 38, 19),
    (16, 64, 38, 38),
    (16, 64, 48, 32),
    (16, 64, 75, 75),
    (16, 96, 24, 16),
    (16, 96, 48, 32),
    (16, 96, 96, 64),
]
