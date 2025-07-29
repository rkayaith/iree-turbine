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
        memory_format = torch.channels_last
        weight = torch.randn((384,), dtype=torch.float)
        bias = torch.randn((384,), dtype=torch.float)
        running_mean = torch.randn((384,), dtype=torch.float)
        running_var = torch.randn((384,), dtype=torch.float)
        training = True
        exponential_average_factor = 0.1
        epsilon = 1e-5

        def model(input: torch.Tensor):
            return torch.ops.aten.miopen_batch_norm(
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
        input = torch.randn((128, 384, 24, 48), dtype=torch.bfloat16).to(
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
