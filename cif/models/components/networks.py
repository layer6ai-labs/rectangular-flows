import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jvp_layers import get_jvp_from_module, get_jvp_from_activation, unpack_jvp, pack_jvp
from .custom_batchnorm import DetachedBatchNorm2d


class ConstantNetwork(nn.Module):
    def __init__(self, value, fixed):
        super().__init__()
        if fixed:
            self.register_buffer("value", value)
        else:
            self.value = nn.Parameter(value)

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0], *self.value.shape)


class NN_Sequential_JVP(nn.Sequential):
    """Standard NN Sequential but with jvp method"""
    def __init__(self, *args):
        super().__init__(*args)

    def jvp(self, inputs, v):
        for module in self:
            inputs, v = unpack_jvp(get_jvp_from_module(module, inputs, v))
        return pack_jvp(inputs, v)


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, use_batchnorm=True, ignore_batch_effects=False):
        super().__init__()

        self.use_batchnorm = use_batchnorm

        if self.use_batchnorm:
            bn_module = DetachedBatchNorm2d if ignore_batch_effects else nn.BatchNorm2d
            self.bn1 = bn_module(num_channels)
            self.bn2 = bn_module(num_channels)

        self.conv1 = self._get_conv3x3(num_channels)
        self.conv2 = self._get_conv3x3(num_channels)

    def forward(self, inputs):
        out = self.bn1(inputs) if self.use_batchnorm else inputs
        out = torch.relu(out)
        out = self.conv1(out)

        out = self.bn2(out) if self.use_batchnorm else out
        out = torch.relu(out)
        out = self.conv2(out)

        out = out + inputs

        return out

    def jvp(self, inputs, v):
        out = inputs
        v_out = v

        if self.use_batchnorm:
            out, v_out = unpack_jvp(get_jvp_from_module(self.bn1, out, v_out))
        out, v_out = unpack_jvp(get_jvp_from_activation("relu", out, v_out))
        out, v_out = unpack_jvp(get_jvp_from_module(self.conv1, out, v_out))

        if self.use_batchnorm:
            out, v_out = unpack_jvp(get_jvp_from_module(self.bn2, out, v_out))
        out, v_out = unpack_jvp(get_jvp_from_activation("relu", out, v_out))
        out, v_out = unpack_jvp(get_jvp_from_module(self.conv2, out, v_out))

        out = out + inputs
        v_out = v_out + v

        return pack_jvp(out, v_out)

    def _get_conv3x3(self, num_channels):
        return nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,

            # NOTE: We don't add a bias here since any subsequent ResidualBlock
            # will begin with a batch norm (if using batchnorm). However, we add
            # a bias at the output of the whole network.
            bias=(not self.use_batchnorm)
        )


class ScaledTanh2dModule(nn.Module):
    def __init__(self, module, num_channels):
        super().__init__()
        self.module = module
        self.weights = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, inputs):
        out = self.module(inputs)
        out = self.weights * torch.tanh(out) + self.bias
        return out

    def jvp(self, inputs, v):
        out, v = unpack_jvp(self.module.jvp(inputs, v))
        out, v = unpack_jvp(get_jvp_from_activation("tanh", out, v))
        out = self.weights * out + self.bias
        v = self.weights * v
        return pack_jvp(out, v)


def get_resnet(
        num_input_channels,
        hidden_channels,
        num_output_channels,
        use_batchnorm=True,
        ignore_batch_effects=False
):
    num_hidden_channels = hidden_channels[0] if hidden_channels else num_output_channels

    # TODO: Should we have an input batch norm?
    layers = [
        nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
    ]

    for num_hidden_channels in hidden_channels:
        layers.append(ResidualBlock(num_hidden_channels, use_batchnorm, ignore_batch_effects))

    if use_batchnorm:
        if ignore_batch_effects:
            layers.append(DetachedBatchNorm2d(num_hidden_channels))
        else:
            layers.append(nn.BatchNorm2d(num_hidden_channels))

    layers += [
        nn.ReLU(),
        nn.Conv2d(
            in_channels=num_hidden_channels,
            out_channels=num_output_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
    ]

    # TODO: Should we have an output batch norm?
    return ScaledTanh2dModule(
        module=NN_Sequential_JVP(*layers),
        num_channels=num_output_channels
    )


def get_glow_cnn(
        num_input_channels,
        num_hidden_channels,
        num_output_channels,
        zero_init_output
):
    conv1 = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=num_hidden_channels,
        kernel_size=3,
        padding=1,
        bias=False
    )

    bn1 = nn.BatchNorm2d(num_hidden_channels)

    conv2 = nn.Conv2d(
        in_channels=num_hidden_channels,
        out_channels=num_hidden_channels,
        kernel_size=1,
        padding=0,
        bias=False
    )

    bn2 = nn.BatchNorm2d(num_hidden_channels)

    conv3 = nn.Conv2d(
        in_channels=num_hidden_channels,
        out_channels=num_output_channels,
        kernel_size=3,
        padding=1
    )

    if zero_init_output:
        conv3.weight.data.zero_()
        conv3.bias.data.zero_()

    relu = nn.ReLU()

    return nn.Sequential(conv1, bn1, relu, conv2, bn2, relu, conv3)


def get_mlp(
        num_input_channels,
        hidden_channels,
        num_output_channels,
        activation,
        log_softmax_outputs=False
):
    layers = []
    prev_num_hidden_channels = num_input_channels
    for num_hidden_channels in hidden_channels:
        layers.append(nn.Linear(prev_num_hidden_channels, num_hidden_channels))
        layers.append(activation())
        prev_num_hidden_channels = num_hidden_channels
    layers.append(nn.Linear(prev_num_hidden_channels, num_output_channels))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return NN_Sequential_JVP(*layers)


class MaskedLinear(nn.Module):
    def __init__(self, input_degrees, output_degrees):
        super().__init__()

        assert len(input_degrees.shape) == len(output_degrees.shape) == 1

        num_input_channels = input_degrees.shape[0]
        num_output_channels = output_degrees.shape[0]

        self.linear = nn.Linear(num_input_channels, num_output_channels)

        mask = output_degrees.view(-1, 1) >= input_degrees
        self.register_buffer("mask", mask.to(self.linear.weight.dtype))

    def forward(self, inputs):
        return F.linear(inputs, self.mask*self.linear.weight, self.linear.bias)


class AutoregressiveMLP(nn.Module):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_output_heads,
            activation
    ):
        super().__init__()
        self.flat_ar_mlp = self._get_flat_ar_mlp(num_input_channels, hidden_channels, num_output_heads, activation)
        self.num_input_channels = num_input_channels
        self.num_output_heads = num_output_heads

    def _get_flat_ar_mlp(
            self,
            num_input_channels,
            hidden_channels,
            num_output_heads,
            activation
    ):
        assert num_input_channels >= 2
        assert all([num_input_channels <= d for d in hidden_channels]), "Random initialisation not yet implemented"

        prev_degrees = torch.arange(1, num_input_channels + 1, dtype=torch.int64)
        layers = []

        for hidden_channels in hidden_channels:
            degrees = torch.arange(hidden_channels, dtype=torch.int64) % (num_input_channels - 1) + 1

            layers.append(MaskedLinear(prev_degrees, degrees))
            layers.append(activation())

            prev_degrees = degrees

        degrees = torch.arange(num_input_channels, dtype=torch.int64).repeat(num_output_heads)
        layers.append(MaskedLinear(prev_degrees, degrees))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        assert inputs.shape[1:] == (self.num_input_channels,)
        result = self.flat_ar_mlp(inputs)
        result = result.view(inputs.shape[0], self.num_output_heads, self.num_input_channels)
        return result
