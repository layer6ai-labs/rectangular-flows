import torch.nn as nn

from .jvp_layers import unpack_jvp, pack_jvp


class IndependentCoupler(nn.Module):
    def __init__(self, shift_net, log_scale_net):
        super().__init__()
        self.shift_net = shift_net
        self.log_scale_net = log_scale_net

    def forward(self, inputs):
        return {
            "shift": self.shift_net(inputs),
            "log-scale": self.log_scale_net(inputs)
        }

    def jvp(self, inputs, v):
        shift_net_jvp_result = self.shift_net.jvp(inputs, v)
        log_scale_net_jvp_result = self.log_scale_net.jvp(inputs, v)
        return {
            "shift": shift_net_jvp_result,
            "log-scale": log_scale_net_jvp_result
        }


class SharedCoupler(nn.Module):
    _CHANNEL_DIM = 1

    def __init__(self, shift_log_scale_net):
        super().__init__()
        self.shift_log_scale_net = shift_log_scale_net

    def forward(self, inputs):
        result = self.shift_log_scale_net(inputs)
        shift, log_scale = self._split(result)
        return {
            "shift": shift,
            "log-scale": log_scale
        }

    def jvp(self, inputs, v):
        result, result_jvp = unpack_jvp(self.shift_log_scale_net.jvp(inputs, v))
        shift, log_scale = self._split(result)
        shift_jvp, log_scale_jvp = self._split(result_jvp)
        return {
            "shift": pack_jvp(shift, shift_jvp),
            "log-scale": pack_jvp(log_scale, log_scale_jvp)
        }

    def _split(self, shared_outputs):
        raise NotImplementedError


class ChunkedSharedCoupler(SharedCoupler):
    def _split(self, shared_outputs):
        num_channels = shared_outputs.shape[self._CHANNEL_DIM]
        assert num_channels % 2 == 0
        return shared_outputs[:, :num_channels//2], shared_outputs[:, num_channels//2:]


class IndexedSharedCoupler(SharedCoupler):
    def _split(self, shared_outputs):
        assert len(shared_outputs.shape) > 2
        assert shared_outputs.shape[self._CHANNEL_DIM] == 2
        return shared_outputs[:, 0], shared_outputs[:, 1]
