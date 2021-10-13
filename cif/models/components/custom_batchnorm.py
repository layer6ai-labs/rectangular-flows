import torch
import torch.nn as nn

from .jvp_layers import pack_jvp


class DetachedBatchNorm2d(nn.BatchNorm2d):
    """
    This class ignores batching effects on the gradients when training.
    Also adds a jvp method which will be consistent with the vjp arising
    from ignoring batching.
    """
    def forward(self, x):
        if not self.training:
            return super().forward(x)

        mean = torch.mean(x, dim=(0,2,3), keepdim=True).detach()
        var = torch.var(x, dim=(0,2,3), keepdim=True, unbiased=False).detach()

        if self.track_running_stats:
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.view(-1).data)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.view(-1).data)

        x_scaled = (x - mean) / torch.sqrt(var + self.eps)
        return x_scaled * self.weight.view(-1,1,1) + self.bias.view(-1,1,1)

    def jvp(self, x, v):
        outputs = self(x)

        if self.training:
            var = torch.var(x, dim=(0,2,3), unbiased=False, keepdim=True)
        else:
            var = self.running_var.view(-1,1,1)

        jvp = self.weight.view(-1,1,1) * v / torch.sqrt(var + self.eps)

        return pack_jvp(outputs, jvp)
