import torch
import torch.nn as nn

from .density import Density


class WrapperDensity(Density):
    def __init__(self, density):
        super().__init__()
        self.density = density

    def _elbo(self, x, **kwargs):
        return self.density.elbo(x, **kwargs)

    def _sample(self, num_samples):
        return self.density.sample(num_samples)

    def _fixed_sample(self, noise):
        return self.density.fixed_sample(noise=noise)

    def _ood(self, x):
        return self.density.ood(x)

    def _extract_latent(self, x, **kwargs):
        return self.density.extract_latent(x, **kwargs)


class DequantizationDensity(WrapperDensity):
    def _elbo(self, x, **kwargs):
        return super()._elbo(x.add_(torch.rand_like(x)), **kwargs)


class PassthroughBeforeEvalDensity(WrapperDensity):
    def __init__(self, density, x):
        super().__init__(density)

        # XXX: It is inefficient to store the data separately, but this will work for # the (non-image) datasets we consider
        self.register_buffer("x", x)

    # We need to do it like this, i.e. we can't just override self.eval(), since
    # nn.Module.eval() just calls train(train_mode=False), so it wouldn't be called
    # recursively by modules containing this one.
    # TODO: Could do with hooks
    def train(self, train_mode=True):
        if not train_mode:
            self.training = True
            with torch.no_grad():
                self.elbo(self.x)
        super().train(train_mode)


class DataParallelDensity(nn.DataParallel):
    def elbo(self, x, **kwargs):
        return self("elbo", x, **kwargs)

    def ood(self, x):
        return self("ood", x)

    def extract_latent(self, x, **kwargs):
        return self("extract-latent", x, **kwargs)

    def sample(self, num_samples):
        # Bypass DataParallel
        return self.module.sample(num_samples)

    def fixed_sample(self, noise=None):
        # Bypass DataParallel
        return self.module.fixed_sample(noise=noise)
