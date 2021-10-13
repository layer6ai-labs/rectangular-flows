import torch

from .density import Density


class SplitDensity(Density):
    def __init__(self, density_1, density_2, dim, non_square=False):
        super().__init__()

        self.density_1 = density_1
        self.density_2 = density_2
        self.dim = dim
        self.non_square = non_square

    def _elbo(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=self.dim)
        prior_dict_1 = self.density_1.elbo(x1)
        prior_dict_2 = self.density_2.elbo(x2)

        return {
            "elbo": prior_dict_1["elbo"] + prior_dict_2["elbo"],
            "prior-dict": prior_dict_1, # HACK: Assumes prior_dict_1 is more important
            "prior-dict-2": prior_dict_2
        }

    def _jvp(self, x, v):
        return {
            "x": self.pad_inputs(x)["x"],
            "jvp": self.pad_inputs(v)["x"]
        }

    def _fixed_sample(self, noise):
        x1 = self.density_1.fixed_sample(noise=noise)

        if self.non_square:
            return self.pad_inputs(x1)["x"]
        else:
            x2 = self.density_2.fixed_sample(noise=noise)
            return torch.cat((x1, x2), dim=self.dim)

    def _sample(self, num_samples):
        x1 = self.density_1.sample(num_samples)

        if self.non_square:
            return self.pad_inputs(x1)["x"]
        else:
            x2 = self.density_2.sample(num_samples)
            return torch.cat((x1, x2), dim=self.dim)

    def pad_inputs(self, x1):
        x2 = torch.zeros_like(x1)
        return {"x": torch.cat((x1, x2), dim=self.dim)}
