import torch
import torch.nn as nn

from .bijection import Bijection


class AffineCouplingBijection(Bijection):
    def __init__(self, x_shape, coupler):
        super().__init__(x_shape=x_shape, z_shape=x_shape)
        self.coupler = coupler

    def _couple(self, inputs, **kwargs):
        if "u" in kwargs:
            inputs = torch.cat((inputs, kwargs["u"]), dim=1)
        outputs = self.coupler(inputs)
        return outputs["shift"], outputs["log-scale"]

    def _coupler_jvp(self, inputs, v):
        coupler_jvp_result = self.coupler.jvp(inputs, v)
        return coupler_jvp_result["shift"], coupler_jvp_result["log-scale"]

    def _log_jac_x_to_z(self, log_scale):
        return log_scale.flatten(start_dim=1).sum(dim=1, keepdim=True)

    def _log_jac_z_to_x(self, log_scale):
        return -self._log_jac_x_to_z(log_scale)


class Checkerboard2dAffineCouplingBijection(AffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler,
            reverse_mask
    ):
        super().__init__(x_shape=x_shape, coupler=coupler)

        assert len(x_shape) == 3
        num_channels, height, width = x_shape

        self.register_buffer("mask", self._checkerboard_mask(num_channels, height, width, reverse_mask))

    def _x_to_z(self, x, **kwargs):
        shift, log_scale = self._couple(self.mask*x, **kwargs)
        z = self.mask*x + (1-self.mask)*((x + shift) * torch.exp(log_scale))
        return {"z": z, "log-jac": self._log_jac_x_to_z((1-self.mask)*log_scale)}

    def _z_to_x(self, z, **kwargs):
        shift, log_scale = self._couple(self.mask*z, **kwargs)
        x = self.mask*z + (1-self.mask)*(z * torch.exp(-log_scale) - shift)
        return {"x": x, "log-jac": self._log_jac_z_to_x((1-self.mask)*log_scale)}

    def _jvp(self, z, v, **kwargs):
        shift_net_jvp_result, log_scale_net_jvp_result = self._coupler_jvp(self.mask*z, self.mask*v)

        shift, shift_jvp = shift_net_jvp_result["x"], shift_net_jvp_result["jvp"]
        log_scale, log_scale_jvp = log_scale_net_jvp_result["x"], log_scale_net_jvp_result["jvp"]

        x = self.mask*z + (1-self.mask)*(z * torch.exp(-log_scale) - shift)

        v_2 = (1-self.mask)*v
        z_2 = (1-self.mask)*z
        jvp_term_2 = torch.exp(-log_scale)*(v_2 - z_2 * log_scale_jvp) - shift_jvp
        jvp = self.mask*v + (1-self.mask)*jvp_term_2

        return {"x": x, "jvp": jvp}

    def _checkerboard_mask(self, num_channels, height, width, reverse_mask):
        mask = torch.empty((height, width))
        for i in range(height):
            for j in range(width):
                mask[i, j] = (i + j) % 2 == 1
        mask = mask.expand(num_channels, -1, -1)

        if reverse_mask:
            mask = 1 - mask

        return mask


class ChannelwiseAffineCouplingBijection(AffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler_factory,
            num_passthrough_channels,
            reverse_mask
    ):
        if reverse_mask:
            num_passthrough_channels = x_shape[0] - num_passthrough_channels

        assert num_passthrough_channels > 0, "Not a bijection without passthrough"

        super().__init__(
            x_shape=x_shape,
            coupler=coupler_factory(num_passthrough_channels)
        )

        self.reverse_mask = reverse_mask

    def _x_to_z(self, x, **kwargs):
        passthrough_x, modified_x = self._split(x)

        shift, log_scale = self._couple(passthrough_x, **kwargs)

        z = self._combine(
            passthrough_x,
            (modified_x + shift) * torch.exp(log_scale)
        )

        return {"z": z, "log-jac": self._log_jac_x_to_z(log_scale)}

    def _z_to_x(self, z, **kwargs):
        passthrough_z, modified_z = self._split(z)

        shift, log_scale = self._couple(passthrough_z, **kwargs)

        x = self._combine(
            passthrough_z,
            modified_z * torch.exp(-log_scale) - shift
        )

        return {"x": x, "log-jac": self._log_jac_z_to_x(log_scale)}

    def _jvp(self, z, v, **kwargs):
        passthrough_z, modified_z = self._split(z)
        passthrough_v, modified_v = self._split(v)

        shift_net_jvp_result, log_scale_net_jvp_result = self._coupler_jvp(
            passthrough_z, passthrough_v
        )
        shift, shift_jvp = shift_net_jvp_result["x"], shift_net_jvp_result["jvp"]
        log_scale, log_scale_jvp = log_scale_net_jvp_result["x"], log_scale_net_jvp_result["jvp"]

        x = self._combine(
            passthrough_z,
            modified_z * torch.exp(-log_scale) - shift
        )
        jvp = self._combine(
            passthrough_v,
            torch.exp(-log_scale)*(
                modified_v - modified_z*log_scale_jvp
            ) - shift_jvp
        )

        return {"x": x, "jvp": jvp}

    def _split(self, inputs):
        passthrough, modified = self._do_split(inputs)

        if self.reverse_mask:
            passthrough, modified = modified, passthrough

        return passthrough, modified

    def _combine(self, outputs1, outputs2):
        if self.reverse_mask:
            outputs1, outputs2 = outputs2, outputs1

        return self._do_combine(outputs1, outputs2)

    def _do_split(self, inputs):
        raise NotImplementedError

    def _do_combine(self, outputs1, outputs2):
        raise NotImplementedError


class SplitChannelwiseAffineCouplingBijection(ChannelwiseAffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler_factory,
            reverse_mask
    ):
        self.num_passthrough_channels = x_shape[0] // 2

        super().__init__(
            x_shape=x_shape,
            coupler_factory=coupler_factory,
            num_passthrough_channels=self.num_passthrough_channels,
            reverse_mask=reverse_mask
        )

    def _do_split(self, inputs):
        return inputs[:, :self.num_passthrough_channels], inputs[:, self.num_passthrough_channels:]

    def _do_combine(self, inputs1, inputs2):
        return torch.cat((inputs1, inputs2), dim=1)


class AlternatingChannelwiseAffineCouplingBijection(ChannelwiseAffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler_factory,
            reverse_mask
    ):
        super().__init__(
            x_shape=x_shape,
            coupler_factory=coupler_factory,
            num_passthrough_channels=(x_shape[0] + 1) // 2,
            reverse_mask=reverse_mask
        )

    def _do_split(self, inputs):
        return inputs[:, ::2], inputs[:, 1::2]

    def _do_combine(self, inputs1, inputs2):
        # TODO: It might be possible to speed this up by using stack
        result = inputs1.new_empty(inputs1.shape[0], *self.x_shape)
        result[:, ::2] = inputs1
        result[:, 1::2] = inputs2
        return result


# TODO: Unit test
class MaskedChannelwiseAffineCouplingBijection(ChannelwiseAffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler_factory,
            mask
    ):
        assert torch.any(mask), "Not a bijection without passthrough"

        super().__init__(
            x_shape=x_shape,
            coupler_factory=coupler_factory,
            num_passthrough_channels=mask.sum().item(),
            reverse_mask=False
        )

        self.mask = mask

    def _do_split(self, inputs):
        return inputs[:, self.mask], inputs[:, ~self.mask]

    def _do_combine(self, inputs1, inputs2):
        result = inputs1.new_empty(inputs1.shape[0], *self.x_shape)
        result[:, self.mask] = inputs1
        result[:, ~self.mask] = inputs2
        return result
