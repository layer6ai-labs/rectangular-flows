import sys
import shutil
from pathlib import Path

import numpy as np


def get_non_square_parameters(module, m_flow):
    if not m_flow:
        return [module.parameters()]
    else:
        # NOTE: Need to separate the reconstruction from the likelihood parameters
        #       to allow for training the parameters separately.

        # First need to locate the m_flow head module to access specialized
        # parameters() function
        m_flow_head_module = module
        while not type(m_flow_head_module).__name__ == "ManifoldFlowHeadDensity":
            if "module" in m_flow_head_module._modules:     # For nn.DataParallel
                m_flow_head_module = m_flow_head_module.module
            elif "density" in m_flow_head_module._modules:  # For WrapperDensity
                m_flow_head_module = m_flow_head_module.density
            elif "prior" in m_flow_head_module._modules:    # For other Density objects
                m_flow_head_module = m_flow_head_module.prior
            else:
                raise RuntimeError(f"Module {m_flow_head_module} has no prior")

        return m_flow_head_module.separate_parameters()


def get_non_square_train_metrics(config):
    # HACK: If we are using m_flow, we consider every two epochs to be one epoch
    num_objectives = 2 if config["m_flow"] else 1

    if config["likelihood_warmup"]:
        warmup_start_epoch = num_objectives * config["likelihood_warmup_start"]
        warmup_end_epoch = num_objectives * config["likelihood_warmup_end"]
        warmup_bounds = [warmup_start_epoch, warmup_end_epoch]

        likelihood_introduction_epoch = warmup_start_epoch
        early_stopping_start_epoch = warmup_end_epoch
    else:
        likelihood_introduction_epoch = 0
        early_stopping_start_epoch = 0

    def train_metrics(density, x, epoch):
        # HACK: Whenever num_objectives == 1, all of the modular arithmetic expressions
        #       below will evaluate to True. This is intended behaviour, covering both
        #       the m-flow and non-m-flow cases.
        if config["likelihood_warmup"]:
            if (epoch + 1) % num_objectives == 0:
                likelihood_weight = np.interp(epoch, warmup_bounds, [0, 1])
            else:
                likelihood_weight = 0
        else:
            likelihood_weight = float((epoch + 1) % num_objectives == 0)

        add_reconstruction = epoch % num_objectives == 0

        loss = -density.elbo(
            x,
            likelihood_wt=likelihood_weight,
            add_reconstruction=add_reconstruction
        )["elbo"].mean()

        return {"loss": loss}

    return train_metrics, likelihood_introduction_epoch, early_stopping_start_epoch
