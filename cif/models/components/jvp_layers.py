import torch
import torch.nn as nn
import torch.nn.functional as F


def get_jvp_from_module(module, inputs, v):
    try:
        return module.jvp(inputs, v)
    except AttributeError:
        return get_jvp_from_default_module(module, inputs, v)


def unpack_jvp(jvp_result):
    return jvp_result["x"], jvp_result["jvp"]


def pack_jvp(x, jvp):
    return {"x": x, "jvp": jvp}


def get_jvp_from_default_module(module, inputs, v):
    if type(module) == nn.Linear:
        jvp_function = get_linear_jvp
    elif type(module) == nn.Conv2d:
        jvp_function = get_conv2d_jvp
    elif "BatchNorm2d" in type(module).__name__:
        jvp_function = get_batchnorm2d_jvp
    elif type(module) == nn.ReLU:
        return get_jvp_from_activation("relu", inputs, v)
    elif type(module) == nn.Tanh:
        return get_jvp_from_activation("tanh", inputs, v)
    else:
        raise ValueError(f"jvp unsupported for module {type(module).__name__}")

    return jvp_function(module, inputs, v)


def get_jvp_from_activation(activation_name, inputs, v):
    if activation_name == "relu":
        return pack_jvp(torch.relu(inputs), (inputs > 0)*v)
    elif activation_name == "tanh":
        outputs = torch.tanh(inputs)
        grad_tanh = 1 - outputs**2
        return pack_jvp(outputs, grad_tanh*v)
    else:
        raise ValueError(f"Unsupported activation function {activation_name} for jvp.")


def get_linear_jvp(module, inputs, v):
    outputs = module(inputs)
    jvp = F.linear(v, module.weight)
    return pack_jvp(outputs, jvp)


def get_conv2d_jvp(module, inputs, v):
    outputs = module(inputs)
    jvp = F.conv2d(
        input=v,
        weight=module.weight,
        bias=None,
        stride=module.stride,
        padding=module.padding
    )
    return pack_jvp(outputs, jvp)


def get_batchnorm2d_jvp(module, inputs, v):
    outputs = module(inputs)

    if module.training:
        mean = torch.mean(inputs, dim=(0,2,3), keepdim=True)
        var = torch.var(inputs, dim=(0,2,3), unbiased=False, keepdim=True)
    else:
        mean = module.running_mean.view(-1,1,1)
        var = module.running_var.view(-1,1,1)

    gamma = module.weight.view(-1,1,1)
    var_plus_eps = var + module.eps

    if module.training:
        term_1 = var_plus_eps*(v - torch.mean(v, dim=(0,2,3), keepdim=True))
        term_2 = (mean - inputs)*torch.mean((inputs - mean)*v, dim=(0,2,3), keepdim=True)

        jvp = gamma * (term_1 + term_2) / var_plus_eps**1.5
    else:
        jvp = gamma * v / torch.sqrt(var_plus_eps)

    return pack_jvp(outputs, jvp)
