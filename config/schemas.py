def get_schema(config):
    schema = get_base_schema(config=config)

    if config.get("non_square", False):
        schema = apply_non_square_settings(schema=schema, config=config)

    if config["pure_cond_affine"]:
        assert config["use_cond_affine"]
        schema = remove_non_normalise_layers(schema=schema)

    if config["use_cond_affine"]:
        assert config["num_u_channels"] > 0
        schema = add_cond_affine_before_each_normalise(schema=schema, config=config)

    schema = apply_pq_coupler_config_settings(schema=schema, config=config)

    schema = get_preproc_schema(config=config) + schema

    if config["batch_norm"]:
        schema = replace_normalise_with_batch_norm(schema=schema, config=config)
    else:
        schema = remove_normalise_layers(schema=schema)

    if config.get("non_square", False):
        schema = remove_cond_affine_before_base(schema)

    return schema


def get_preproc_schema(config):
    if config["dequantize"]:
        schema = [{"type": "dequantization"}]
    else:
        schema = []

    if config.get("logit_tf_lambda") is not None and config.get("logit_tf_scale") is not None:
        assert config.get("rescale_tf_scale") is None
        schema += get_logit_tf_schema(
            lam=config["logit_tf_lambda"],
            scale=config["logit_tf_scale"]
        )

    elif config.get("centering_tf_scale") is not None:
        assert config.get("logit_tf_lambda") is None
        assert config.get("logit_tf_scale") is None
        schema += get_centering_tf_schema(
            scale=config["centering_tf_scale"]
        )

    return schema


def apply_non_square_settings(schema, config):
    head_layer = {
        "type": "non-square-head",
        "regularization_param": config["regularization_param"],
        "log_jacobian_method": config["log_jacobian_method"],
        "hutchinson_distribution": config.get("hutchinson_distribution", "normal"),
        "hutchinson_samples": config.get("hutchinson_samples", 1),
        "m_flow": config["m_flow"],
        "max_cg_iterations": config.get("max_cg_iterations", None),
        "cg_tolerance": config.get("cg_tolerance", 1),
        "latent_dimension": config["latent_dimension"]
    }

    tail_layers = []
    tail_layers.append({
        "type": "non-square-base",
        "latent_dimension": config["latent_dimension"],
        "m_flow": config["m_flow"]
    })

    if config["prior"] == "affine":
        tail_layers.append({
            "type": "affine",
            "per_channel": False
        })
    elif config["prior"] == "realnvp":
        tail_layers += get_flat_realnvp_schema(
            num_density_layers=config["prior_num_density_layers"],
            coupler_shared_nets=True,
            coupler_hidden_channels=config["prior_hidden_channels"],
            batch_norm=True
        )
    elif config["prior"] == "nsf":
        # HACK: Hard-coding the low-dim NSF config
        NUM_BINS = 8
        TAIL_BOUND = 3.
        DROPOUT = 0.

        # HACK: Assumes the same number of channels per layer
        tail_layers += get_nsf_schema(
            num_density_layers=config["prior_num_density_layers"],
            use_linear=True,
            autoregressive=True,
            num_hidden_channels=config["prior_hidden_channels"][0],
            num_hidden_layers=len(config["prior_hidden_channels"]),
            num_bins=NUM_BINS,
            tail_bound=TAIL_BOUND,
            dropout_probability=DROPOUT
        )

    return [head_layer] + schema + tail_layers


def remove_cond_affine_before_base(schema):
    new_schema = []
    for i, layer in enumerate(schema):
        if layer["type"] == "non-square-base":
            return new_schema + schema[i:]

        if layer["type"] != "cond-affine":
            new_schema.append(layer)


# TODO: Could just pass the whole config to each constructor
def get_base_schema(config):
    ty = config["schema_type"]

    if ty == "multiscale-realnvp":
        return get_multiscale_realnvp_schema(
            coupler_hidden_channels=config["g_hidden_channels"],
            non_square=config.get("non_square", False),
            resnet_batchnorm=config.get("resnet_batchnorm", True),
            ignore_batch_effects=config.get("ignore_batch_effects", False),
            smaller_schema=config.get("smaller_realnvp", False)
        )

    elif ty == "flat-realnvp":
        return get_flat_realnvp_schema(
            num_density_layers=config["num_density_layers"],
            coupler_shared_nets=config["coupler_shared_nets"],
            coupler_hidden_channels=config["coupler_hidden_channels"]
        )

    elif ty == "maf":
        return get_maf_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["ar_map_hidden_channels"]
        )

    elif ty == "sos":
        return get_sos_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["g_hidden_channels"],
            num_polynomials_per_layer=config["num_polynomials_per_layer"],
            polynomial_degree=config["polynomial_degree"],
        )

    elif ty == "nsf":
        return get_nsf_schema(
            num_density_layers=config["num_density_layers"],
            use_linear=config.get("use_linear", True),
            autoregressive=config["autoregressive"],
            num_hidden_channels=config["num_hidden_channels"],
            num_hidden_layers=config["num_hidden_layers"],
            num_bins=config["num_bins"],
            tail_bound=config["tail_bound"],
            dropout_probability=config["dropout_probability"]
        )

    elif ty == "bnaf":
        return get_bnaf_schema(
            num_density_layers=config["num_density_layers"],
            num_hidden_layers=config["num_hidden_layers"],
            activation=config["activation"],
            hidden_channels_factor=config["hidden_channels_factor"]
        )

    elif ty == "glow":
        return get_glow_schema(
            num_scales=config["num_scales"],
            num_steps_per_scale=config["num_steps_per_scale"],
            coupler_num_hidden_channels=config["g_num_hidden_channels"],
            lu_decomposition=True,
            non_square=config.get("non_square", False)
        )

    elif ty == "planar":
        return get_planar_schema(config=config)

    elif ty == "cond-affine":
        return get_cond_affine_schema(config=config)

    elif ty == "affine":
        return get_affine_schema(config=config)

    else:
        assert False, f"Invalid schema type `{ty}'"


def remove_non_normalise_layers(schema):
    return [layer for layer in schema if layer["type"] == "normalise"]


def remove_normalise_layers(schema):
    return [layer for layer in schema if layer["type"] != "normalise"]


def replace_normalise_with_batch_norm(schema, config):
    if config["batch_norm_use_running_averages"]:
        new_schema = []
        momentum = config["batch_norm_momentum"]

    else:
        new_schema = [
            {
                "type": "passthrough-before-eval",
                # XXX: This should be sufficient for most of the non-image
                # datasets we have but can be made a config value if necessary
                "num_passthrough_data_points": 100_000
            }
        ]
        momentum = 1.

    apply_affine = config["batch_norm_apply_affine"]

    for layer in schema:
        if layer["type"] == "normalise":
            new_schema.append({
                "type": "batch-norm",
                "per_channel": True, # Hard coded for now; seems always to do better
                "momentum": momentum,
                "apply_affine": config["batch_norm_apply_affine"],
                "detach": config.get("ignore_batch_effects", False)
            })

        else:
            new_schema.append(layer)

    return new_schema


def add_cond_affine_before_each_normalise(schema, config):
    new_schema = []
    flattened = False
    for layer in schema:
        if layer["type"] == "flatten":
            flattened = True
        elif layer["type"] == "normalise":
            new_schema.append(get_cond_affine_layer(config, flattened))

        new_schema.append(layer)

    return new_schema


def apply_pq_coupler_config_settings(schema, config):
    new_schema = []
    flattened = False
    for layer in schema:
        if layer["type"] == "flatten":
            flattened = True

        if layer.get("num_u_channels", 0) > 0:
            layer = {
                **layer,
                "p_coupler": get_p_coupler_config(config, flattened),
                "q_coupler": get_q_coupler_config(config, flattened)
            }

        new_schema.append(layer)

    return new_schema


def get_logit_tf_schema(lam, scale):
    return [
        {"type": "scalar-mult", "value": (1 - 2*lam) / scale},
        {"type": "scalar-add", "value": lam},
        {"type": "logit"}
    ]


def get_centering_tf_schema(scale):
    return [
        {"type": "scalar-mult", "value": 1 / scale},
        {"type": "scalar-add", "value": -.5}
    ]


def get_cond_affine_layer(config, flattened):
    return {
        "type": "cond-affine",
        "num_u_channels": config["num_u_channels"],
        "st_coupler": get_st_coupler_config(config, flattened),
    }


def get_st_coupler_config(config, flattened):
    return get_coupler_config("t", "s", "st", config, flattened)


def get_p_coupler_config(config, flattened):
    return get_coupler_config("p_mu", "p_sigma", "p", config, flattened)


def get_q_coupler_config(config, flattened):
    return get_coupler_config("q_mu", "q_sigma", "q", config, flattened)


def get_coupler_config(
        shift_prefix,
        log_scale_prefix,
        shift_log_scale_prefix,
        config,
        flattened
):
    shift_key = f"{shift_prefix}_nets"
    log_scale_key = f"{log_scale_prefix}_nets"
    shift_log_scale_key = f"{shift_log_scale_prefix}_nets"

    if shift_key in config and log_scale_key in config:
        assert shift_log_scale_key not in config, "Over-specified coupler config"
        return {
            "independent_nets": True,
            "shift_net": get_coupler_net_config(config[shift_key], flattened),
            "log_scale_net": get_coupler_net_config(config[log_scale_key], flattened)
        }

    elif shift_log_scale_key in config:
        assert shift_key not in config and log_scale_key not in config, \
                "Over-specified coupler config"
        return {
            "independent_nets": False,
            "shift_log_scale_net": get_coupler_net_config(config[shift_log_scale_key], flattened)
        }

    else:
        assert False, f"Must specify either `{shift_log_scale_key}', or both `{shift_key}' and `{log_scale_key}'"


def get_coupler_net_config(net_spec, flattened):
    if net_spec in ["fixed-constant", "learned-constant"]:
        return {
            "type": "constant",
            "value": 0,
            "fixed": net_spec == "fixed-constant"
        }

    elif net_spec == "identity":
        return {
            "type": "identity"
        }

    elif isinstance(net_spec, list):
        if flattened:
            return {
                "type": "mlp",
                "activation": "tanh",
                "hidden_channels": net_spec
            }
        else:
            return {
                "type": "resnet",
                "hidden_channels": net_spec
            }

    elif isinstance(net_spec, int):
        if flattened:
            return {
                "type": "mlp",
                "activation": "tanh",
                # Multiply by 2 to match the 2 hidden layers of the glow-cnns
                "hidden_channels": [net_spec] * 2
            }
        else:
            return {
                "type": "glow-cnn",
                "num_hidden_channels": net_spec,
                "zero_init_output": True
            }

    else:
        assert False, f"Invalid net specifier {net_spec}"


def get_multiscale_realnvp_schema(
        coupler_hidden_channels,
        non_square,
        resnet_batchnorm,
        ignore_batch_effects,
        smaller_schema=False
    ):
    if smaller_schema:
        base_schema = [
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
            {"type": "squeeze", "factor": 2},
            {"type": "acl", "mask_type": "split-channel", "reverse_mask": False},
            {"type": "acl", "mask_type": "split-channel", "reverse_mask": True},
            {"type": "split", "non_square": non_square},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        ]
    else:
        base_schema = [
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
            {"type": "squeeze", "factor": 2},
            {"type": "acl", "mask_type": "split-channel", "reverse_mask": True},
            {"type": "acl", "mask_type": "split-channel", "reverse_mask": False},
            {"type": "acl", "mask_type": "split-channel", "reverse_mask": True},
            {"type": "split", "non_square": non_square},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
            {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True}
        ]

    schema = []
    for layer in base_schema:
        if layer["type"] == "acl":
            schema += [
                {
                    **layer,
                    "num_u_channels": 0,
                    "coupler": {
                        "independent_nets": False,
                        "shift_log_scale_net": {
                            "type": "resnet",
                            "hidden_channels": coupler_hidden_channels,
                            "batchnorm": resnet_batchnorm,
                            "ignore_batch_effects": ignore_batch_effects
                        }
                    }
                },
                {
                    "type": "normalise"
                }
            ]

        else:
            schema.append(layer)

    return schema


def get_glow_schema(
        num_scales,
        num_steps_per_scale,
        coupler_num_hidden_channels,
        lu_decomposition,
        non_square
):
    schema = []
    for i in range(num_scales):
        if i > 0:
            schema.append({"type": "split", "non_square": non_square})

        schema.append({"type": "squeeze", "factor": 2})

        for _ in range(num_steps_per_scale):
            schema += [
                {
                    "type": "normalise"
                },
                {
                    "type": "invconv",
                    "lu": lu_decomposition
                },
                {
                    "type": "acl",
                    "mask_type": "split-channel",
                    "reverse_mask": False,
                    "coupler": {
                        "independent_nets": False,
                        "shift_log_scale_net": {
                            "type": "glow-cnn",
                            "num_hidden_channels": coupler_num_hidden_channels,
                            "zero_init_output": True
                        }
                    },
                    "num_u_channels": 0
                }
            ]

    return schema


def get_flat_realnvp_schema(
        num_density_layers,
        coupler_shared_nets,
        coupler_hidden_channels,
        batch_norm=True
):
    result = [{"type": "flatten"}]

    if coupler_shared_nets:
        coupler_config = {
            "independent_nets": False,
            "shift_log_scale_net": {
                "type": "mlp",
                "hidden_channels": coupler_hidden_channels,
                "activation": "tanh"
            }
        }

    else:
        coupler_config = {
            "independent_nets": True,
            "shift_net": {
                "type": "mlp",
                "hidden_channels": coupler_hidden_channels,
                "activation": "relu"
            },
            "log_scale_net": {
                "type": "mlp",
                "hidden_channels": coupler_hidden_channels,
                "activation": "tanh"
            }
        }

    for i in range(num_density_layers):
        result.append(
            {
                "type": "acl",
                "mask_type": "alternating-channel",
                "reverse_mask": i % 2 != 0,
                "coupler": coupler_config,
                "num_u_channels": 0
            }
        )
        if batch_norm:
            result.append({"type": "normalise"})

    return result


def get_maf_schema(
        num_density_layers,
        hidden_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result += [
            {
                "type": "made",
                "hidden_channels": hidden_channels,
                "activation": "tanh"
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_sos_schema(
        num_density_layers,
        hidden_channels,
        num_polynomials_per_layer,
        polynomial_degree
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            # TODO: Try replacing with invconv
            result.append({"type": "flip"})

        result += [
            {
                "type": "sos",
                "hidden_channels": hidden_channels,
                "activation": "tanh",
                "num_polynomials": num_polynomials_per_layer,
                "polynomial_degree": polynomial_degree
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_nsf_schema(
        num_density_layers,
        use_linear,
        autoregressive,
        num_hidden_channels,
        num_hidden_layers,
        num_bins,
        tail_bound,
        dropout_probability
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result += [{"type": "rand-channel-perm"}]
        if use_linear: result += [{"type": "linear"}]

        layer = {
            "type": "nsf-ar" if autoregressive else "nsf-c",
            "num_hidden_channels": num_hidden_channels,
            "num_hidden_layers": num_hidden_layers,
            "num_bins": num_bins,
            "tail_bound": tail_bound,
            "activation": "relu",
            "dropout_probability": dropout_probability
        }

        if not autoregressive:
            layer["reverse_mask"] = i % 2 == 0

        result.append(layer)

        result.append(
            {
                "type": "normalise"
            }
        )

    result += [{"type": "rand-channel-perm"}]
    if use_linear: result += [{"type": "linear"}]

    return result


def get_bnaf_schema(
        num_density_layers,
        num_hidden_layers, # TODO: More descriptive name
        activation,
        hidden_channels_factor
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result += [
            {
                "type": "bnaf",
                "num_hidden_layers": num_hidden_layers,
                "hidden_channels_factor": hidden_channels_factor,
                "activation": activation,
                "residual": i < num_density_layers - 1
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_planar_schema(config):
    if config["num_u_channels"] == 0:
        layer = {"type": "planar"}

    else:
        layer = {
            "type": "cond-planar",
            "num_u_channels": config["num_u_channels"],
            "cond_hidden_channels": config["cond_hidden_channels"],
            "cond_activation": "tanh"
        }

    result = [
        layer,
        {"type": "normalise"}
    ] * config["num_density_layers"]

    return [{"type": "flatten"}] + result


def get_cond_affine_schema(config):
    return (
        [{"type": "flatten"}] +
        [{"type": "normalise"}] * config["num_density_layers"]
    )


# TODO: Try using just cond-affines with constant u
def get_affine_schema(config):
    return (
        [{"type": "flatten"}] +
        [{"type": "affine", "per_channel": False}] * config["num_density_layers"]
    )
