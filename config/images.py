from .dsl import group, base, provides


group(
    "images",
    [
        "mnist",
        "fashion-mnist",
        "cifar10",
        "svhn"
    ]
)


@base
def config(dataset, use_baseline):
    return {
        "num_u_channels": 1,
        "use_cond_affine": True,
        "pure_cond_affine": False,

        "dequantize": True,

        "batch_norm": False,
        "batch_norm_apply_affine": use_baseline,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,

        "lr_schedule": "none",
        "max_bad_valid_epochs": 20,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 5,
        "early_stopping": True,

        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 1,

        "use_fid": True,
        "num_fid_samples": 10000,
        "fid_dims": 2048
    }


@provides("realnvp")
def realnvp(dataset, model, use_baseline):
    config = {
        "schema_type": "multiscale-realnvp",

        "g_hidden_channels": [64]*8 if use_baseline else [64]*4,

        "st_nets": [8] * 2,
        "p_nets": [64] * 2,
        "q_nets": [64] * 2,

        "train_batch_size": 100,
        "valid_batch_size": 100,
        "test_batch_size": 100,
        "opt": "adam",
        "lr": 1e-4,
        "weight_decay": 0.
    }

    if dataset in ["cifar10", "svhn"]:
        config["logit_tf_lambda"] = 0.05
        config["logit_tf_scale"] = 256

    elif dataset in ["mnist", "fashion-mnist"]:
        config["logit_tf_lambda"] = 1e-6
        config["logit_tf_scale"] = 256

    return config


@provides("glow")
def glow(dataset, model, use_baseline):
    if use_baseline:
        config = {
            "num_scales": 3,
            "num_steps_per_scale": 32,
            "g_num_hidden_channels": 512,
            "valid_batch_size": 500,
            "test_batch_size": 500
        }

    else:
        config = {
            "num_scales": 2,
            "num_steps_per_scale": 32,
            "g_num_hidden_channels": 256,
            "st_nets": 64,
            "p_nets": 128,
            "q_nets": 128,
            "valid_batch_size": 100,
            "test_batch_size": 100
        }

    config["schema_type"] = "glow"

    config["early_stopping"] = False
    config["train_batch_size"] = 64
    config["opt"] = "adamax"
    config["lr"] = 5e-4

    if dataset in ["cifar10"]:
        config["weight_decay"] = 0.1
    else:
        config["weight_decay"] = 0.

    config["centering_tf_scale"] = 256

    return config


@provides("non-square")
def non_square_flow(dataset, model, use_baseline):
    return {
        "non_square": True,
        "m_flow": use_baseline,
        "num_u_channels": 0,

        "batch_norm": False,
        "resnet_batchnorm": False,
        "ignore_batch_effects": False,

        "train_batch_size": 50,
        "valid_batch_size": 50,
        "test_batch_size": 50,

        "schema_type": "multiscale-realnvp",
        "underlying_flow": "realnvp",
        "g_hidden_channels": [64]*8,

        "smaller_realnvp": False,

        "num_density_layers": 10,

        "max_epochs": 1000,
        "epochs_per_test": 1,

        "regularization_param": 50,

        "log_jacobian_method": "hutch_with_cg",
        "hutchinson_distribution": "normal",
        "hutchinson_samples": 1,

        "latent_dimension": 20,

        "likelihood_warmup": True,
        "likelihood_warmup_start": 25,
        "likelihood_warmup_end": 50,

        "max_bad_valid_epochs": 20,

        "cg_tolerance": 1,

        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 1,

        "prior": "realnvp",
        "prior_num_density_layers": 10,
        "prior_hidden_channels": [32]*4,
        "prior_batch_norm": False
    }
