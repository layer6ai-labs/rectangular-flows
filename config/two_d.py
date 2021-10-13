from .dsl import group, base, provides, GridParams


group(
    "2d",
    [
        "2uniforms",
        "2lines",
        "8gaussians",
        "checkerboard",
        "2spirals",
        "rings",
        "2marginals",
        "1uniform",
        "annulus",
        "split-gaussian",
        "von-mises-circle",
        "sin-wave-mixture"
    ]
)


@base
def config(dataset, use_baseline):
    return {
        "num_u_channels": 1,
        "use_cond_affine": not use_baseline,
        "pure_cond_affine": False,

        "dequantize": False,

        "batch_norm": False,

        "max_epochs": 2000,
        "max_grad_norm": None,
        "early_stopping": True,
        "max_bad_valid_epochs": 50,
        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "opt": "adam",
        "lr": 1e-3,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "epochs_per_test": 5,

        "num_valid_elbo_samples": 10,
        "num_test_elbo_samples": 100
    }


@provides("affine")
def affine(dataset, model, use_baseline):
    assert use_baseline
    return {
        "schema_type": "affine",
        "num_density_layers": 10
    }


# TODO: Make 2uniforms match paper
@provides("maf")
def maf(dataset, model, use_baseline):
    return  {
        "schema_type": "maf",

        "num_density_layers": 20 if use_baseline else 5,
        "ar_map_hidden_channels": [50] * 4,

        "st_nets": [10] * 2,
        "p_nets": [50] * 4,
        "q_nets": [50] * 4,
    }


@provides("maf-grid")
def maf_grid(dataset, model, use_baseline):
    return {
        "schema_type": "maf",

        "num_density_layers": 20 if use_baseline else 5,
        "ar_map_hidden_channels": GridParams([10] * 2, [50] * 4),

        "num_u_channels": 2,

        "st_nets": GridParams([10] * 2, [50] * 4),
        "p_nets": [10] * 2,
        "q_nets": [50] * 4,
    }


@provides("cond-affine-shallow-grid", "cond-affine-deep-grid")
def cond_affine_grid(dataset, model, use_baseline):
    assert not use_baseline

    if "deep" in model:
        num_layers = 5
        net_factor = 1
    else:
        num_layers = 1
        net_factor = 5

    return {
        "schema_type": "cond-affine",

        "num_density_layers": num_layers,

        "num_u_channels": 2,

        "st_nets": GridParams([10] * 2 * net_factor, [50] * 4 * net_factor),
        "p_nets": [10] * 2 * net_factor,
        "q_nets": [50] * 4 * net_factor
    }


@provides("dlgm-deep", "dlgm-shallow")
def dlgm_deep(dataset, model, use_baseline):
    assert not use_baseline

    if "deep" in model:
        cond_affine_model = "cond-affine-deep-grid"
    else:
        cond_affine_model = "cond-affine-shallow-grid"

    config = cond_affine_shallow_grid(dataset=dataset, model=cond_affine_model, use_baseline=False)

    del config["st_nets"]
    config["s_nets"] = "fixed-constant"
    config["t_nets"] = "identity"

    return config


@provides("realnvp")
def realnvp(dataset, model, use_baseline):
    return {
        "schema_type": "flat-realnvp",

        "num_density_layers": 1,
        "coupler_shared_nets": True,
        "coupler_hidden_channels": [10] * 2,

        "use_cond_affine": True,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
        "q_nets": [10] * 2,
    }


@provides("sos")
def sos(dataset, model, use_baseline):
    return {
        "schema_type": "sos",
        
        "num_density_layers": 3 if use_baseline else 2,
        "num_polynomials_per_layer": 2,
        "polynomial_degree": 4,

        "st_nets": [40] * 2,
        "p_nets": [40] * 4,
        "q_nets":  [40] * 4
    }


@provides("planar")
def planar(dataset, model, use_baseline):
    return {
        "schema_type": "planar",

        "num_density_layers": 10,

        "use_cond_affine": False,
        "cond_hidden_channels": [10] * 2,

        "p_nets": [50] * 4,
        # TODO: Make [50] * 4
        "q_nets": [10] * 2
    }


@provides("nsf-ar")
def nsf(dataset, model, use_baseline):
    return {
        "schema_type": "nsf",
        "autoregressive": True,
        "use_linear": False,

        "max_grad_norm": 5,

        "num_density_layers": 5,
        # "num_bins": 64 if use_baseline else 24,
        "num_bins": 8,
        # "num_hidden_channels": 32,
        "num_hidden_channels": 256,
        "num_hidden_layers": 2,
        # "tail_bound": 5,
        "tail_bound": 3,
        "dropout_probability": 0.,

        "lr_schedule": "cosine",
        "lr": 0.0005,
        "max_epochs": 1000,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
        "q_nets": [10] * 2,
    }


@provides("bnaf")
def bnaf(dataset, model, use_baseline):
    return {
        "schema_type": "bnaf",

        "num_density_layers": 1,
        "num_hidden_layers": 2,
        "hidden_channels_factor": 50 if use_baseline else 45,
        "activation": "soft-leaky-relu",

        "st_nets": [24] * 2,
        "p_nets": [24] * 3,
        "q_nets": [24] * 3
    }


@provides("non-square")
def non_square_flow(dataset, model, use_baseline):
    return {
        "non_square": True,
        "m_flow": use_baseline,

        "schema_type": "flat-realnvp",
        "underlying_flow": "realnvp",

        "num_density_layers": 5,

        "lr": GridParams(1e-3, 1e-4),
        "max_epochs": 5000,
        "epochs_per_test": 25,

        "regularization_param": GridParams(100, 1000, 10000) if use_baseline else GridParams(10, 50, 200),
        # "regularization_param": GridParams(10, 50, 200),
        "log_jacobian_method": "cholesky",
        "latent_dimension": 1,

        "likelihood_warmup": GridParams(False, True),
        "likelihood_warmup_start": 500,
        "likelihood_warmup_end": 1000,

        # NOTE: Adjust these for different datasets
        "vis_log_prob_min": -3,
        "vis_log_prob_max": -1,

        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 1,

        "prior": "affine" if use_baseline else "standard-normal",

        "num_u_channels": 0
    }
