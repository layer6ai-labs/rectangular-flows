import json
import random
from pathlib import Path
import sys
import subprocess
import os
import tqdm

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.optim as optim
import torch.nn as nn

from .trainer import Trainer
from .datasets import get_loaders
from .visualizer import (
    DummyDensityVisualizer,
    ImageDensityVisualizer,
    TwoDimensionalDensityVisualizer,
    TwoDimensionalNonSquareVisualizer
)
from .models import get_density
from .writer import Writer, DummyWriter
from .metrics import metrics, get_fid_function
from .non_square_helpers import get_non_square_parameters, get_non_square_train_metrics

from config import get_schema


def train(config, resume_dir):

    experiment_info = setup_experiment(config=config, resume_dir=resume_dir)
    writer = experiment_info["writer"]
    density = experiment_info["density"]

    writer.write_json("config", config)

    writer.write_json("model", {
        "num_params": num_params(density),
        "schema": get_schema(config)
    })

    writer.write_textfile("git-head", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii"))
    writer.write_textfile("git-diff", subprocess.check_output(["git", "diff"]).decode("ascii"))

    print("\nConfig:")
    print(json.dumps(config, indent=4))
    print(f"\nNumber of parameters: {num_params(density):,}\n")

    experiment_info["trainer"].train()


def test_and_visualize(config, resume_dir, write_test=True, overwrite=False, test_fid=False):
    EVAL_FID_SAMPLES = 50000
    config["num_fid_samples"] = EVAL_FID_SAMPLES
    config["use_test_fid"] = test_fid

    if not os.path.isdir(resume_dir):
        print(f"\t{resume_dir} is not a directory")
        return

    metrics_path = os.path.join(resume_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            old_metrics = json.load(f)
        if not overwrite:
            print("\tNot testing since metrics.json exists")
            return old_metrics
    else:
        old_metrics = {}

    experiment_info = setup_experiment(
        config={**config, "write_to_disk": False},
        resume_dir=resume_dir,
        testing=True
    )
    trainer = experiment_info["trainer"]
    density = experiment_info["density"]
    visualizer = experiment_info["visualizer"]

    if write_test:
        with torch.no_grad():
            test_metrics = trainer.test()

        test_metrics = {k: v.item() for k, v in test_metrics.items()}
        if config["use_test_fid"]:
            test_metrics["test_fid"] = test_metrics.pop("fid")

        test_metrics = {
            **old_metrics,
            **test_metrics
        }
        json_dump = json.dumps(test_metrics, indent=4)

        print(json_dump)
        with open(metrics_path, "w") as f:
            f.write(json_dump)

        return test_metrics

    if not (config["dataset"] in ["power", "gas", "hepmass", "miniboone", "bsds300"]):
        visualizer.visualize(density, epoch=0, write_folder=resume_dir)


def visualize_two_dim_manifold(config, resume_dir):
    assert config["dataset"] in ["mnist", "fashion-mnist"]
    assert config["latent_dimension"] == 2

    config["use_fid"] = False

    experiment_info = setup_experiment(
        config={**config, "write_to_disk": False},
        resume_dir=resume_dir,
        testing=True
    )
    density = experiment_info["density"]
    visualizer = experiment_info["visualizer"]

    MIN = -3
    MAX = 3
    N_GRID = 8

    x = np.linspace(MIN, MAX, N_GRID)
    y = np.linspace(MAX, MIN, N_GRID)

    xv, yv = np.meshgrid(x, y)
    xy = np.stack((xv.reshape(N_GRID*N_GRID, ), yv.reshape(N_GRID*N_GRID, )), axis=1)

    noise = torch.from_numpy(xy).to(torch.device("cuda"), dtype=torch.float32)
    visualizer.visualize(density, epoch=0, write_folder=resume_dir, fixed_noise=noise,
        extent=[MIN, MAX, MIN, MAX], labels=["$z_1$", "$z_2$"])


def generate_ood_metrics(config, resume_dir):
    ood_mapping_table = {
        "mnist": "fashion-mnist",
        "fashion-mnist": "mnist",
        "cifar10": "svhn",
        "svhn": "cifar10"
    }
    config["ood"] = True
    config["use_fid"] = False
    original_dataset = config["dataset"]

    print(f"OOD for model trained on {config['dataset']}")

    BATCH_SIZE = 1000
    config["train_batch_size"] = BATCH_SIZE
    config["test_batch_size"] = BATCH_SIZE

    config["log_jacobian_method"] = "cholesky"

    # Start with two runs of metrics in-sample
    config["other_dataset"] = False

    config["ood_train"] = True
    single_ood_test(config, resume_dir)

    config["ood_train"] = False
    single_ood_test(config, resume_dir)

    # Now do two runs out-of-sample
    config["dataset"] = ood_mapping_table[original_dataset]
    config["other_dataset"] = True

    config["ood_train"] = True
    single_ood_test(config, resume_dir)

    config["ood_train"] = False
    single_ood_test(config, resume_dir)


def ood_classification(resume_dir, low_dim=False):
    array_tail = "_ld" if low_dim else ""
    in_sample_train_array = np.load(os.path.join(resume_dir, f"ood_metrics_train_in{array_tail}.npy"))
    in_sample_test_array = np.load(os.path.join(resume_dir, f"ood_metrics_test_in{array_tail}.npy"))
    out_sample_train_array = np.load(os.path.join(resume_dir, f"ood_metrics_train_out{array_tail}.npy"))
    out_sample_test_array = np.load(os.path.join(resume_dir, f"ood_metrics_test_out{array_tail}.npy"))

    def classify_ood(index):
        def make_dataset(arr, zeros):
            if zeros:
                labels = np.zeros(arr.shape[0])
            else:
                labels = np.ones(arr.shape[0])
            return np.stack((arr[:, index], labels), axis=1)

        train_in = make_dataset(in_sample_train_array, True)
        train_out = make_dataset(out_sample_train_array, False)
        train_dataset = np.concatenate((train_in, train_out), axis=0)

        test_in = make_dataset(in_sample_test_array, True)
        test_out = make_dataset(out_sample_test_array, False)
        test_dataset = np.concatenate((test_in, test_out), axis=0)

        clf = DecisionTreeClassifier(max_depth=1)
        clf = clf.fit(train_dataset[:,0,np.newaxis], train_dataset[:,1])

        predictions = clf.predict(test_dataset[:,0,np.newaxis])
        classification_rate = np.mean(predictions == test_dataset[:,1])
        return classification_rate

    likelihood_classification_rate = classify_ood(index=0)
    reconstruction_classification_rate = classify_ood(index=1)

    print("**** Classification Rate ****")
    print(f"\tLikelihood: {likelihood_classification_rate:.2f}")
    print(f"\tReconstruction: {reconstruction_classification_rate:.2f}")


def single_ood_test(config, resume_dir):
    torch.cuda.empty_cache()

    experiment_info = setup_experiment(
        config={**config, "write_to_disk": False},
        resume_dir=resume_dir,
        testing=True
    )
    trainer = experiment_info["trainer"]
    density = experiment_info["density"]
    density.eval()

    with torch.no_grad():
        test_metrics = trainer.test()
    test_metrics = {k: v.item() for k, v in test_metrics.items()}

    json_dump = json.dumps(test_metrics, indent=4)
    print(json_dump)

    file_id = f"{config['dataset']}_train={config['ood_train']}"
    ood_metrics_path = os.path.join(resume_dir, f"ood_metrics_{file_id}.json")
    with open(ood_metrics_path, "w") as f:
        f.write(json_dump)


def print_model(config):
    density, _, _, _ = setup_density_and_loaders(
        config={**config, "write_to_disk": False},
        device=torch.device("cpu")
    )
    print(density)


def print_num_params(config):
    density, _, _, _ = setup_density_and_loaders(
        config={**config, "write_to_disk": False},
        device=torch.device("cpu")
    )
    print(f"Number of parameters: {num_params(density):,}")


def setup_density_and_loaders(config, device):
    train_loader, valid_loader, test_loader = get_loaders(
        dataset=config["dataset"],
        device=device,
        data_root=config["data_root"],
        # NOTE: Just use train data for FID as is standard in the literature
        make_valid_loader=(config["early_stopping"] and not config.get("use_fid", False) and not config.get("ood", False)),
        train_batch_size=config["train_batch_size"],
        valid_batch_size=config["valid_batch_size"],
        test_batch_size=config["test_batch_size"]
    )

    density = get_density(
        schema=get_schema(config=config),
        x_train=train_loader.dataset.x
    )

    # TODO: Could do lazily inside Trainer
    density.to(device)

    return density, train_loader, valid_loader, test_loader


def load_run(run_dir, device):
    run_dir = Path(run_dir)

    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    density, train_loader, valid_loader, test_loader = setup_density_and_loaders(
        config=config,
        device=device
    )

    try:
        checkpoint = torch.load(run_dir / "checkpoints" / "best_valid.pt", map_location=device)
    except FileNotFoundError:
        checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location=device)

    print("Loaded checkpoint after epoch", checkpoint["epoch"])

    density.load_state_dict(checkpoint["module_state_dict"])

    return density, train_loader, valid_loader, test_loader, config, checkpoint


def get_visualizer(config, writer, train_data, device):
    if config["dataset"] in ["cifar10", "svhn", "fashion-mnist", "mnist"]:
        return ImageDensityVisualizer(writer=writer)

    elif train_data.shape[1:] == (2,):
        if config["model"] == "non-square":
            return TwoDimensionalNonSquareVisualizer(
                writer=writer,
                x_train=train_data,
                device=device,
                log_prob_low=config["vis_log_prob_min"],
                log_prob_high=config["vis_log_prob_max"]
            )
        else:
            return TwoDimensionalDensityVisualizer(
                writer=writer,
                x_train=train_data,
                num_elbo_samples=config["num_test_elbo_samples"],
                device=device
            )

    else:
        return DummyDensityVisualizer(writer=writer)


def setup_experiment(config, resume_dir, testing=False):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"]+1)
    random.seed(config["seed"]+2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    density, train_loader, valid_loader, test_loader = setup_density_and_loaders(
        config=config,
        device=device
    )

    if config["opt"] == "sgd":
        opt_class = optim.SGD
    elif config["opt"] == "adam":
        opt_class = optim.Adam
    elif config["opt"] == "adamax":
        opt_class = optim.Adamax
    else:
        assert False, f"Invalid optimiser type {config['opt']}"

    if config.get("non_square", False):
        parameter_list = get_non_square_parameters(density, config["m_flow"])
    else:
        parameter_list = [density.parameters()]

    optimizers = [
        opt_class(
            params,
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        for params in parameter_list
    ]

    def get_lr_scheduler(opt):
        if config["lr_schedule"] == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=opt,
                T_max=config["max_epochs"]*len(train_loader),
                eta_min=0.
            )
        elif config["lr_schedule"] == "none":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt,
                lr_lambda=lambda epoch: 1.
            )
        else:
            assert False, f"Invalid learning rate schedule `{config['lr_schedule']}'"

        return lr_scheduler

    lr_schedulers = [get_lr_scheduler(opt) for opt in optimizers]

    if config["write_to_disk"]:
        if resume_dir is None:
            logdir = config["logdir_root"]
            make_subdir = True
        else:
            logdir = resume_dir
            make_subdir = False

        writer = Writer(
            logdir=logdir,
            make_subdir=make_subdir,
            tag_group=config["dataset"],
            rundir_tail=config["rundir_tail"]
        )
    else:
        writer = DummyWriter(logdir=resume_dir)

    visualizer = get_visualizer(
        config=config,
        writer=writer,
        train_data=train_loader.dataset.x,
        device=device
    )

    # NOTE: Sorry about the spaghetti code below
    if config.get("non_square", False):
        (
            train_metrics,
            likelihood_introduction_epoch,
            early_stopping_start_epoch
        ) = get_non_square_train_metrics(config)

        if config["dataset"] in ["cifar10", "svhn", "fashion-mnist", "mnist",
                    "power", "gas", "hepmass", "miniboone", "bsds300"]:
            def valid_loss(density, x):
                return torch.tensor(0.)

            if testing and config.get("ood", False):
                def test_metrics(density, x):
                    return density.ood(x)
            else:
                def test_metrics(density, x):
                    return {"loss": torch.tensor(0.)}

        else:
            def valid_loss(density, x):
                return -metrics(density, x, config["num_valid_elbo_samples"])["elbo"]
            def test_metrics(density, x):
                return {"loss": -density.elbo(x, add_reconstruction=False, likelihood_wt=1.)["elbo"].mean()}

    else:
        def train_metrics(density, x, epoch):
            return {"loss": -density.elbo(x)["elbo"].mean()}

        likelihood_introduction_epoch = 0
        early_stopping_start_epoch = 0

        def valid_loss(density, x):
            return -metrics(density, x, config["num_valid_elbo_samples"])["log-prob"]

        def test_metrics(density, x):
            return metrics(density, x, config["num_test_elbo_samples"])

    fid_function = None
    if (
        config["dataset"] in ["cifar10", "svhn", "fashion-mnist", "mnist",
            "power", "gas", "hepmass", "miniboone", "bsds300"]
        and
        config["use_fid"]
    ):
        loader = test_loader if config.get("use_test_fid", False) else train_loader
        fid_function = get_fid_function(config, loader)

    trainer = Trainer(
        module=density,
        train_metrics=train_metrics,
        valid_loss=valid_loss,
        test_metrics=test_metrics,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        max_epochs=config["max_epochs"],
        max_grad_norm=config["max_grad_norm"],
        early_stopping=config["early_stopping"],
        early_stopping_start_epoch=early_stopping_start_epoch,
        max_bad_valid_epochs=config["max_bad_valid_epochs"],
        valid_frequency=2 if config.get("m_flow", False) else 1,
        visualizer=visualizer,
        writer=writer,
        epochs_per_test=config["epochs_per_test"],
        should_checkpoint_latest=config["should_checkpoint_latest"],
        should_checkpoint_best_valid=config["should_checkpoint_best_valid"],
        device=device,
        non_square=config.get("non_square", False),
        likelihood_introduction_epoch=likelihood_introduction_epoch,
        fid_function=fid_function,
        only_testing=testing,
        ood_test=config.get("ood", False),
        ood_with_train=config.get("ood_train", False),
        other_dataset=config.get("other_dataset", False)
    )

    return {
        "density": density,
        "trainer": trainer,
        "writer": writer,
        "visualizer": visualizer
    }


def num_params(module):
    return sum(p.view(-1).shape[0] for p in module.parameters())
