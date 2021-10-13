#!/usr/bin/env python3

import glob
import os
import json
import shutil
import argparse
import numpy as np

from cif.experiment import test_and_visualize


_DEFAULT_RUNS_DIRECTORY = "runs"
_OUTPUT_FILE = "tabular_table.csv"
_EXPECTED_ENTRIES = 5
_K_SMALL = 1
_K_LARGE = 10
_DATASETS = ["power", "gas", "hepmass", "miniboone"]
_METHODS = ["RNF-ML-EXACT", f"RNF-ML-K-{_K_SMALL}", f"RNF-ML-K-{_K_LARGE}", "RNF-TS"]


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default=_DEFAULT_RUNS_DIRECTORY,
    help="Location for runs directory")
parser.add_argument("--overwrite-metrics", action="store_true",
    help="Overwrite existing metrics for each run")
parser.add_argument("--move-runs", action="store_true",
    help="Rearrange runs into hierarchical structure")
args = parser.parse_args()


all_runs = glob.glob(os.path.join(args.dir, "*"))


# Create full results table, and directory for runs if args.move_runs == True.
# You will get errors if the directories exist.
results_table = {}
for method in _METHODS:
    results_table[method] = {}
    for dataset in _DATASETS:
        results_table[method][dataset] = []
        if args.move_runs:
            os.makedirs(os.path.join(args.dir, method, dataset))


# Test all relevant runs and move to new directories
for run in all_runs:
    try:
        with open(os.path.join(run, "config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {run} because no config")
        continue

    dataset = config["dataset"]

    if not dataset in _DATASETS:
        print(f"Skipping {run} because {dataset} not a tabular dataset")
        continue

    # Figure out the method by referencing the config
    if config["m_flow"]:
        method = "RNF-TS"
    elif config["log_jacobian_method"] == "cholesky":
        method = "RNF-ML-EXACT"
    elif config["log_jacobian_method"] == "hutch_with_cg" and config["hutchinson_samples"] == _K_SMALL:
        method = f"RNF-ML-K-{_K_SMALL}"
    elif config["log_jacobian_method"] == "hutch_with_cg" and config["hutchinson_samples"] == _K_LARGE:
        method = f"RNF-ML-K-{_K_LARGE}"
    else:
        print(f"Skipping {run} because method is unknown")

    fid_like_metric = test_and_visualize(config, run, overwrite=args.overwrite_metrics, test_fid=True)["test_fid"]
    results_table[method][dataset].append(fid_like_metric)

    if args.move_runs:
        shutil.move(run, os.path.join(args.dir, method, dataset, os.path.basename(run)))


# Collect all the results, excluding NaN values. Give warnings for NaNs and fewer entries than expected.
csv_output = "Method, " + ", ".join(_DATASETS) + "\n"
for method, datasets in results_table.items():
    csv_output += f"{method}"
    for dataset, results in datasets.items():
        if len(results) < _EXPECTED_ENTRIES:
            print(f"Warning: Method {method} on dataset {dataset} only has {len(results)} entries")
        num_non_nan = np.sum(~np.isnan(results))
        if num_non_nan < len(results):
            print(f"Warning: Method {method} on dataset {dataset} has only {num_non_nan} non-NaN entries")

        mean = np.nanmean(results)
        stderr = np.nanstd(results, ddof=1)/np.sqrt(num_non_nan)

        csv_output += f", {mean:.3f} +/- {stderr:.3f}"

    csv_output += "\n"


print(csv_output)
with open(_OUTPUT_FILE, "w") as f:
    f.write(csv_output)
