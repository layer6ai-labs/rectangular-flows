<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>


# Rectangular Normalizing Flows README

This is the code we used for our paper, [Rectangular Flows for Manifold Learning](https://arxiv.org/abs/2106.01413) (NeurIPS 2021). Our code builds directly off commit hash `d87adf09e37896f7095fc0c15f89940b8967973a`   from the [CIF codebase](https://github.com/jrmcornish/cif) accompanying [Cornish et al.](https://arxiv.org/abs/1909.13833) (ICML 2020), which we cite in our manuscript. Cornish et al.'s code did not have a license, and given its public availability, we used it freely. Some dependencies do have licenses (see `/gitmodules`), all of which do allow us to use the respective code.

Note that there are many options leftover from the CIFs codebase which are not relevant for our purposes, particularly the larger number of 2D datasets and flow models implemented. For synthetic data we only use `von-mises-circle`, and our `--model` argument throughout is simply `non-square`. We have also removed the git submodules `ffjord` and `residual-flows` as they were causing errors with our conda environments. We highlight other major differences at the end of this README. Please refer to the CIF codebase `README` file for a more general description of the features of the codebase.

## Setup

First, install submodules:

    $ git submodule init
    $ git submodule update

Next, install dependencies. If you use `conda`, the following will create an environment called `rnf`:

    conda env create -f environment-lock.yml

Activate this with

    conda activate rnf

before running any code or tests.

If you don't use `conda`, then please see `environment.yml` for a list of required packages, which will need to be installed manually e.g. via `pip`.

### Obtaining datasets

Our code runs on several types of datasets, including synthetic 2-D data, tabular data, and image data. The 2-D datasets are automatically generated, and the image datasets are downloaded automatically. However the tabular datasets will need to be manually downloaded from [this location](https://zenodo.org/record/1161203). The following should do the trick:

    mkdir -p data/ && wget -O - https://zenodo.org/record/1161203/files/data.tar.gz | tar --strip-components=1 -C data/ -xvzf - data/{gas,hepmass,miniboone,power}

This will download the data to `data/`. If `data/` is in the same directory as `main.py`, then everything will work out-of-the-box by default. If not, the location to the data directory will need to be specified as an argument to `main.py` (see `--help`).

# Generating Results

Here we provide specific instructions for generating the experimental results in the paper. In all cases, completed runs will be sent to the `runs` directory by default, and can then be inspected or evaluated from there. We have also included pre-trained versions of the more expensive models, including all the tabular and image runs featured in the paper. Details are in their respective sections.

## Von Mises Experiments

To train our model on the von Mises circle dataset, run:

    ./main.py --model non-square --dataset von-mises-circle
 
Note that this will actually launch a grid of runs over various values for the `regularization_param`, `likelihood_warmup`, and `lr` config values as described in Appendix F.1. To overrirde the grid search and just launch a specific configuration, please edit the `non-square` section of the file `config/two_d.py` by removing the `GridParams` specification.

If you are on a GPU device, please specify `CUDA_VISIBLE_DEVICES=`, i.e. to the empty string, as this experiment is not currently supported to run on the GPU.

To launch a baseline two-step procedure run, add the flag `--baseline` to the command above.

To visualize the result of an experiment, either use tensorboard (described below), or locate the directory in which information about the run is stored (e.g. `<run>=runs/MthDD_HH-MM-SS`) and use the command

    ./main.py --resume <run> --test

This will produce the plots:
1. `density.png` showing the density on the manifold
2. `distances.png` showing the "speed" at which the manifold is parametrized
3. `pullback.png` showing the pullback density required to be learned.

We can also generate the combined pullback density as shown in the paper using the notebook `pullback_plots.ipynb`. These runs were chosen with the following hyperparameters:
- __Baseline:__
  - "lr": 0.0001
  - "regularization_param": 10000
  - "likelihood_warmup": false
- __Ours:__ 
  - "lr": 0.001
  - "regularization_param": 50
  - "likelihood_warmup": false

Note that these are not the _best_ runs; rather, they are the _most representative_ runs of the overall performance of each method. As noted in the Appendix, our runs do consistently perform better than the baseline, capturing the density more often than not.

## Tabular Data Experiments

To train a tabular model, run:

    CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --dataset <dataset-name>

where `<dataset-name>` is one of `power`, `gas`, `hepmass`, or `miniboone`.
This will launch the RNFs-ML (exact) variant of our model.
To launch the RNFs-ML ($K=$`<k>`) variant, add `--config log_jacobian_method=hutch_with_cg --config hutchinson_samples=<k>` to the command above, replacing `<k>` with the desired number of samples in the Hutchinson estimator.
To launch the RNFs-TS baseline, add `--baseline` to the command above; this is agnostic to exact vs. approximate.

To evaluate a single completed run, locate its run directory -- say it is `<run>` -- and run the command

    CUDA_VISIBLE_DEVICES=0 ./main.py --resume <run> --test

To reproduce the results in the table, we have included the script `tabular_runs.sh`, which first launches the entire suite of (4 models) x (4 datasets) x (5 seeds) = 80 runs, and then invokes the script `tabular_evaluate.py` to evaluate the runs. The table will be both outputted to the screen and saved as CSV to `tabular_table.csv` in this directory.

__NOTE THAT THIS PROCESS WILL TAKE QUITE A LONG TIME.__ Alternatively, you can just evaluate the pre-trained runs provided at [this link](https://www.dropbox.com/s/ic8j19zuo3b7gst/all_tabular_runs.zip?dl=0). You can obtain the table in the text by running

    ./tabular_evaluate.py -d <path-to-unzipped-directory>

You can add the option `--overwrite-metrics` to overwrite the metrics attached to each run (may also be good to specify `CUDA_VISIBLE_DEVICES` in that case). You can also use `--move-runs` to rearrange the runs into a hierarchical directory format upon completion.

## Image Experiments

To train an image model, run:

    CUDA_VISIBLE_DEVICES=<device(s)> ./main.py --model non-square --dataset <dataset-name>

where `<devices(s)>` is a string specifying one or more `CUDA` devices, e.g. `<devices(s)>=2,3,4`, and `<dataset-name>` is either `mnist`, `fashion-mnist`, `svhn`, or `cifar10` (although we only include results from MNIST and Fashion-MNIST in the manuscript).

Again, an RNFs-TS method can be launched by appending the flag `--baseline`.

A variety of parameters were modified as noted in Appendix F.3; to launch a particular run matching what is noted in the paper, you can either check the list of runs in `image_runs.sh`, or modify the hyperparameters directly in the `config/images.py` file under the heading `non-square`. Switching between RNFs-ML (exact) and RNFs-ML ($K=$`<k>`) is the same as the previous section, although increasing `K` too much will greatly increase memory requirements.

__We have also included each of the trained runs from the main text__ at [this link](https://www.dropbox.com/s/32jueauj9ttwx6h/image_runs.zip?dl=0). The mapping from tags to method/dataset combination is as follows:

|                   | MNIST                 | Fashion-MNIST         |
| -----------       | -----------           | -----------           |
| RNFs-ML (exact)   | `May14_16-39-46`      | `Apr21_18-17-16`      |
| RNFs-ML ($K=1$)   | `May11_21-25-09`      | `May19_14-01-39`      |
| RNFs-ML ($K=4$)   | `May17_21-57-51`      | `May23_14-35-35`      |
| RNFs-TS           | `Apr28_23-52-35_5`    | `May03_22-03-39_2`    |

Also included in this link are runs `Aug05_14-56-27` and `Aug05_14-57-56`, which are runs on MNIST and Fashion-MNIST, respectively, with a latent dimension of just `2`. Furthermore, we have included `Jul27_21-55-35` and `Aug04_16-02-21`, which are models trained on CIFAR10 with RNFs-ML (exact) and RNFs-TS, respectively.

To evaluate a completed run, locate the run directory and launch the command

    CUDA_VISIBLE_DEVICES=<device(s)> ./main.py --resume <run> <test-option>

where `<test-option>` is either `--test` to evaluate on FID, or `--test-ood` to perform an evaluation of Out-of-Distribution detection performance. For the two-dimensional runs noted above, you have an additional option `--two-dim-manifold` which will output a two-dimensional manifold visualization called `samples.png` to the run directory.

You may also refer to `ood_hist_plots.ipynb` to output the OOD histograms from the paper. Again, ensure that you are mapping to the correct run location and that you have run the `--test` option first on the runs considered.

## Miscellaneous - Run Directory and Tensorboard Logging

By default, running `./main.py` will create a directory inside `runs/` that contains

- Configuration info for the run
- Version control info for the point at which the run was started
- Checkpoints created by the run

This allows easily resuming a previous run via the `--resume` flag, which takes as argument the directory of the run that should be resumed.
To avoid creating this directory, use the `--nosave` flag.

The `runs/` directory also contain Tensorboard logs giving various information about the training run, including 2-D density plots in this case. To inspect this run the following command in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.
For 2D datasets, the "Images" tab shows the learned density, and for image datasets, the "Images" tab shows samples from the model over training.
The "Text" tab also shows the config used to produce each run.

# Major Differences Versus CIF Codebase

Besides the differences listed in the first section, there are some major code changes which allow rectangular flows to function in this codebase. The main elements we highlight are the `NonSquareHeadDensity`/`NonSquareTailDensity` classes, additions to the `cif/experiment.py` file (which themselves tie in with added metrics), added functionality to the `Density` class, and additional specification of Jacobian-vector product layers.

## Non-Square Density

The main workhorse of the codebase is the `NonSquareHeadDensity` class (in `cif/models/components/densities/non_square.py`), which allows for specification of rectangular flows. This class acts as a `Density` object, and specifies the head (or end, in the generative direction) of the rectangular flow. Its main purpose is twofold:
1. Provide support for computing $\log \det J^\top J$ for rectangular $J$, either exactly or approximately.
2. Compute reconstruction error.
This class eventually links back to `NonSquareTailDensity`, building a stack of transformations from an instance of that class to itself from which we can then compute the generative direction of the flow, along with the Jacobian-vector product. 

The class `ManifoldFlowHeadDensity` inherits from `NonSquareHeadDensity` and allows the specification of a baseline two-step training procedure (as per [Brehmer & Cranmer, 2020](https://arxiv.org/abs/2003.13913)). This class mainly allows for separating the parameters of the low-dimensional flow from the flow in high-dimensional ambient space.

__NOTE:__ It may be quite easy to improve the results of rectangular flows by modifying the upsampling procedure used in `NonSquareTailDensity`. Currently, we simply embed the low dimensional random vector into ambient space using a permutation which is randomized at the start of training but then held fixed. This could be replaced by some kind of more expressive upsampling layer.

## Added Experiments and Metrics

This codebase adds two new metrics over the CIF codebase:
1. FID calculation
2. Evaluation of likelihood-based out-of-distribution (OOD) detection

The FID calculation can be found in `cif/metrics.py`. The OOD calculation can be found in `cif/experiment.py`, which also demonstrates the invocation of the FID metric. We are required to make some modifications to `trainer.py` to accommodate these new metrics.

An additional experimental tool added in this codebase is the capacity to visualize manifolds, which is not possible in the standard normalizing flow setting. For visualizing two-dimensional data on a one-dimensional manifold, we have added `TwoDimensionalNonSquareVisualizer` in the `cif/visualizer.py` file. For visualizing image data on a two-dimensional manifold, we have added the function `visualize_two_dim_manifold` within `cif/experiment.py`, whose invocation we describe in the Image Experiments section above.

## More Functionality in `Density`

Within the generic `Density` base class in `cif/models/components/densities/density.py`, we have added several `mode`s to `forward`, including:
1. `jvp` - for calculating Jacobian-vector products
2. `ood` - for getting relevant out-of-distribution detection metrics
3. `extract-latent` - for extracting the latent variable in low dimensions 

## Jacobian-Vector Product Layers

We have mentioned Jacobian-vector products (jvps) already, but provide more context in this section. A major part of rectangular flows is the computation of both vector-Jacobian products (vjps) and jvps. PyTorch already has a very efficient method for computation of vjps, since backpropagation is an instance of a vjp and is ubiquitous in neural network training. However, native computation of the jvp is based on the less efficient [double backward trick](https://j-towns.github.io/2017/06/12/A-new-trick.html), thus requiring us to define custom jvp layers. You can see these throughout the code, for example in `cif/models/couplers.py`, `cif/models/networks.py`, or `cif/models/jvp_layers.py`. A chain of jvps is called on a flow in the ambient space in the `jvp_forward` method of `NonSquareHeadDensity`; this assumes that the aforementioned stack of transformations is already built up. 

Extending this codebase to other flow methods and neural networks will require writing new jvp layers, at least until jvps are coded efficiently in native PyTorch.

# Bibtex

    @inproceedings{caterini2021rectangular,
        title={Rectangular Flows for Manifold Learning},
        author={Anthony L. Caterini and Gabriel Loaiza-Ganem and Geoff Pleiss and John P. Cunningham},
        year={2021},
        journal={Advances in Neural Information Processing Systems}
    }
