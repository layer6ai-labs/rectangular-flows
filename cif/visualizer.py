import os

from collections import defaultdict

import numpy as np

import torch
import torch.utils.data
import torchvision.utils

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
from matplotlib.collections import LineCollection

import tqdm

from .metrics import metrics

from scipy.special import i0
from scipy.stats import vonmises
from scipy.stats import gaussian_kde


# TODO: Make return a matplotlib figure instead. Writing can be done outside.
class DensityVisualizer:
    def __init__(self, writer):
        self._writer = writer

    def visualize(self, density, epoch):
        raise NotImplementedError


class DummyDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch):
        return


class ImageDensityVisualizer(DensityVisualizer):
    @torch.no_grad()
    def visualize(self, density, epoch, write_folder=None, fixed_noise=None, extent=None, labels=None):
        density.eval()
        imgs = density.fixed_sample(fixed_noise)

        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        # NOTE: This may not normalize exactly as we would like, since we might not
        #       observe the full range of pixel values. It may be best to specify the
        #       range as the standard {0, ..., 255}, but just leaving it for now.
        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        grid_permuted = grid.permute((1,2,0))
        plt.imshow(grid_permuted.detach().cpu().numpy(), extent=extent)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        if write_folder:
            plt.savefig(os.path.join(write_folder, "samples.png"))
        else:
            self._writer.write_image("samples", grid, global_step=epoch)


class TwoDimensionalVisualizerBase(DensityVisualizer):
    _NUM_TRAIN_POINTS_TO_SHOW = 500
    _BATCH_SIZE = 1000
    _CMAP = "viridis"
    _CMAP_LL = "autumn_r"
    _CMAP_D = "cool"
    _PADDING = .2

    def __init__(self, writer, x_train, device):
        super().__init__(writer=writer)

        self._x = x_train
        self._device = device

    def _lims(self, t):
        return (
            t.min().item() - self._PADDING,
            t.max().item() + self._PADDING
        )

    def _plot_x_train(self):
        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        plt.scatter(x[:, 0], x[:, 1], c="k", marker=".", s=7, linewidth=0.5, alpha=0.5)

    def _plot_density(self, density):
        raise NotImplementedError

    def visualize(self, density, epoch, write_folder=None):
        self._plot_density(density)
        self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "density.png"), bbox_inches='tight')
        else:
            self._writer.write_figure("density", plt.gcf(), epoch)

        plt.close()


class TwoDimensionalDensityVisualizer(TwoDimensionalVisualizerBase):
    _GRID_SIZE = 150
    _CONTOUR_LEVELS = 50

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer, x_train, device)

        self._num_elbo_samples = num_elbo_samples

        x1_lims = self._lims(self._x[:, 0])
        x2_lims = self._lims(self._x[:, 1])

        self._grid_x1, self._grid_x2 = torch.meshgrid((
            torch.linspace(*x1_lims, self._GRID_SIZE),
            torch.linspace(*x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((grid_x1, grid_x2), dim=2).view(-1, 2)

        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

    def _plot_density(self, density):
        probs = []
        for x1_x2_batch, in tqdm.tqdm(self._loader, leave=False, desc="Plotting"):
            with torch.no_grad():
                log_prob = metrics(density, x1_x2_batch, self._num_elbo_samples)["log-prob"]
            probs.append(torch.exp(log_prob))

        probs = torch.cat(probs, dim=0).view(*self._grid_x1.shape).cpu()

        plt.figure()

        contours = plt.contourf(self._grid_x1, self._grid_x2, probs, levels=self._CONTOUR_LEVELS, cmap=self._CMAP)
        for c in contours.collections:
            c.set_edgecolor("face")
        cb = plt.colorbar()
        cb.solids.set_edgecolor("face")


class TwoDimensionalNonSquareVisualizer(TwoDimensionalVisualizerBase):
    _LINSPACE_SIZE = 1000
    _LINSPACE_LIMITS = [-3, 3]

    def __init__(self, writer, x_train, device, log_prob_low, log_prob_high):
        super().__init__(writer, x_train, device)
        self._log_prob_limits = [log_prob_low, log_prob_high]
        self._low_dim_space = torch.linspace(*self._LINSPACE_LIMITS, self._LINSPACE_SIZE)

    def visualize(self, density, epoch, write_folder=None):
        self._embedded_manifold = density.fixed_sample(self._low_dim_space.unsqueeze(1))

        super().visualize(density, epoch, write_folder)

        self._plot_manifold_distance(density)
        self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "distances.png"), bbox_inches='tight')
        else:
            self._writer.write_figure("manifold-distances", plt.gcf(), epoch)

        plt.close()

        self._plot_pullback_density(density, write_folder)
        if write_folder:
            plt.savefig(os.path.join(write_folder, "pullback.png"), bbox_inches='tight')
        else:
            self._writer.write_figure("pullback-density", plt.gcf(), epoch)

        plt.close()

        self._plot_ground_truth(density, write_folder)
        self._plot_x_train()

        if write_folder:
            plt.savefig(os.path.join(write_folder, "ground_truth.png"), bbox_inches='tight')
        else:
            self._writer.write_figure("ground-truth", plt.gcf(), epoch)

        plt.close()

    def _plot_density(self, density):
        log_probs = density.elbo(
            x=self._embedded_manifold,
            add_reconstruction=False,
            likelihood_wt=1.,
            visualization=True
        )["elbo"].squeeze()

        self._plot_along_manifold(
            pytorch_colours=log_probs[1:],
            cbar_limits=self._log_prob_limits,
            metric="log-likelihood"
        )

    def _plot_manifold_distance(self, density):
        squared_distances = ((self._embedded_manifold[:-1] - self._embedded_manifold[1:])**2).sum(axis=1)
        distances = torch.sqrt(squared_distances).detach()

        self._plot_along_manifold(
            pytorch_colours=distances,
            cbar_limits=[0, torch.max(distances)],
            metric="speed"
        )

    def _plot_along_manifold(self, pytorch_colours, metric, cbar_limits):
        fig, ax = plt.subplots(1, 1)
        plt.axis("off")

        colours = pytorch_colours.detach().numpy()

        xy = self._embedded_manifold.detach().numpy()
        xy_collection = np.concatenate([xy[:-1,np.newaxis,:], xy[1:,np.newaxis,:]], axis=1)

        cbar_min, cbar_max = cbar_limits
        cmap = self._CMAP
        if metric == "log-likelihood":
            cmap = self._CMAP_LL
            label = r'$\log p(x)$'
        if metric == "speed":
            cmap = self._CMAP_D
            label = r'speed'
        lc = LineCollection(
            xy_collection,
            cmap=cmap,
            norm=plt.Normalize(cbar_min, cbar_max),
            linewidths=3
        )
        lc.set_array(np.clip(colours, cbar_min, cbar_max))
        ax.add_collection(lc)
        axcb = fig.colorbar(lc, extend="both")
        axcb.set_label(label, fontsize=15)

    def _plot_pullback_density(self, density, write_folder):
        log_jac_jac_t = density.pullback_log_jac_jac_transpose(self._embedded_manifold)

        circle_projections = self._embedded_manifold / torch.sqrt(torch.sum(self._embedded_manifold**2, dim=1, keepdim=True))
        log_groundtruth_numerator = circle_projections[:,1]

        norm_const = 2*np.pi*i0(1)

        probs_unnorm = torch.exp(log_groundtruth_numerator - 1/2*log_jac_jac_t).detach().cpu().numpy()
        probs = probs_unnorm/norm_const

        pullback_np = np.stack([self._low_dim_space.detach().cpu().numpy(), probs], axis=0)
        if write_folder:
            np.save(os.path.join(write_folder, "pullback.npy"), pullback_np)

        plt.plot(self._low_dim_space.detach().cpu().numpy(), probs)

    def _plot_ground_truth(self, density, write_folder):
        log_probs = vonmises.logpdf(np.linspace(-np.pi, np.pi, num=1000, endpoint=False), 1., loc=np.pi/2)

        self._plot_along_circle(
            density=density,
            colours=log_probs,
            cbar_limits=self._log_prob_limits,
            write_folder=write_folder
        )

    def _plot_along_circle(self, density, colours, cbar_limits, write_folder):
        fig, ax = plt.subplots(1, 1)
        plt.axis("off")

        theta = np.linspace(-np.pi, np.pi, num=1000, endpoint=False)
        xy = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        xy_collection = np.concatenate([xy[:-1, np.newaxis, :], xy[1:, np.newaxis, :]], axis=1)

        cbar_min, cbar_max = cbar_limits
        cmap = self._CMAP_LL
        lc = LineCollection(
            xy_collection,
            cmap=cmap,
            norm=plt.Normalize(cbar_min, cbar_max),
            linewidths=3
        )
        lc.set_array(np.clip(colours, cbar_min, cbar_max))
        ax.add_collection(lc)
        axcb = fig.colorbar(lc, extend="both")
        axcb.set_label(r'$\log p(x)$', fontsize=15)

        theta = vonmises.rvs(1, size=1000, loc=np.pi/2)
        xy2 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        xy_torch = torch.tensor(xy2, dtype=torch.float32)
        z = density._extract_latent(xy_torch, **{"earliest_latent": True}).detach().cpu().numpy().reshape(-1)
        kde = gaussian_kde(z)
        xs = np.linspace(-3, 3, 1000)
        kde_np = np.stack([xs, kde.pdf(xs)], axis=0)
        if write_folder:
            np.save(os.path.join(write_folder, "kde.npy"), kde_np)
        # ax.plot(xs, kde.pdf(xs))
