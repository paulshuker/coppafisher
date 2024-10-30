import enum
import importlib.resources as importlib_resources
import itertools
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from coppafish.omp import coefs, scores_torch
from coppafish.setup import config
from coppafish.setup.notebook import NotebookPage
from coppafish.spot_colours import base as spot_colours_base
from coppafish.plot.results_viewer.subplot import Subplot


class ViewOMPImage(Subplot):
    class Options(enum.Enum):
        COEFFICIENTS = enum.auto()
        DP_SCORES = enum.auto()
        ITERATIONS = enum.auto()

    def __init__(
        self,
        nbp_basic: NotebookPage,
        nbp_filter: NotebookPage,
        nbp_register: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage | None,
        local_yxz: np.ndarray[int],
        spot_tile: int,
        spot_no: int,
        spot_gene_no: int,
        spot_colour: np.ndarray,
        method: str,
        im_size: int = 8,
        z_planes: Tuple[int] = (-2, -1, 0, 1, 2),
        show: bool = True,
    ):
        """
        Display omp coefficients around the local neighbourhood of a found spot position.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_filter (NotebookPage): `filter` notebook page.
            nbp_register (NotebookPage): `register` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage): `omp` notebook page or none.
            local_yxz (`(3) ndarray[int]`): the pixel position relative to its tile's bottom-left corner.
            spot_tile (int-like): tile index the pixel is on.
            spot_no (int-like or none): spot index to be plotted.
            spot_gene_no (int-like or none): spot's gene index. The index will be a selectable option on the slider if
                given. None if not given.
            spot_colour (`(n_rounds_use x n_channels_use) ndarray[float]`): the spot's colour.
            method (str): gene calling method.
            im_size (int, optional): number of pixels out from the central pixel to plot in x and y to create the
                square images. Default: 8.
            z_planes (tuple of int, optional): z planes to show. 0 is the central z plane. Default: (-2, -1, 0, 1, 2).
            show (bool, optional): display the plot once built. False is useful when unit testing. Default: true.
        """
        assert len(z_planes) > 3
        n_genes = nbp_call_spots.gene_names.size
        n_rounds_use, n_channels_use = len(nbp_basic.use_rounds), len(nbp_basic.use_channels)
        min_intensity = config.get_default_for("omp", "minimum_intensity")
        max_genes = config.get_default_for("omp", "max_genes")
        dot_product_weight = config.get_default_for("omp", "dot_product_weight")
        dot_product_threshold = config.get_default_for("omp", "dot_product_threshold")
        norm_shift = config.get_default_for("omp", "coefficient_normalisation_shift")
        mean_spot_filepath = importlib_resources.files("coppafish.omp").joinpath("mean_spot.npy")
        mean_spot: np.ndarray = np.load(mean_spot_filepath).astype(np.float32)
        if nbp_omp is not None:
            min_intensity = float(nbp_omp.associated_configs["omp"]["minimum_intensity"])
            max_genes = int(nbp_omp.associated_configs["omp"]["max_genes"])
            dot_product_weight = float(nbp_omp.associated_configs["omp"]["dot_product_weight"])
            dot_product_threshold = float(nbp_omp.associated_configs["omp"]["dot_product_threshold"])
            norm_shift = float(nbp_omp.associated_configs["omp"]["coefficient_normalisation_shift"])
            mean_spot = nbp_omp.mean_spot
        yxz_min = local_yxz.copy() + np.array([-im_size, -im_size, min(z_planes)], int)
        yxz_max = local_yxz.copy() + np.array([im_size, im_size, max(z_planes)], int) + 1
        image_shape = tuple((yxz_max - yxz_min).tolist())
        yxz = np.meshgrid(
            *[np.linspace(yxz_min[i], yxz_max[i] - 1, yxz_max[i] - yxz_min[i]) for i in range(3)], indexing="ij"
        )
        yxz = np.array(yxz).reshape((3, -1), order="F")
        yxz = yxz.T
        colours = spot_colours_base.get_spot_colours_new(
            yxz,
            nbp_filter.images,
            nbp_register.flow,
            nbp_register.icp_correction,
            int(spot_tile),
            nbp_basic.use_rounds,
            nbp_basic.use_channels,
            out_of_bounds_value=0,
        )
        colours *= nbp_call_spots.colour_norm_factor[[spot_tile]].astype(np.float32)
        intensity = colours.max(2).min(1)
        is_intense = (intensity >= min_intensity).nonzero()
        del intensity
        bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
        solver = coefs.CoefficientSolverOMP()
        coefficients = np.zeros((colours.shape[0], n_genes), np.float32)
        coefficients[is_intense], dp_scores = solver.solve(
            colours[is_intense],
            bled_codes,
            solver.create_background_bled_codes(n_rounds_use, n_channels_use),
            max_genes,
            dot_product_weight,
            dot_product_threshold,
            norm_shift,
            return_dp_scores=True,
        )
        dp_scores_all = np.zeros((dp_scores.shape[0], colours.shape[0], n_genes + n_channels_use), np.float32)
        dp_scores_all[:, is_intense[0]] = dp_scores
        dp_scores = dp_scores_all.copy()
        del dp_scores_all
        shape = image_shape + (-1,)
        xy_diameter = im_size * 2 + 1
        shape_kwargs = dict(order="F")
        coefficients = coefficients.reshape(shape, **shape_kwargs)
        dp_scores = np.array([score.reshape(shape, **shape_kwargs) for score in dp_scores], float)
        dp_scores_assigned = [
            [[0 for _ in range(len(z_planes))] for _ in range(xy_diameter)] for _ in range(xy_diameter)
        ]
        iteration_counts = (~np.isclose(coefficients, 0)).sum(3)
        assigned_genes = [[[[] for _ in range(len(z_planes))] for _ in range(xy_diameter)] for _ in range(xy_diameter)]
        all_assigned_genes = set()
        for i, j, k in itertools.product(range(xy_diameter), range(xy_diameter), range(len(z_planes))):
            assigned_genes[i][j][k] = (~np.isclose(coefficients[i, j, k], 0)).nonzero()[0].tolist()
            # Keep the dot product score the assigned gene(s) had when they were assigned.
            dp_scores_assigned_ijk = [0] * len(assigned_genes[i][j][k])
            for g_ind, g in enumerate(assigned_genes[i][j][k]):
                for iter in range(dp_scores.shape[0]):
                    score = dp_scores[iter, i, j, k]
                    if np.argsort(score)[-1].item() != g:
                        continue
                    if score.max() < dot_product_threshold:
                        continue
                    if dp_scores_assigned_ijk[g_ind] > 0:
                        continue
                    dp_scores_assigned_ijk[g_ind] = float(score.max())
                    all_assigned_genes.add(g)
                    break
            dp_scores_assigned[i][j][k] = dp_scores_assigned_ijk
        self.method = method
        self.spot_no = spot_no
        self.z_planes = z_planes
        self.image_shape = image_shape
        self.selected_gene = spot_gene_no
        all_assigned_genes.add(self.selected_gene)
        coefficients = coefficients.transpose((3, 0, 1, 2))
        self.coefficients = coefficients
        self.dp_scores_assigned = dp_scores_assigned
        self.iteration_counts = iteration_counts
        self.assigned_genes = assigned_genes
        scores = scores_torch.score_coefficient_image(torch.from_numpy(coefficients), torch.from_numpy(mean_spot))[
            :, image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2
        ]
        self.scores = scores.numpy()

        self.fig, self.axes = plt.subplots(2, len(z_planes) + 1, height_ratios=(6, 1))
        self.fig.set_figwidth(14)
        self.fig.set_figheight(8)
        for ax in self.axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        for c in range(self.axes.shape[1]):
            ax: plt.Axes = self.axes[1, c]
            ax.spines.clear()

        self.gene_slider = Slider(
            self.axes[1, 0],
            label="Gene",
            valmin=min(all_assigned_genes),
            valmax=max(all_assigned_genes),
            valstep=list(all_assigned_genes),
            valinit=self.selected_gene,
        )
        self.gene_slider.on_changed(self.gene_selected_updated)
        self.button_colour = "red"
        self.button_colour_press = "green"
        self.coef_button = Button(self.axes[1, 1], "Coefficients", hovercolor="0.275")
        self.dp_score_button = Button(self.axes[1, 2], "Dot Product Scores", hovercolor="0.275")
        self.iter_count_button = Button(self.axes[1, 3], "Iteration Counts", hovercolor="0.275")
        self.coef_button.on_clicked(self.pressed_coef_button)
        self.dp_score_button.on_clicked(self.pressed_dp_score_button)
        self.iter_count_button.on_clicked(self.pressed_iter_button)
        self.pressed_coef_button()

        if show:
            self.fig.show()

    def draw_data(self) -> None:
        title = "OMP Coefficients end_msg"
        if self.selected_button == self.Options.DP_SCORES:
            title = "OMP Dot Product Scores end_msg"
        elif self.selected_button == self.Options.ITERATIONS:
            title = "OMP Iteration Count"
        score_str = "{:.3f}".format(self.scores[self.selected_gene])
        title = title.replace("end_msg", f"near {self.method.capitalize()} Spot {self.spot_no}, Score {score_str}")
        self.fig.suptitle(title)
        data = []
        for k in range(len(self.z_planes)):
            if self.selected_button == self.Options.COEFFICIENTS:
                z_data = self.coefficients[self.selected_gene, :, :, k]
                abs_max = self.coefficients.max()
            elif self.selected_button == self.Options.DP_SCORES:
                z_data = np.zeros_like(self.coefficients[self.selected_gene, :, :, k])
                for i, j in itertools.product(range(z_data.shape[0]), range(z_data.shape[1])):
                    if self.selected_gene not in self.assigned_genes[i][j][k]:
                        continue
                    g_index = self.assigned_genes[i][j][k].index(self.selected_gene)
                    z_data[i, j] = self.dp_scores_assigned[i][j][k][g_index]
            elif self.selected_button == self.Options.ITERATIONS:
                z_data = self.iteration_counts[:, :, k]
            else:
                raise ValueError(f"Unknown option: {self.selected_button}")
            data.append(z_data)
        if self.selected_button != self.Options.COEFFICIENTS:
            abs_max = np.max(np.abs(data.copy()))
        cmap = mpl.cm.seismic
        norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        for k, z_plane in enumerate(self.z_planes):
            ax: plt.Axes = self.axes[0, k]
            ax.clear()
            # Two perpendicular lines to help see where the central pixel is.
            ax.hlines(
                self.image_shape[0] / 2 - 0.5,
                -1,
                self.image_shape[1] + 1,
                colors="green",
                linewidth=0.3,
            )
            ax.vlines(
                self.image_shape[1] / 2 - 0.5,
                -1,
                self.image_shape[0] + 1,
                colors="green",
                linewidth=0.3,
            )
            if z_plane == 0:
                ax.set_title(f"Central Plane")
            else:
                ax.set_title(f"{'+' if z_plane > 0 else '-'}{abs(z_plane)}")
            im = ax.imshow(data[k], cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        ax_cbar: plt.Axes = self.axes[0, -1]
        ax_cbar.clear()
        self.fig.colorbar(im, cax=ax_cbar)
        self.fig.canvas.draw_idle()

    def gene_selected_updated(self, _=None) -> None:
        self.selected_gene = int(self.gene_slider.val)
        if self.selected_button == self.Options.ITERATIONS:
            return
        self.draw_data()

    def pressed_coef_button(self, _=None) -> None:
        self.selected_button = self.Options.COEFFICIENTS
        self.draw_data()
        self.coef_button.label.set_color(self.button_colour_press)
        self.dp_score_button.label.set_color(self.button_colour)
        self.iter_count_button.label.set_color(self.button_colour)

    def pressed_dp_score_button(self, _=None) -> None:
        self.selected_button = self.Options.DP_SCORES
        self.draw_data()
        self.coef_button.label.set_color(self.button_colour)
        self.dp_score_button.label.set_color(self.button_colour_press)
        self.iter_count_button.label.set_color(self.button_colour)

    def pressed_iter_button(self, _=None) -> None:
        self.selected_button = self.Options.ITERATIONS
        self.draw_data()
        self.coef_button.label.set_color(self.button_colour)
        self.dp_score_button.label.set_color(self.button_colour)
        self.iter_count_button.label.set_color(self.button_colour_press)
