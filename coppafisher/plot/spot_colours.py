import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import CheckButtons, Slider

from ..spot_colours import base as spot_colours_base
from .results_viewer.subplot import Subplot


class ViewSpotColourRegion(Subplot):
    def __init__(
        self,
        spot_no: int,
        spot_score: float,
        spot_local_yxz: np.ndarray,
        spot_tile: int,
        gene_index: int,
        gene_name: str,
        filter_images: zarr.Array,
        flow: zarr.Array,
        affine: zarr.Array,
        colour_norm_factor: np.ndarray,
        use_rounds: list[int],
        use_channels: list[int],
        method: str,
        show: bool = True,
    ):
        """
        Build a grid of pyplot imshows for each channel and round showing the registered, filtered image intensity in a
        2D neighbourhood around the spot in Y and X. Out of bounds pixels are not plotted.

        Args:
            spot_no (int): index of spot in the notebook (number between 0 and n_spots - 1).
            spot_score (float): score of spot gene assignment.
            spot_local_yxz (`(3) ndarray[int]`): the spot's local y, x, and z position relative to the spot's tile's
                bottom-left corner.
            spot_tile (int): index of tile spot is on.
            gene_index (int): the spot's gene's index.
            gene_name (str): the spot's gene's name.
            filter_images (`(n_tiles x n_rounds x n_channels) zarray[float16]`): the filtered images.
            flow (`(n_tiles x n_rounds x 3 x tile_sz x tile_sz x len(use_z)) zarray[float16]`): the register optical
                flow results.
            affine (`(n_tiles x n_rounds x n_channels x 4 x 3) ndarray[float32]`): the register affine corrections.
            colour_norm_factor (`(n_tiles x n_rounds x n_channels_use) ndarray[float32]`): normalisation factor for
                each tile, round, and channel that is applied to colours.
            use_rounds (list of int): sequencing rounds.
            use_channels (list of int): sequencing channels.
            method (str): spot's method. Can be 'anchor', 'omp' or 'prob'.
            show (bool, optional): show the plot after creating. Turn off for unit testing. Default: true.
        """
        assert method.lower() in ["anchor", "omp", "prob"], "method must be 'anchor', 'omp' or 'prob'"
        self.local_region_shape_yxz = (27, 27, 5)
        assert all([shape % 2 == 1 for shape in self.local_region_shape_yxz]), "Must be odd numbers only"

        self.n_rounds = len(use_rounds)
        self.n_channels = len(use_channels)
        Y = np.linspace(0, self.local_region_shape_yxz[0] - 1, self.local_region_shape_yxz[0])
        Y -= self.local_region_shape_yxz[0] // 2
        Y += spot_local_yxz[0]
        X = np.linspace(0, self.local_region_shape_yxz[1] - 1, self.local_region_shape_yxz[1])
        X -= self.local_region_shape_yxz[1] // 2
        X += spot_local_yxz[1]
        Z = np.linspace(0, self.local_region_shape_yxz[2] - 1, self.local_region_shape_yxz[2])
        Z -= self.local_region_shape_yxz[2] // 2
        Z += spot_local_yxz[2]
        yxz_local_region = np.array(np.meshgrid(Y, X, Z, indexing="ij"), int)
        # Becomes shape (n_pixels, 3).
        yxz_local_region = yxz_local_region.reshape((3, -1), order="F").T
        colours = spot_colours_base.get_spot_colours_new(
            yxz_local_region,
            filter_images,
            flow,
            affine,
            int(spot_tile),
            use_rounds,
            use_channels,
            out_of_bounds_value=0,
        )
        self.colours = colours.reshape(self.local_region_shape_yxz + (self.n_rounds, self.n_channels), order="F")
        self.colours = self.colours.transpose((1, 0, 2, 3, 4))

        self.spot_tile = spot_tile
        self.colour_norm_factor = colour_norm_factor.astype(np.float32).copy()

        self.fig, self.axes = plt.subplots(self.n_channels, self.n_rounds, squeeze=False, sharex=True, sharey=True)
        self.fig.suptitle(
            f"Spot Colour Region, z=({spot_local_yxz.item(2) - self.local_region_shape_yxz[2] // 2},"
            + f"{spot_local_yxz.item(2) + self.local_region_shape_yxz[2] // 2})\n{method.capitalize()} index {spot_no},"
            + f"gene {gene_index} {gene_name}, score: {'{:.2f}'.format(spot_score)}"
        )
        self.fig.supxlabel("Round")
        self.fig.supylabel("Channel")

        # Colour bar on right.
        cbar_pos = [0.90, 0.075, 0.04, 0.85]  # left, bottom, width, height
        self.cbar_ax = self.fig.add_axes(cbar_pos)

        self.norm_button_ax = self.fig.add_axes([0.02, 0.02, 0.08, 0.04])
        self.norm_button = CheckButtons(self.norm_button_ax, ["Normalisation"])
        self.norm_button.set_active(0, True)
        self.norm_button.on_clicked(self._on_gui_changed)

        self.log_button_ax = self.fig.add_axes([0.12, 0.02, 0.08, 0.04])
        self.log_button = CheckButtons(self.log_button_ax, ["Log Scale"])
        self.log_button.set_active(0, False)
        self.log_button.on_clicked(self._on_gui_changed)

        self.z_slider_ax = self.fig.add_axes([0.28, 0.02, 0.4, 0.04])
        self.z_slider = Slider(
            self.z_slider_ax,
            "Z Plane",
            0,
            self.local_region_shape_yxz[2] - 1,
            valinit=self.local_region_shape_yxz[2] // 2,
            valstep=1,
        )
        self.z_slider.on_changed(self._on_gui_changed)

        self.show = show
        if self.show:
            self.fig.show()

        self.spot_local_yxz = spot_local_yxz

        self._draw_axes()

    def _draw_axes(self) -> None:
        # Spot colour images.
        abs_max = 0
        plot_colours = np.zeros(self.local_region_shape_yxz[:2] + (self.n_rounds, self.n_channels), float)
        for c in range(self.n_channels):
            for r in range(self.n_rounds):
                rc_colours = self.colours[:, :, int(self.z_slider.val), r, c].copy()
                if self.norm_button.get_status()[0]:
                    rc_colours *= self.colour_norm_factor[self.spot_tile, r, c]
                plot_colours[:, :, r, c] = rc_colours

        # plot_colours -= np.percentile(plot_colours, 25, axis=3, keepdims=True)
        # plot_colours /= np.linalg.norm(plot_colours, axis=(2, 3), keepdims=True)

        self.norm_button.set_label_props({"c": "green" if self.norm_button.get_status()[0] else "red"})

        abs_max = np.abs(plot_colours).max()
        self.cmap = mpl.cm.seismic
        if self.log_button.get_status()[0]:
            self.norm = mpl.colors.SymLogNorm(0.05, vmin=-abs_max, vmax=+abs_max)
            self.log_button.set_label_props({"c": "green"})
        else:
            self.norm = mpl.colors.Normalize(vmin=-abs_max, vmax=+abs_max)
            self.log_button.set_label_props({"c": "red"})

        for c in range(self.n_channels):
            for r in range(self.n_rounds):
                ax: plt.Axes = self.axes[c, r]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                im = ax.imshow(plot_colours[:, :, r, c].T, cmap=self.cmap, norm=self.norm)
                # Two perpendicular lines to help see where the central pixel is.
                ax.hlines(
                    self.local_region_shape_yxz[0] / 2,
                    -1,
                    self.local_region_shape_yxz[0] + 1,
                    colors="green",
                    linewidth=0.3,
                )
                ax.vlines(
                    self.local_region_shape_yxz[1] / 2,
                    -1,
                    self.local_region_shape_yxz[1] + 1,
                    colors="green",
                    linewidth=0.3,
                )
        self.cbar_ax.clear()
        plt.colorbar(im, cax=self.cbar_ax, pad=0.1, orientation="vertical", label="Intensity")

        if self.show:
            self.fig.canvas.draw()

    def _on_gui_changed(self, _: str | None = None) -> None:
        self._draw_axes()
