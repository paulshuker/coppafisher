import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ...omp.coefs import CoefficientSolverOMP
from ...setup import config
from ...setup.notebook_page import NotebookPage
from ..results_viewer.subplot import Subplot


class ViewOMPColourSum(Subplot):
    def __init__(
        self,
        nbp_basic: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage | None,
        method: str,
        local_yxz: np.ndarray[int],
        spot_tile: int,
        spot_colour: np.ndarray[float],
        show: bool = True,
    ) -> None:
        """
        Show the weighted gene bled codes that are summed together by OMP to try and produce the total pixel's colour
        after completing all iterations. It also displays the residual colour left behind that is unaccounted for.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_filter (NotebookPage): `filter` notebook page.
            nbp_register (NotebookPage): `register` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage or none): `omp` notebook page or none.
            method (str): gene calling method.
            local_yxz (`(3) ndarray[int]`): the pixel position relative to its tile's bottom-left corner.
            spot_tile (int-like): tile index the pixel is on.
            spot_colour (`(n_rounds_use x n_channels_use) ndarray[float]`): the spot's colour.
            show (bool, optional): display the plot once built. False is useful when unit testing. Default: true.
        """
        n_rounds_use = len(nbp_basic.use_rounds)
        n_channels_use = len(nbp_basic.use_channels)
        min_intensity = config.get_default_for("omp", "minimum_intensity")
        alpha = config.get_default_for("omp", "alpha")
        beta = config.get_default_for("omp", "beta")
        max_genes = config.get_default_for("omp", "max_genes")
        dot_product_threshold = config.get_default_for("omp", "dot_product_threshold")
        self.gene_names = nbp_call_spots.gene_names
        if nbp_omp is not None:
            min_intensity = float(nbp_omp.associated_configs["omp"]["minimum_intensity"])
            alpha = float(nbp_omp.associated_configs["omp"]["alpha"])
            beta = float(nbp_omp.associated_configs["omp"]["beta"])
            max_genes = int(nbp_omp.associated_configs["omp"]["max_genes"])
            dot_product_threshold = float(nbp_omp.associated_configs["omp"]["dot_product_threshold"])

        self.colour = spot_colour.copy().astype(np.float32)
        self.colour *= nbp_call_spots.colour_norm_factor[spot_tile].astype(np.float32)
        omp_solver = CoefficientSolverOMP()
        bled_codes = nbp_call_spots.bled_codes.astype(np.float32)
        bg_bled_codes = omp_solver.create_background_bled_codes(n_rounds_use, n_channels_use)
        coefficients, gene_weights = omp_solver.solve(
            pixel_colours=self.colour[np.newaxis],
            bled_codes=bled_codes,
            background_codes=bg_bled_codes,
            maximum_iterations=max_genes,
            dot_product_threshold=dot_product_threshold,
            minimum_intensity=min_intensity,
            alpha=alpha,
            beta=beta,
            return_all_weights=True,
        )
        self.coefficient = coefficients[0]
        self.gene_weight = gene_weights[0]
        self.assigned_genes: np.ndarray[int] = (~np.isnan(self.gene_weight)).nonzero()[0]
        self.gene_weight = self.gene_weight[self.assigned_genes]
        self.coefficient = self.coefficient[self.assigned_genes]
        n_iterations = self.assigned_genes.size

        column_count = max(2, self.assigned_genes.size)
        self.fig, self.axes = plt.subplots(2, column_count, figsize=(column_count * 2.7, 5.5))
        if n_iterations == 0:
            return
        self.assigned_bled_codes = nbp_call_spots.bled_codes[self.assigned_genes]
        # Weight the bled codes.
        self.assigned_bled_codes *= self.gene_weight[:, np.newaxis, np.newaxis]
        self.residual_colour = self.colour - self.assigned_bled_codes.sum(0)
        abs_max = np.abs(self.assigned_bled_codes).max()
        abs_max = np.max([abs_max, np.abs(self.colour).max()])
        abs_max = np.max([abs_max, np.abs(self.residual_colour).max()]).item()

        self.cmap = mpl.cm.seismic
        self.norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        self.draw_data()
        self.fig.suptitle(f"{method.capitalize()} spot at {tuple(local_yxz)} OMP colour sum")
        self.fig.tight_layout()
        if show:
            self.fig.show()

    def draw_data(self) -> None:
        for ax in self.axes.ravel():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for i, g in enumerate(self.assigned_genes):
            w_str = "{:.3f}".format(self.gene_weight[i])
            c_str = "{:.3f}".format(self.coefficient[i])
            self.axes[0, i].set_title(f"{g}: {self.gene_names[g]}\nweight: {w_str}\ncoefficient: {c_str}")
            self.axes[0, i].imshow(self.assigned_bled_codes[i].T, cmap=self.cmap, norm=self.norm)

        self.axes[1, -1].set_title(f"Total colour")
        self.axes[1, -1].imshow(self.colour.T, cmap=self.cmap, norm=self.norm)
        self.axes[1, -2].set_title(f"Residual colour")
        self.axes[1, -2].imshow(self.residual_colour.T, cmap=self.cmap, norm=self.norm)

        self.fig.canvas.draw_idle()
