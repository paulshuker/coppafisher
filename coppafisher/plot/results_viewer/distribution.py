import math as maths
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.axes import Axes
from matplotlib.widgets import Button, RangeSlider

from ...results.base import MethodData
from .subplot import Subplot


class ViewScoreIntensityDistributions(Subplot):
    # The number of bins within a region of 0 to 1 in score or intensity values.
    _bin_density = 100

    def __init__(self, method: str, scores: np.ndarray[float], intensities: np.ndarray[float], show: bool = True):
        """
        Show how the spot scores and intensities distribute for the specific method given.

        Args:
            method (str): the selected method. Must be 'prob', 'anchor', or 'omp'.
            scores (`(n_spots) ndarray[float]`): the scores for the method to view. These should be between 0 and 1.
            intensities (`(n_spots) ndarray[float]`): the intensities for the method's spots.
            show (bool, optional): show the plot after building it. Default: true.
        """
        assert method in ("prob", "anchor", "omp")
        assert scores.ndim == 1
        assert scores.size > 0
        assert (scores >= 0).all()
        assert intensities.ndim == 1
        assert intensities.size > 0
        assert scores.size == intensities.size

        self.log_counts = True
        self.fig, self.axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [5, 1]})
        self.fig.suptitle(f"{method.capitalize()} Score/Intensity Distributions")
        self.score_range = [0.0, max(scores.max().item(), 1.0)]
        score_bins = maths.ceil((self.score_range[1] - self.score_range[0]) * self._bin_density)
        self.intensity_range = [0.0, max(intensities.max().item(), 1.0)]
        intensity_bins = maths.ceil((self.intensity_range[1] - self.intensity_range[0]) * self._bin_density)
        H, self.xedges, self.yedges = np.histogram2d(
            scores, intensities, bins=(score_bins, intensity_bins), range=[self.score_range, self.intensity_range]
        )
        self.H = H.T
        # Logarithmic scaling to help see low count features in the plot.
        self.H_log = np.log1p(self.H)
        self.draw_data()

        # Check box to toggle logarithmic counting.
        self.button_colour_not_pressed = "red"
        self.button_colour_pressed = "green"
        rax = plt.axes([0.1, 0.8, 0.1, 0.05])  # Position of the checkbox
        self.checkbox = Button(rax, "Log Counts", hovercolor="0.275")
        self.checkbox.on_clicked(self.toggle_log_counts)
        self.checkbox.label.set_color(self.button_colour_pressed)

        if show:
            self.fig.show()

    def draw_data(self) -> None:
        ax: plt.Axes = self.axes[0]
        ax.clear()
        ax.grid(False)
        ax.set_xlim(*self.score_range)
        ax.set_xlabel("Score")
        ax.set_ylim(*self.intensity_range)
        ax.set_ylabel("Intensity")
        data = self.H
        if self.log_counts:
            data = self.H_log
        # We allow non-square pixels so the subplot can correctly fill the plot space even if intensity has many more
        # bins than score for example.
        im = ax.imshow(
            data,
            interpolation="nearest",
            origin="lower",
            extent=(self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]),
            aspect="auto",
        )
        cbar_ax: plt.Axes = self.axes[1]
        cbar_ax.clear()
        label = "Count"
        if self.log_counts:
            label = "Ln (1 + count)"
        self.fig.colorbar(im, cax=cbar_ax, label=label)
        self.fig.canvas.draw()

    def toggle_log_counts(self, _=None) -> None:
        self.log_counts = not self.log_counts
        if self.log_counts:
            self.checkbox.label.set_color(self.button_colour_pressed)
        else:
            self.checkbox.label.set_color(self.button_colour_not_pressed)
        self.draw_data()


class ViewSpotScoreAndSimilarityDensityPlots(Subplot):
    POINT_COUNT: int = 100

    def __init__(
        self,
        method: str,
        spot_data: MethodData,
        bled_codes: np.ndarray,
        starting_score_threshold: Tuple[float, float],
        starting_intensity_threshold: Tuple[float, float],
        show: bool = True,
    ):
        """
        Plot and show a density plot of spot scores and similarity scores.

        Args:
            method (str): the gene calling method. Can be 'prob', 'anchor', or 'omp'.
            spot_data (MethodData): the spot data for the gene calling method.
            bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray[float32]`): the final gene bled codes.
            starting_score_threshold (tuple of two floats): the starting spot score thresholds for spot selection.
            starting_intensity_threshold (tuple of two floats): the starting spot intensity thresholds for spot selection.
            show (bool, optional): show plot after building it. Default: true.
        """
        assert method in ("prob", "anchor", "omp")
        assert isinstance(spot_data, MethodData)
        assert type(bled_codes) is np.ndarray
        assert bled_codes.size > 0
        assert bled_codes.ndim == 3
        assert type(starting_score_threshold) is tuple
        assert type(starting_intensity_threshold) is tuple
        assert len(starting_score_threshold) == 2
        assert len(starting_intensity_threshold) == 2

        self.method = method
        self.spot_data = spot_data
        self.bled_codes = bled_codes.astype(np.float32)
        self.score_threshold = [None] * 2
        self.intensity_threshold = [None] * 2

        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 10))
        self.fig.subplots_adjust(bottom=0.25)
        self.fig.suptitle(f"{self.method.upper()} Gaussian KDE of Spot Scores")

        self.score_slider_ax = self.fig.add_axes([0.20, 0.1, 0.60, 0.03])
        self.score_slider = RangeSlider(
            self.score_slider_ax, "Spot Score Threshold", 0, 1, valinit=starting_score_threshold
        )

        self.intensity_slider_ax = self.fig.add_axes([0.20, 0.03, 0.60, 0.03])
        self.intensity_slider = RangeSlider(
            self.intensity_slider_ax,
            "Spot Intensity Threshold",
            0,
            max(1, starting_intensity_threshold[1]),
            valinit=starting_intensity_threshold,
        )

        self._update_score_keep(starting_score_threshold, redraw=False)
        self._update_intensity_keep(starting_intensity_threshold, redraw=False)
        self._update_data()

        self.score_slider.on_changed(self._update_score_keep)
        self.intensity_slider.on_changed(self._update_intensity_keep)
        self.fig.legend()
        if show:
            self.fig.show()

    def _update_data(self) -> None:
        assert type(self.ax) is Axes
        self.ax.clear()

        keep = self.keep_scores & self.keep_intensities
        spot_scores = self.spot_data.score[keep].astype(np.float32)
        xs = np.arange(self.POINT_COUNT + 1, dtype=np.float32)
        xs /= self.POINT_COUNT
        # Compute similarity scores.
        n_spots = keep.sum()
        if not n_spots:
            return
        spot_bled_codes = self.bled_codes[self.spot_data.gene_no[keep]]
        spot_colours = self.spot_data.colours[keep]
        spot_colours = spot_colours.clip(0, None).reshape((n_spots, -1))
        spot_bled_codes = spot_bled_codes.reshape((n_spots, -1))
        similarity_scores = (spot_colours * spot_bled_codes).sum(1) / (
            np.linalg.norm(spot_colours, axis=1) * np.linalg.norm(spot_bled_codes, axis=1)
        )

        assert similarity_scores.size == spot_scores.size
        assert (similarity_scores >= 0).all()
        assert (similarity_scores <= 1).all()

        spot_score_density = scipy.stats.gaussian_kde(spot_scores)
        spot_score_density = spot_score_density(xs)
        self.ax.set_xlabel("Score")
        self.ax.set_ylabel("Density")
        self.ax.set_xlim(0, 1)
        self.ax.plot(xs, spot_score_density, linewidth=1, color="darkviolet", label=f"{self.method} score")
        self.ax.fill_between(xs, spot_score_density, alpha=0.3, color="violet")

        spot_similarity_density = scipy.stats.gaussian_kde(similarity_scores)
        spot_similarity_density = spot_similarity_density(xs)
        assert type(self.ax) is Axes
        self.ax.set_xlabel("Score")
        self.ax.set_ylabel("Density")
        self.ax.set_xlim(0, 1)
        self.ax.plot(xs, spot_similarity_density, linewidth=1, color="darkgreen", label=f"{self.method} similarity")
        self.ax.fill_between(xs, spot_similarity_density, alpha=0.3, color="limegreen")

        self.fig.canvas.draw()

    def _update_score_keep(self, threshold: Tuple[float, float], redraw: bool = True) -> None:
        self.score_threshold[0] = threshold[0]
        self.score_threshold[1] = threshold[1]
        self.keep_scores = (self.spot_data.score >= self.score_threshold[0]) & (
            self.spot_data.score <= self.score_threshold[1]
        )
        if redraw:
            self._update_data()

    def _update_intensity_keep(self, threshold: Tuple[float, float], redraw: bool = True) -> None:
        self.intensity_threshold[0] = threshold[0]
        self.intensity_threshold[1] = threshold[1]
        self.keep_intensities = (self.spot_data.intensity >= self.intensity_threshold[0]) & (
            self.spot_data.intensity <= self.intensity_threshold[1]
        )
        if redraw:
            self._update_data()
