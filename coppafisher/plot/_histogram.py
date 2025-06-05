from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


@overload
def build_histograms(
    plot_values: list[np.ndarray],
    plot_subtitles: list[str],
    title: str,
    bin_counts: int,
    bin_range: tuple[int | float, int | float],
    log: bool,
    vertical_lines: list[list[int | float]],
    vertical_line_labels: list[list[str]],
) -> tuple[Figure, list[plt.Axes]]: ...


@overload
def build_histograms(
    plot_values: list[np.ndarray],
    plot_subtitles: list[str],
    title: str,
    bin_counts: int,
    bin_range: tuple[int | float, int | float],
    log: bool,
) -> tuple[Figure, list[plt.Axes]]: ...


def build_histograms(
    plot_values: list[np.ndarray],
    plot_subtitles: list[str],
    title: str,
    bin_counts: int,
    bin_range: tuple[int | float, int | float],
    log: bool,
    vertical_lines: list[list[int | float]] | None = None,
    vertical_line_labels: list[list[str]] | None = None,
) -> tuple[Figure, list[plt.Axes]]:
    """
    Build a grid of histograms.

    Each histogram is a subplot. Vertical lines are drawn on the subplots if given.

    Args:
        plot_values (list of `(n_bins) ndarray`): the values to bin for each subplot.
        plot_subtitles (list of str): each histogram's subtitle.
        title (str): figure title.
        bin_counts (list of int): the number of bins. This is the same for every subplot.
        bin_range (tuple of two numbers): the range of data to plot. This is the same for every subplot.
        log (bool): set the histogram axis to a log scale.
        vertical_lines (list of list of int/float, optional): vertical line positions for each subplot. Default: no
            lines.
        vertical_line_labels (list of list of string, optional): vertical line labels shown in the legend for each
            subplot. Default: no labels.

    Returns a tuple containing:
        - (pyplot.Figure): fig. The histogram figure.
        - (list of Axes): axes. The histogram axes. Some can be empty to maintain a rectangular screen shape.
    """
    assert type(plot_values) is list
    assert all([type(v) is np.ndarray for v in plot_values])
    assert all([type(v) is str for v in plot_subtitles])
    assert len(plot_values) == len(plot_subtitles)
    assert type(title) is str
    assert type(bin_counts) is int
    assert type(bin_range) is tuple
    assert len(bin_range) == 2
    if vertical_lines is not None:
        assert type(vertical_lines) is list
        assert type(vertical_line_labels) is list
        assert len(vertical_lines) == len(vertical_line_labels)

    X_ASPECT = 4
    Y_ASPECT = 3
    LINE_STYLES = ("solid", "dashed")
    LINE_COLOURS = ("yellow", "blue", "green", "aqua")

    subplot_count = len(plot_values)
    x_subplot_count = X_ASPECT
    y_subplot_count = Y_ASPECT
    while subplot_count > (x_subplot_count * y_subplot_count):
        x_subplot_count *= 2
        y_subplot_count *= 2

    x_min = 999_999
    x_max = -1
    plot_max = -1

    fig, axes = plt.subplots(y_subplot_count, x_subplot_count, squeeze=False)
    axes: list[plt.Axes] = axes.ravel().tolist()
    for i, (values, subtitle) in enumerate(zip(plot_values, plot_subtitles, strict=True)):
        ax = axes[i]
        ax.set_title(subtitle, fontdict={"fontsize": "x-small"})
        ax_counts, _, _ = ax.hist(values, bins=bin_counts, range=bin_range, log=log, color="red")
        x_min = min(ax.get_xlim()[0], x_min)
        x_max = max(ax.get_xlim()[1], x_max)
        plot_max = max(ax_counts.max().item(), plot_max)
        if vertical_lines is None:
            continue

        line_styles = [LINE_STYLES[j % len(LINE_STYLES)] for j in range(len(vertical_lines[i]))]
        line_labels = [vertical_line_labels[i][j] for j in range(len(vertical_lines[i]))]
        line_colours = [LINE_COLOURS[j % len(LINE_COLOURS)] for j in range(len(vertical_lines[i]))]
        for line_position, line_colour, line_style, line_label in zip(
            vertical_lines[i], line_colours, line_styles, line_labels, strict=True
        ):
            # ax.vlines(line_position, colors=line_colours, linestyles=line_style, label=line_label)
            ax.axvline(line_position, color=line_colour, linestyle=line_style, label=line_label)
            ax.legend(loc="upper right", fontsize="xx-small")
    for ax in axes[:subplot_count]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.1 if log else 0, plot_max)

    for ax in axes[subplot_count:]:
        ax.spines[["left", "right", "bottom", "top"]].set_visible(False)
        ax.set_xlabel("")
        ax.set_xticks([], [])
        ax.set_ylabel("")
        ax.set_yticks([], [])

    fig.suptitle(title)

    return fig, axes
