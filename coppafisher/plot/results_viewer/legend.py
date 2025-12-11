import colorsys
import math as maths
from collections import OrderedDict
from typing import Any, Literal, Tuple

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasHeadless
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ...utils import markers


# A headless (Agg) backend version of MplCanvas, this is required for unit testing the gene legend.
class MplCanvasHeadless(FigureCanvasHeadless):
    ax: Axes

    def __init__(self, _=None):
        fig = Figure()
        fig.set_size_inches(5, 4)
        self.ax = fig.subplots(1, 1)
        super(MplCanvasHeadless, self).__init__(fig)


class MplCanvas(FigureCanvas):
    ax: Axes

    def __init__(self, _=None):
        fig = Figure()
        fig.set_size_inches(5, 4)
        self.ax = fig.subplots(1, 1)
        super(MplCanvas, self).__init__(fig)


class Legend:
    canvas: MplCanvas | MplCanvasHeadless | None

    # Minimum width / height for a grid cell for a single gene in the gene legend.
    # Increasing this trades a smaller grid height for a wider cell for the text.
    _minimum_cell_aspect_ratio: float = 1.0
    # As a fraction of the cell width from the left edge.
    _marker_padding: float = 0.1
    # As a fraction of the cell width from the right edge.
    _marker_text_padding: float = 0.05

    _order_by_options: tuple[str, ...] = ("row", "colour", "cell_type")

    def get_order_by_options(self) -> tuple[str, ...]:
        return self._order_by_options

    order_by_options: tuple[str, ...] = property(get_order_by_options)
    _selected_opacity: float = 1.0
    _unselected_opacity: float = 0.25
    _napari_to_mpl_marker: dict[str, str] = {
        "cross": "+",
        "disc": "o",
        "square": "s",
        "triangle_up": "^",
        "triangle_down": "v",
        "hbar": "_",
        "vbar": "|",
        "star": "*",
        "arrow": ">",
        "ring": "o",  # A ring is the same as disc but the face colour is set to None with only an edge colour.
        "clobber": r"$\clubsuit$",
        "x": "x",
        "diamond": "d",
        "tailed_arrow": r"$\rightarrow$",
    }

    def __init__(self) -> None:
        pass

    def create_gene_legend(self, genes: tuple, order_by: Literal["row"] | Literal["colour"] | Literal["cell_type"]):
        assert order_by in self._order_by_options

        self.categorised_genes: OrderedDict[str, list[Any]] = OrderedDict()
        match order_by:
            case "row":
                self.categorised_genes[""] = list(genes)
            case "cell_type":
                cell_types = np.sort(np.unique([gene.cell_type for gene in genes])).tolist()
                for category in cell_types:
                    self.categorised_genes[category] = [gene for gene in genes if gene.cell_type == category]
            case "colour":
                self.categorised_genes[""] = [
                    genes[ind] for ind in self._hue_argsort(np.unique([gene.colour for gene in genes], axis=0))
                ]
            case _:
                raise ValueError(f"Unknown order_by: {order_by}")
        self.category_count = len(self.categorised_genes)

        self._plot_index_to_gene_index: list[int] = []
        for category in self.categorised_genes.keys():
            self._plot_index_to_gene_index += [genes.index(gene) for gene in self.categorised_genes[category]]

        self.canvas = MplCanvasHeadless() if matplotlib.get_backend() == "Agg" else MplCanvas()
        self._draw()
        self.canvas.ax.spines["top"].set_visible(False)
        self.canvas.ax.spines["right"].set_visible(False)
        self.canvas.ax.spines["bottom"].set_visible(False)
        self.canvas.ax.spines["left"].set_visible(False)
        self.current_active_genes = [True for _ in genes]
        self.update_selected_legend_genes(self.current_active_genes)
        self.canvas.mpl_connect("resize_event", self._on_resize_event)

    def update_selected_legend_genes(self, active_genes: list[bool]) -> None:
        """
        Update which genes are currently selected in the viewer. A selected gene is given a high opacity.
        """
        active_genes_sorted = [active_genes[i] for i in self._plot_index_to_gene_index]
        for axes, is_active in zip(self.scatter_axes, active_genes_sorted, strict=True):
            axes[0].set_alpha(self._selected_opacity if is_active else self._unselected_opacity)
            axes[1].set_font(FontProperties(weight="medium" if is_active else "normal"))
        self.canvas.draw_idle()

    def _draw(self) -> None:
        self.canvas.ax.clear()
        self.canvas.ax.set_xticks([], [])
        self.canvas.ax.set_yticks([], [])

        fig_width, fig_height = self.canvas.figure.get_size_inches().tolist()

        # Calculate the highest cells per row possible.
        cells_per_row = 1
        row_count = self._calculate_row_count(cells_per_row)
        cell_width = fig_width
        cell_height = fig_width / row_count
        while True:
            next_cell_width = fig_width / (cells_per_row + 1)
            next_cell_height = fig_height / self._calculate_row_count(cells_per_row + 1)
            if (next_cell_width / next_cell_height) < self._minimum_cell_aspect_ratio:
                break
            cell_width = next_cell_width
            cell_height = next_cell_height
            cells_per_row += 1
            row_count = self._calculate_row_count(cells_per_row)

        s = 10

        # Draw the gene legend inside a rectangle of size figure width by figure height.
        self.scatter_axes = []
        row = -1
        for category_name, genes in self.categorised_genes.items():
            row += 1
            if self._include_category_names():
                position: Tuple[float, float] = (fig_width / 2, row * cell_height + 0.5 * cell_height)
                self.canvas.ax.annotate(category_name, position, ha="center", va="center")
                row += 1

            col = 0
            for gene in genes:
                x = col * cell_width
                y = row * cell_height + 0.5 * cell_height
                scatter_kwargs = {"s": s, "color": gene.colour}
                scatter_kwargs["marker"] = markers.align_marker(
                    self._napari_to_mpl_marker[gene.symbol_napari], halign="left"
                )
                if gene.symbol_napari == "ring":
                    scatter_kwargs["facecolor"] = "none"
                    scatter_kwargs["edgecolor"] = gene.colour
                scatter_ax = self.canvas.ax.scatter(x + self._marker_padding * cell_width, y, **scatter_kwargs)
                text_ax = self.canvas.ax.annotate(
                    gene.name,
                    (x + (1 - self._marker_text_padding) * cell_width, y),
                    ha="right",
                    va="center",
                    weight="medium",
                )
                self.scatter_axes.append((scatter_ax, text_ax))

                col += 1
                if col >= cells_per_row:
                    col = 0
                    row += 1

        self.canvas.ax.set_xlim(0, fig_width)
        self.canvas.ax.set_ylim(0, fig_height)
        self.canvas.draw_idle()

    def _on_resize_event(self, _=None) -> None:
        self._draw()
        self.update_selected_legend_genes(self.current_active_genes)

    def _calculate_row_count(self, cells_per_row: int) -> int:
        row_count = self.category_count if self._include_category_names() else 0
        for genes in self.categorised_genes.values():
            row_count += maths.ceil(len(genes) / cells_per_row)

        return row_count

    def _include_category_names(self) -> bool:
        return not (self.category_count == 1 and "" in self.categorised_genes)

    def _hue_argsort(self, colours: np.ndarray[float]) -> np.ndarray[int]:
        """
        The given colours are sorted based on their hues.

        Args:
            colours (`(n_colours x 3) ndarray[float]`): each colours RGB value, ranging from 0 to 1.

        Returns:
            (`(n_colours) ndarray[int]`): Indices to sort the colours by hue from lowest to highest.
        """
        assert type(colours) is np.ndarray
        assert colours.ndim == 2
        assert colours.shape[0] > 0
        assert colours.shape[1] == 3

        hues = []
        for colour in colours:
            assert colour.shape == (3,)
            hues.append(colorsys.rgb_to_hsv(colour[0].item(), colour[1].item(), colour[2].item())[0])
        return np.argsort(hues)
