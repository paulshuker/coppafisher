import colorsys
from typing import Any, Literal

import matplotlib
import numpy as np
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasHeadless
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# A headless (Agg) backend version of MplCanvas, this is required for unit testing the gene legend.
class MplCanvasHeadless(FigureCanvasHeadless):
    def __init__(self, _=None):
        fig = Figure()
        fig.set_size_inches(5, 4)
        self.axes = fig.subplots(1, 1)
        super(MplCanvasHeadless, self).__init__(fig)


class MplCanvas(FigureCanvas):
    def __init__(self, _=None):
        fig = Figure()
        fig.set_size_inches(5, 4)
        self.axes = fig.subplots(1, 1)
        super(MplCanvas, self).__init__(fig)


class Legend:
    canvas: FigureCanvasBase | None

    # Minimum width / height for a grid cell for a single gene in the gene legend.
    # Increasing this gives each gene a smaller height but a wider cell for the text.
    _minimum_cell_aspect_ratio: float = 2.0
    _order_by_options: tuple[str, ...] = ("row", "colour", "cell_type")

    def __init__(self) -> None:
        pass

    def create_gene_legend(self, genes: tuple, order_by: Literal["row"] | Literal["colour"] | Literal["cell_type"]):
        assert order_by in self._order_by_options

        self.canvas = MplCanvasHeadless() if matplotlib.get_backend() == "Agg" else MplCanvas()

        active_genes = [gene for gene in genes if gene.active]
        # active_count = len(active_genes)
        categorised_genes: dict[str, list[Any]] = {}
        match order_by:
            case "row":
                categorised_genes[""] = active_genes
            case "cell_type":
                cell_types = np.sort(np.unique([gene.cell_type for gene in active_genes])).tolist()
                for category in cell_types:
                    categorised_genes[category] = [gene for gene in active_genes if gene.cell_type == category]
            case "colour":
                categorised_genes[""] = [
                    active_genes[ind]
                    for ind in self._hue_argsort(np.unique([gene.colour for gene in active_genes], axis=0))
                ]
            case _:
                raise ValueError(f"Unknown order_by: {order_by}")
        # category_count = len(categorised_genes)

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
