from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from napari.layers import Shapes

from .subplot import Subplot


class ExportTool2D(Subplot):
    DESCRIPTION: str = """Go back to the Viewer window, press P (or R or Shift + P) to start
 building a polygon shape, press Enter when finished. Build other shapes if needed. Press the
 Export button below to export the spots once done."""

    shapes_layer: Shapes
    on_click: Callable[["ExportTool2D"], None] | None = None
    on_close: Callable[["ExportTool2D"], None] | None = None

    def __init__(self, show: bool) -> None:
        """
        Build the user interface for the 2D exporter tool.

        This contains a button for the user to click when they are done building their polygon shape.
        """
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.clear()
        self.fig.set_size_inches(6.5, 2.5)

        self.fig.suptitle("2D Export Tool", fontsize=14, fontweight="bold")
        self.fig.text(0.5, 0.8, self.DESCRIPTION, fontsize=9, ha="center", va="top")

        self.button_ax = self.fig.add_axes([0.2, 0.1, 0.6, 0.4])  # [left, bottom, width, height]
        self.button = Button(self.button_ax, "Export", color="#89291e", hovercolor="#db3725")
        self.button.label.fontsize = 100

        self.fig.canvas.mpl_connect("close_event", self.on_figure_closed)
        self.button.on_clicked(self.on_export_button_clicked)

        if show:
            self.fig.show()

    def on_export_button_clicked(self, _=None) -> None:
        if self.on_click is None:
            return

        self.on_click(self)

    def on_figure_closed(self, _=None) -> None:
        if self.on_close is None:
            return

        self.on_close(self)
