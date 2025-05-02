from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from napari.layers import Shapes

from .subplot import Subplot


class ExportTool2D(Subplot):
    DESCRIPTION: str = """Go back to the Viewer window, press P (or R or Shift + P) to start
 building a polygon shape, press Enter when finished. Build other shapes if needed. Press the
 Export button below to export the spots once done."""

    shapes_layer: Shapes
    on_export_clicked: Callable[["ExportTool2D"], None] | None = None
    on_dilate_clicked: Callable[["ExportTool2D"], None] | None = None
    on_closed: Callable[["ExportTool2D"], None] | None = None

    _current_scale_factor: float = 1.0

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

        self.button_ax = self.fig.add_axes([0.2, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
        self.export_button = Button(self.button_ax, "Export", color="#89291e", hovercolor="#db3725")
        self.export_button.label.fontsize = 100

        self.dilate_button_ax = self.fig.add_axes([0.5, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
        self.dilate_button = Button(self.dilate_button_ax, "Dilate", color="#89291e", hovercolor="#db3725")
        self.dilate_button.label.fontsize = 100

        self.scale_factor_ax = self.fig.add_axes([0.5, 0.3, 0.4, 0.2])
        self.scale_factor_box = TextBox(self.scale_factor_ax, "Scale Factor", "1")
        self.scale_factor_box.label.fontsize = 70
        self.scale_factor_box.on_submit(self.on_scale_factor_submitted)

        self.fig.canvas.mpl_connect("close_event", self.on_figure_closed)
        self.export_button.on_clicked(self.on_export_button_clicked)
        self.dilate_button.on_clicked(self.on_dilate_button_clicked)

        if show:
            self.fig.show()

    def get_current_scale_factor(self) -> float:
        return self._current_scale_factor

    def on_export_button_clicked(self, _=None) -> None:
        if self.on_export_clicked is None:
            return

        self.on_export_clicked(self)

    def on_dilate_button_clicked(self, _=None) -> None:
        if self.on_dilate_clicked is None:
            return

        self.on_dilate_clicked(self)

    def on_scale_factor_submitted(self, _=None) -> None:
        try:
            new_scale_factor = float(self.scale_factor_box.text)
        except ValueError:
            print(f"Invalid scale factor '{self.scale_factor_box.text}'")
            return

        self._current_scale_factor = new_scale_factor

    def on_figure_closed(self, _=None) -> None:
        if self.on_closed is None:
            return

        self.on_closed(self)
