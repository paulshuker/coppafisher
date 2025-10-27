from typing import Callable, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from ...plot.results_viewer.subplot import Subplot


class ManualThreshold(Subplot):
    # Called when the score threshold lower and/or upper bound is changed.
    _score_threshold_changed: Callable[[Tuple[float, float]], None]
    # Called when the intensity threshold lower and/or upper bound is changed.
    _intensity_threshold_changed: Callable[[Tuple[float, float]], None]

    _current_score: Tuple[float, float]
    _current_intensity: Tuple[float, float]

    def __init__(
        self,
        score_threshold_callback: Callable[[Tuple[float, float]], None],
        intensity_threshold_callback: Callable[[Tuple[float, float]], None],
        current_score_threshold: Tuple[float, float],
        current_intensity_threshold: Tuple[float, float],
        show: bool = True,
    ) -> None:
        """
        Set specific values of score and intensity thresholds for the Viewer.

        Args:
            score_threshold_callback (callable): this function is called when the score threshold is changed. It is
                given a tuple of two floats for the minimum and maximum score thresholds. The function must return none.
            intensity_threshold_callback (callable): this function is called when the intensity threshold is changed. It
                is given a tuple of two floats for the minimum and maximum score thresholds. The function must return
                none.
            show (bool, optional): show the plot after building it. Default: true.
        """
        self._score_threshold_changed = score_threshold_callback
        self._intensity_threshold_changed = intensity_threshold_callback
        self._current_score = current_score_threshold
        self._current_intensity = current_intensity_threshold

        self.fig, self.axes = plt.subplots(4, 1, figsize=(5, 5))
        self.fig.set_layout_engine("constrained")

        self._min_score_box = TextBox(
            self.axes[0], "Minimum Score Threshold", self._current_score[0], color="gray", hovercolor="gray"
        )
        self._max_score_box = TextBox(
            self.axes[1], "Maximum Score Threshold", self._current_score[1], color="gray", hovercolor="gray"
        )
        self._min_intensity_box = TextBox(
            self.axes[2], "Minimum Intensity Threshold", self._current_intensity[0], color="gray", hovercolor="gray"
        )
        self._max_intensity_box = TextBox(
            self.axes[3], "Maximum Intensity Threshold", self._current_intensity[1], color="gray", hovercolor="gray"
        )

        self._min_score_box.on_submit(self._on_min_score_changed)
        self._max_score_box.on_submit(self._on_max_score_changed)
        self._min_intensity_box.on_submit(self._on_min_intensity_changed)
        self._max_intensity_box.on_submit(self._on_max_intensity_changed)

        if show:
            self.fig.show()

    def _on_min_score_changed(self, value: str) -> None:
        value = float(value)
        self._score_threshold_changed((value, self._current_score[1]))
        self._current_score = (value, self._current_score[1])

    def _on_max_score_changed(self, value: str) -> None:
        value = float(value)
        self._score_threshold_changed((self._current_score[0], value))
        self._current_score = (self._current_score[0], value)

    def _on_min_intensity_changed(self, value: str) -> None:
        value = float(value)
        self._intensity_threshold_changed((value, self._current_intensity[1]))
        self._current_intensity = (value, self._current_intensity[1])

    def _on_max_intensity_changed(self, value: str) -> None:
        value = float(value)
        self._intensity_threshold_changed((self._current_intensity[0], value))
        self._current_intensity = (self._current_intensity[0], value)
