from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from .. import log
from ..omp import base as omp_base
from ..setup.notebook import Notebook
from ..setup.notebook_page import NotebookPage


def get_spot_intensity(spot_colors: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.

    Args:
        spot_colors (`[n_spots x n_rounds x n_channels] ndarray[float]`: spot colors normalised to equalise intensities
            between channels (and rounds).

    Returns:
        `[n_spots] ndarray[float]`: index `s` is the intensity of spot `s`.

    Notes:
        Logic is that we expect spots that are genes to have at least one large intensity value in each round
        so high spot intensity is more indicative of a gene.
    """
    if (spot_colors <= -15_000).sum() > 0:
        log.warn(f"Found spot colors <= -15000")
    # Max over all channels, then median over all rounds
    return np.median(np.max(np.abs(spot_colors), axis=2), axis=1)


def get_intensity_thresh(nb: Notebook) -> float:
    """
    Gets threshold for intensity from parameters in `config file` or Notebook.

    Args:
        nb: Notebook containing at least the `call_spots` page.

    Returns:
        intensity threshold
    """
    if nb.has_page("thresholds"):
        intensity_thresh = nb.thresholds.intensity
    else:
        intensity_thresh = nb.call_spots.abs_intensity_percentile[50]
    return intensity_thresh


def quality_threshold(
    nb: Notebook, method: str = "omp", intensity_thresh: float = 0, score_thresh: float = 0
) -> np.ndarray:
    """
    Indicates which spots pass both the score and intensity quality thresholding.

    Args:
        nb: Notebook containing at least the `ref_spots` page.
        method: `'ref'` or `'omp'` or 'prob' indicating which spots to consider.
        intensity_thresh: Intensity threshold for spots included.
        score_thresh: Score threshold for spots included.

    Returns:
        `bool [n_spots]` indicating which spots pass quality thresholding.

    """
    if method.lower() != "omp" and method.lower() != "anchor" and method.lower() != "prob":
        raise ValueError(f"method must be 'omp', 'prob', or 'anchor' but {method} given.")
    method_omp = method.lower() == "omp"
    method_anchor = method.lower() == "anchor"
    method_prob = method.lower() == "prob"

    # No intensity threshold is available for omp.
    if method_omp:
        score, _ = omp_base.get_all_scores(nb.basic_info, nb.omp)
    elif method_anchor:
        score = nb.call_spots.dot_product_gene_score
    elif method_prob:
        score = np.max(nb.call_spots.gene_probabilities, axis=1)
    if method_omp:
        intensity = np.ones_like(score)
    else:
        intensity = nb.call_spots.intensity
    qual_ok = np.array([score > score_thresh, intensity > intensity_thresh]).all(axis=0)
    return qual_ok
