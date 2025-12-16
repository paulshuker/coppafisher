import numpy as np
from matplotlib import markers
from matplotlib.path import Path


def align_marker(marker, halign: str | float = "center", valign: str | float = "middle") -> np.ndarray:
    """
    Create marker with specified alignment.

    Args:
        marker: a valid marker specification. See mpl.markers
        halign (str or float): 'left', 'center', or 'right' specifies the horizontal alignment of the marker. *float*
            values specify the alignment in units of the markersize/2 (0 is 'center', -1 is 'right', 1 is 'left').
        valign (str or float): 'top', 'middle', or 'bottom' specifies the vertical alignment of the marker. *float*
            values specify the alignment in units of the markersize/2 (0 is 'middle', -1 is 'top', 1 is 'bottom').

    Returns:
        (`(N x 2) ndarray`): marker_array. A Nx2 array that specifies the marker path relative to the plot target
            point at (0, 0).
    """

    if isinstance(halign, str):
        halign = {
            "right": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "left": 1.0,
        }[halign]

    if isinstance(valign, str):
        valign = {
            "top": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "bottom": 1.0,
        }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)
