import itertools
from typing import List, Optional

import napari
import numpy as np

from ..setup.notebook import Notebook


def view_filtered_images(
    nb: Notebook,
    tiles: Optional[List[int]] = None,
    rounds: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
) -> None:
    """
    View the filtered images.

    Args:
        - nb (Notebook): notebook.
        - tiles (Optional[List[int]], optional): tiles to view. Default: all tiles.
        - rounds (Optional[List[int]], optional): rounds to view. Default: all rounds.
        - channels (Optional[List[int]], optional): channels to view. Default: all channels.
    """
    assert nb.has_page("filter"), "Filter must be run first"

    if tiles is None:
        tiles = nb.basic_info.use_tiles.copy()
    if rounds is None:
        rounds = nb.basic_info.use_rounds.copy()
    if channels is None:
        channels = nb.basic_info.use_channels.copy()

    viewer = napari.Viewer(title="Coppafish filtered images")

    for t, r, c in itertools.product(tiles, rounds, channels):
        image_trc: np.ndarray = nb.filter.images[t, r, c].astype(np.float32)
        # y, x, z -> z, y, x.
        image_trc = image_trc.swapaxes(1, 2).swapaxes(0, 1)
        viewer.add_image(image_trc, name=f"Filtered {t=}, {r=}, {c=}")

    napari.run()
