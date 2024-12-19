from typing import List, Optional

import napari
import numpy as np

from ..setup.notebook import Notebook
from ..spot_colours import base


def view_intensity_images(
    nb: Notebook,
    tiles: Optional[List[int]] = None,
    share_contrast_limits: bool = True,
) -> None:
    """
    View the computed intensity for the given tile(s). The intensity is defined as min_r(max_c(abs(colours))) where
    colours have the call spot colour norm factor applied.

    Args:
        nb (Notebook): notebook.
        tiles (list of int, optional): tiles to view. Default: first tile.
        share_contrast_limits (bool, optional): use the same contrast limits for all filtered images shown. Default:
            true.
    """
    assert nb.has_page("register"), "Register must be run first"

    if tiles is None:
        tiles = nb.basic_info.use_tiles[:1]

    factor = nb.call_spots.colour_norm_factor.astype(np.float32)

    min = 1e20
    max = -1e20
    images = []
    names = []
    for t in tiles:
        intensities = base.get_spot_colours_new_safe(
            nb.basic_info,
            image=nb.filter.images,
            flow=nb.register.flow,
            affine=nb.register.icp_correction,
            tile=t,
            use_rounds=nb.basic_info.use_rounds,
            use_channels=nb.basic_info.use_channels,
            out_of_bounds_value=0,
        )
        intensities *= factor[[t]]
        intensities = np.abs(intensities).max(2).min(1)
        intensities = intensities.reshape(
            (nb.basic_info.tile_sz, nb.basic_info.tile_sz, len(nb.basic_info.use_z)), order="F"
        )

        # y, x, z -> z, y, x.
        intensities = intensities.swapaxes(1, 2).swapaxes(0, 1)
        images.append(intensities)
        names.append(f"Intensities tile={t}")
        image_min = intensities.min()
        image_max = intensities.max()
        if image_min < min:
            min = image_min
        if image_max > max:
            max = image_max

    viewer = napari.Viewer(title="Coppafish intensities")
    limits = None
    for image, name in zip(images, names):
        if share_contrast_limits:
            limits = [min, max]
        viewer.add_image(image, name=name, rgb=False, contrast_limits=limits)

    napari.run()
