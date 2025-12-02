from typing import Literal

import numpy as np
import tifffile

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class TifReader(RawReader):
    """
    Reader for raw tif files.
    """

    def read(
        self,
        nbp_basic: NotebookPage,
        nbp_file: NotebookPage,
        tile: int,
        round: int,
        channels: list[int],
        z_planes: list[int] | Literal["all"] | None = None,
    ) -> np.ndarray:
        """
        Similar to ND2, expect a separate directory for each round.

        Every directory has one array for each tile with shape (n_tiles * n_channels * n_z_planes, n_y_pixels,
        n_x_pixels). The first axis is flattened such that such that the first n_channels are tile 0 and z plane 0 on
        each channel, then the next n_channels are tile 0 and z plane 1 on each channel, ... then after n_z z-planes the
        next n_channels are tile 1 and z plane 0 on each channel etc...
        """
        super().read(nbp_basic, nbp_file, tile, round, channels)

        file_path = super().get_round_file_path(nbp_file, round)
        if not file_path.endswith(nbp_file.raw_extension):
            file_path += nbp_file.raw_extension

        # NOTE: n_tiles in the basic_info page must be the number of total tiles in the raw files, NOT necessarily the
        # same as len(nbp_basic.use_tiles). Same for n_channels too.
        n_tiles = nbp_basic.n_tiles
        n_channels = nbp_basic.n_channels

        result = []

        # Has shape (n_tiles * n_channels * im_z, im_y, im_x).
        all_images = tifffile.imread(file_path)
        all_images = all_images.astype(np.uint16, casting="safe")

        n_total_z = all_images.shape[0] // (n_channels * n_tiles)

        for c in channels:
            start_index = c + tile * n_channels * n_total_z
            combined_slice = slice(start_index, start_index + n_channels * n_total_z, n_channels)
            c_image = all_images[combined_slice]
            if z_planes is None:
                c_image = c_image[nbp_basic.use_z]
            elif z_planes != "all":
                c_image = c_image[z_planes]
            # ZYX -> YXZ.
            c_image = c_image.swapaxes(0, 1).swapaxes(1, 2)

            result.append(c_image)

        result = np.array(result, np.uint16)

        return result
