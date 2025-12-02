from typing import Literal

import dask.array
import numpy as np

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class NumpyReader(RawReader):
    """
    Reader for raw numpy files.

    For example, Robominnie (the pipeline integration tester) uses raw numpy files as input.
    Read ND2 files for the given channels.
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

        Each directory has one ndarray for each tile with shape (1, n_z_planes, n_channels, n_y_pixels, n_x_pixels).
        """
        super().read(nbp_basic, nbp_file, tile, round, channels)

        file_path = super().get_round_file_path(nbp_file, round)

        # Has shape (n_channels, im_y, im_x, im_z).
        tile_round_images: np.ndarray = dask.array.from_npy_stack(file_path)[tile].compute()

        tile_round_images = tile_round_images[channels]

        if z_planes is None:
            tile_round_images = tile_round_images[..., nbp_basic.use_z]
        elif z_planes != "all":
            tile_round_images = tile_round_images[..., z_planes]

        return tile_round_images
