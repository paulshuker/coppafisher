import os
from typing import Any

import nd2
import numpy as np

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class Nd2Reader(RawReader):
    """
    Reader for raw, ND2 files.
    """

    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        super().read(nbp_basic, nbp_file, tile, round, channels)

        tile_raw = super().get_tile_raw_index(tile, nbp_basic.tilepos_yx_nd2, nbp_basic.tilepos_yx)

        if nbp_basic.use_anchor:
            # Always have anchor after imaging rounds.
            round_files = nbp_file.round + [nbp_file.anchor]
        else:
            round_files = nbp_file.round

        round_file = os.path.join(nbp_file.input_dir, round_files[round])
        file_path = round_file + nbp_file.raw_extension

        with nd2.ND2File(file_path) as images:
            images = images.to_dask()

        images = images[tile_raw]

        # Convert the dask array into an ndarray as all channels for one tile/round can fit in memory.
        # This is the fastest way because the dask array is not chunked over channels like previously thought.
        images = images.compute()

        images = images[channels]

        # Put z index to end.
        # czyx -> cyxz where c is the channels index I think?
        images = images.swapaxes(1, 3)

        assert images.ndim == 4

        return images

    def _get_all_metadata(file_path: str) -> dict[str, Any]:
        """
        Get all metadata from the given nd2 file.

        Args:
            file_path (str): path to nd2 file.

        Returns:
            (dict[str, any]): all_metadata. Dictionary containing all found metadata for given ND2 file.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Could not find ND2 file at {file_path}")

        with nd2.ND2File(file_path) as images:
            metadata = images.unstructured_metadata()
        return dict(metadata)
