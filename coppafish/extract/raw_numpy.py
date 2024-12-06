import os
from typing import Tuple

import dask
import nd2
import numpy as np

from ..setup.notebook_page import NotebookPage
from .raw_reader import RawReader


class NumpyReader(RawReader):
    """
    Reader for raw numpy files.

    For example, Robominnie (the pipeline integration tester) uses raw numpy files as input.
    """

    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        super().read(nbp_basic, nbp_file, tile, round, channels)
