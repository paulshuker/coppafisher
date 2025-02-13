import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ...pipeline import basic_info
from ...setup.config import Config

TILE_PAD: float = 0.05


def view_tile_indexing_grid(config_file_path: str, show: bool = True) -> None:
    """
    View a grid of every tile for the given dataset. Each tile has a unique coppafisher tile index given to it This way
    users can decide what use_tiles to set in the config file for a dataset run.

    Args:
        config_file_path (str): the path to the dataset's config file.
        show (bool, optional): show the plot after creating it. Default: true.
    """
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"No config file at {config_file_path}")

    config = Config()
    config.load(config_file_path, post_check=False)
    nbp_basic_info = basic_info.set_basic_info(config)

    tilepos_min_yx: list[int] = nbp_basic_info.tilepos_yx.min(0).tolist()
    tilepos_max_yx: list[int] = nbp_basic_info.tilepos_yx.max(0).tolist()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for y in range(tilepos_min_yx[0], tilepos_max_yx[0] + 1):
        for x in range(tilepos_min_yx[1], tilepos_max_yx[1] + 1):
            tilepos = np.array([y, x], int)
            if not (nbp_basic_info.tilepos_yx == tilepos[np.newaxis]).all(1).any():
                continue
            tile_index = np.flatnonzero((nbp_basic_info.tilepos_yx == tilepos[np.newaxis]).all(1))[0]

            tile_anchor = tuple((tilepos[[1, 0]].astype(float) + TILE_PAD).tolist())
            tile_length = 1.0 - 2 * TILE_PAD

            new_tile = mpl.patches.Rectangle(tile_anchor, tile_length, tile_length, color="orange", fc="none", lw=2)
            ax.add_patch(new_tile)
            ax.annotate(str(tile_index.item()), tuple(tilepos[[1, 0]].astype(float) + 0.5))

    ax.set_ylim(min(tilepos_min_yx), max(tilepos_max_yx) + 1)
    ax.set_xlim(min(tilepos_min_yx), max(tilepos_max_yx) + 1)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_ylabel("Y")
    ax.set_xlabel("X")

    fig.suptitle("Coppafisher Tile Indexing\nNote: In the Viewer, the y axis is flipped!")
    fig.tight_layout()

    if show:
        fig.canvas.draw_idle()
        plt.show()
