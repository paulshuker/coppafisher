import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import nd2
import numpy as np

from ...pipeline import basic_info
from ...setup import tile_details
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
            ax.annotate(str(tile_index.item()), tuple(tilepos[[1, 0]].astype(float) + 0.5), ha="center", va="center")

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


def plot_coords_nd2_coppafish(preseq_file: str, channel: int, reverse: bool = False):
    """
    Plot the tile locations with their given nd2 and coppafisher tile indices.

    Args:
        preseq_file (str): presequence file location.
        channel (int): the channel index to display.
        reverse (bool, optional): reverse the tile positions. Default: false.
    """
    # Load presequencing tiles
    f = nd2.ND2File(preseq_file)
    d = f.to_dask()
    tiles = np.asarray(d[:, d.shape[1] // 2, channel])
    f.close()
    # Load transforms
    true_nd2pos, true_pos = coords_nd2_coppafish(preseq_file, reverse=reverse)
    cf_map = map_coppafish_to_nd2(preseq_file, reverse)
    nd2_map = map_nd2_to_coppafish(preseq_file, reverse)
    vmax = np.quantile(tiles, 0.98)
    vmin = np.quantile(tiles, 0.01)
    # Plot in nd2 space
    plt.figure()
    for i in range(0, tiles.shape[0]):
        index = true_nd2pos[i // true_nd2pos.shape[1], i % true_nd2pos.shape[1]]
        plt.subplot(true_nd2pos.shape[0], true_nd2pos.shape[1], i + 1)
        plt.imshow(tiles[index], vmax=vmax, vmin=vmin)
        plt.text(0, 0, f"ND2 tile {index}\nCoppafish tile {nd2_map[index]}", fontsize="small")
        plt.axis("off")
    plt.tight_layout()
    plt.suptitle("ND2 tile numbers")
    # Plot in coppafish space
    plt.figure()
    for i in range(0, tiles.shape[0]):
        index = true_pos[i // true_pos.shape[1], i % true_pos.shape[1]]
        plt.subplot(true_pos.shape[0], true_pos.shape[1], i + 1)
        plt.imshow(np.rot90(tiles[cf_map[index]]), vmax=vmax, vmin=vmin)
        plt.text(0, 0, f"Coppafish tile {index}\nND2 tile {cf_map[index]}", fontsize="small")
        plt.axis("off")
    plt.tight_layout()
    plt.suptitle("Coppafish tile numbers")
    plt.show()


def coords_nd2_coppafish(preseq_file, reverse):
    # Get the coordinates of each tile using the same method that coppafish uses
    f = nd2.ND2File(preseq_file)
    _ = np.round(np.asarray([np.asarray(p.stagePositionUm) for p in f.experiment[0].parameters.points]) / 100)
    stage_coords = np.asarray([p.stagePositionUm[:2] for p in f.experiment[0].parameters.points])
    reverse_x = -1 if reverse else 1
    reverse_y = -1 if reverse else 1
    stage_coords = stage_coords * [reverse_x, reverse_y]
    cf_nd2pos, cf_pos = tile_details.get_tilepos(
        (stage_coords - np.min(stage_coords, 0)) / f.metadata.channels[0].volume.axesCalibration[0], f.sizes["X"], 0.1
    )
    f.close()
    # Swap x and y coordinates to correct for the fact that nd2 coordinates are
    # in yx instead of xy format
    true_nd2pos = np.flip(cf_nd2pos, axis=1)
    # Correct for the y flip of coppafish coordinates
    true_pos = cf_pos.copy()
    true_pos[:, 0] = np.max(true_pos[:, 0]) - true_pos[:, 0]
    # Create a grid of positions for each
    rows = np.max(true_pos[:, 0]) + 1
    cols = np.max(true_pos[:, 1]) + 1
    true_pos_mat = np.zeros((rows, cols), dtype="int") - 1
    true_nd2pos_mat = np.zeros((cols, rows), dtype="int") - 1
    for i, p in enumerate(true_pos):
        true_pos_mat[tuple(p)] = i
    for i, p in enumerate(true_nd2pos):
        true_nd2pos_mat[tuple(p)] = i
    # if reverse:
    #    return true_nd2pos_mat[:,::-1],true_pos_mat[:,::-1]
    return true_nd2pos_mat[::-1], true_pos_mat[::-1]


def map_nd2_to_coppafish(preseq_file, reverse):
    """
    1D array to convert nd2 tiles to coppafish tiles.

    coppafish_tile_number = map_nd2_to_coppafish(preseq_file)[nd2_tile_number]
    """
    # Get coordinates of tiles in nd2s and in coppafish
    true_nd2pos_mat, true_pos_mat = coords_nd2_coppafish(preseq_file, reverse=reverse)
    # Since coppafish coordinates are rotated 90deg compared to the nd2s,
    # unrotate them
    true_pos_mat = true_pos_mat[::-1]
    true_nd2pos_mat = np.rot90(true_nd2pos_mat)[::-1]
    # Create a mapping from the nd2 tile number to coppafish tile number
    mapping = []
    for i in range(0, len(true_nd2pos_mat.flatten())):
        mapping.append([true_nd2pos_mat.flatten()[i], true_pos_mat.flatten()[i]])
    mapping = np.asarray(mapping)
    return mapping[np.argsort(mapping[:, 0]), 1]


def map_coppafish_to_nd2(preseq_file, reverse):
    """
    1D array to convert coppafish tiles to nd2 tiles.

    nd2_tile_number = map_coppafish_to_nd2(preseq_file)[coppafish_tile_number]
    """
    return np.argsort(map_nd2_to_coppafish(preseq_file, reverse))
