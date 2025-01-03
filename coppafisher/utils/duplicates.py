import numpy as np
import scipy
import torch


def get_tile_centres(tile_sz: int, n_z_planes: int, tile_origins: np.ndarray[float]) -> torch.Tensor:
    """
    Find all tile centres based on the tile origins and their size.

    Args:
        tile_sz (int): tile size in pixels along the x and y direction.
        n_z_planes (int): the number of z planes in each tile.
        tile_origins (`(n_tiles x 3) ndarray[float]`): all tile origins in y, x, and z in a shared coordinate system.
            nan is populated for any tiles not used in the pipeline.

    Returns:
        `(n_tiles x 3) tensor[float32]`: tile_centres. Every tile's centre in global coordinates. Tile centres are sent
            to near infinity for nan tile origins.
    """
    assert type(tile_sz) is int
    assert type(n_z_planes) is int
    assert type(tile_origins) is np.ndarray
    assert tile_sz > 0
    assert n_z_planes > 0
    assert tile_origins.ndim == 2
    assert tile_origins.shape[0] > 0
    assert tile_origins.shape[1] == 3

    tile_shape: tuple[int, int, int] = tile_sz, tile_sz, n_z_planes
    tile_centres = tile_origins.copy().astype(np.float32)
    # Invalid tiles are sent far away to avoid mistaken duplicate spot detection.
    tile_centres[np.isnan(tile_centres)] = 1e20
    tile_centres = torch.asarray(tile_centres)
    tile_origins = tile_centres.detach().clone()
    tile_centres += torch.asarray(tile_shape).float() / 2

    return tile_centres


def is_duplicate_spot(yxz_global_positions: torch.Tensor, tile_number: int, tile_centres: torch.Tensor) -> torch.Tensor:
    """
    Checks what spot positions are duplicates. A duplicate is defined as any spot that is closer to a different tile
    origin than the one it is assigned to.

    Args:
        yxz_global_positions (`(n_points x 3) tensor[int]`): y, x, and z global positions for each spot.
        tile_number (int): the tile index for all spot positions.
        tile_centres (`(n_tiles x 3) tensor[float]`): each tile's centre in global coordinates.

    Returns:
        (`(n_points) tensor[bool]`): is_duplicate. True for each duplicate spot.
    """
    assert type(yxz_global_positions) is torch.Tensor
    n_points = yxz_global_positions.shape[0]
    assert n_points > 0, "Require at least one spot"
    assert yxz_global_positions.shape[1] == 3
    assert type(tile_number) is int
    assert type(tile_centres) is torch.Tensor
    assert tile_centres.shape[1] == 3
    assert tile_number >= 0 and tile_number < tile_centres.shape[0]

    kdtree = scipy.spatial.KDTree(tile_centres.numpy())
    # Find the nearest tile origin for each spot position.
    # If this is not the tile number assigned to the spot, it is a duplicate.
    closest_tile_numbers = kdtree.query(yxz_global_positions.numpy(), k=1)[1]
    closest_tile_numbers = np.array(closest_tile_numbers)
    closest_tile_numbers = torch.asarray(closest_tile_numbers)
    is_duplicate = closest_tile_numbers != tile_number

    return is_duplicate
