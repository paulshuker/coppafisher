import matplotlib.pyplot as plt
import napari
import numpy as np
import tqdm

from coppafisher.setup.config import Config

from ..setup.notebook import Notebook
from ..spot_colours import base
from ..utils import intensity
from . import _histogram


def view_intensity_histogram(nb: Notebook, tiles: list[int] | None = None, show: bool = True) -> None:
    """
    View the intensity pixel histograms.

    Each tile is a histogram subplot. Vertical lines indicate significant intensity values like the median, mean, 5th
    percentile, and the computed intensity threshold for OMP.

    Args:
        nb (Notebook): the notebook.
        tiles (list of int, optional): tiles to plot. Default: all tiles.
        show (bool, optional): show the plot after drawing. Default: true.
    """
    assert type(nb) is Notebook
    if tiles is None:
        tiles = nb.basic_info.use_tiles
    assert type(tiles) is list
    assert all(type(t) is int for t in tiles)

    if not nb.has_page("register"):
        raise ValueError("Register must be complete")
    if not nb.has_page("call_spots"):
        raise ValueError("Call spots must be complete")

    mid_z: int = nb.basic_info.use_tiles[len(nb.basic_info.use_tiles) // 2]
    tile_shape = (nb.basic_info.tile_sz, nb.basic_info.tile_sz, 1)
    colour_norm_factor = nb.call_spots.colour_norm_factor.astype(np.float32)
    min_intensity_percentile = float(Config.get_default_for("omp", "minimum_intensity_percentile"))
    min_intensity_multiplier = float(Config.get_default_for("omp", "minimum_intensity_multiplier"))

    yxz_mid_z = [np.linspace(0, tile_shape[i] - 1, tile_shape[i]) for i in range(3)]
    yxz_mid_z = np.array(np.meshgrid(*yxz_mid_z, indexing="ij")).astype(np.int16).reshape((3, -1), order="F").T
    yxz_mid_z[:, 2] = mid_z

    intensity_min: float = 0
    intensity_max: float = -1
    all_intensities = []
    vertical_lines = []
    vertical_line_labels = []

    for t in tqdm.tqdm(tiles, desc="Gathering intensity data", unit="tile"):
        spot_colour_kwargs = dict(
            image=nb.filter.images,
            flow=nb.register.flow,
            affine=nb.register.icp_correction,
            tile=t,
            use_rounds=nb.basic_info.use_rounds,
            use_channels=nb.basic_info.use_channels,
            output_dtype=np.float32,
            out_of_bounds_value=0,
        )
        mid_z_colours = base.get_spot_colours_new_safe(nb.basic_info, yxz_mid_z, **spot_colour_kwargs)
        mid_z_colours *= colour_norm_factor[[t]]
        t_intensities = intensity.compute_intensity(mid_z_colours).numpy()

        t_vertical_lines = []
        t_vertical_line_labels = []

        t_vertical_lines.append(t_intensities.mean().item())
        t_vertical_line_labels.append("Mean")

        t_vertical_lines.append(np.median(t_intensities).item())
        t_vertical_line_labels.append("Median")

        t_vertical_lines.append(np.percentile(t_intensities, min_intensity_percentile).item())
        t_vertical_line_labels.append(f"{min_intensity_percentile}th Percentile")

        t_vertical_lines.append(min_intensity_multiplier * t_vertical_lines[2])
        t_vertical_line_labels.append(
            f"{min_intensity_multiplier} " + r"\times" + f" {min_intensity_percentile}th Percentile"
        )

        vertical_lines.append(t_vertical_lines)
        vertical_line_labels.append(t_vertical_line_labels)

        intensity_max = max(t_intensities.max().item(), intensity_max)

        all_intensities.append(t_intensities)

    fig, _ = _histogram.build_histograms(
        all_intensities,
        [f"Tile {t}" for t in tiles],
        f"Pixel intensities at z={mid_z}",
        500,
        (intensity_min, intensity_max),
        log=True,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )
    fig.set_layout_engine("constrained")

    if show:
        plt.show()


def view_intensity_images(
    nb: Notebook,
    tiles: list[int] | None = None,
    z_planes: int | None = None,
    z_plane_start_from: int = 0,
) -> None:
    """
    View the computed intensity for the given tile(s) and the anchor image(s). The intensity is defined as
    min_r(max_c(abs(colours))) where colours have the call spots colour norm factor applied.

    Args:
        nb (Notebook): the notebook.
        tiles (list of int, optional): tiles to view. Default: first tile.
        z_planes (int, optional): the number of z planes to view, starting from `z_plane_start_from`. Default: 20.
        z_plane_start_from (int, optional): the lowest z plane to view. Default: 0.
    """
    if not nb.has_page("register"):
        raise ValueError("Register must be complete")
    if not nb.has_page("call_spots"):
        raise ValueError("Call spots must be complete")

    if tiles is None:
        tiles = nb.basic_info.use_tiles[:1]
    if z_planes is None:
        z_planes = 20
    z_planes = min(z_planes, max(nb.basic_info.use_z))

    tile_shape = (nb.basic_info.tile_sz, nb.basic_info.tile_sz, z_planes)
    yxz = [np.linspace(0, tile_shape[i] - 1, tile_shape[i]) for i in range(2)]
    yxz.append(np.linspace(z_plane_start_from, z_plane_start_from + z_planes - 1, z_planes))
    yxz = np.array(np.meshgrid(*yxz, indexing="ij")).astype(np.int16).reshape((3, -1), order="F").T

    factor = nb.call_spots.colour_norm_factor.astype(np.float32)

    images: list[np.np.ndarray] = []
    names: list[str] = []
    for t in tiles:
        intensities = base.get_spot_colours_new_safe(
            nb.basic_info,
            yxz=yxz,
            image=nb.filter.images,
            flow=nb.register.flow,
            affine=nb.register.icp_correction,
            tile=t,
            use_rounds=nb.basic_info.use_rounds,
            use_channels=nb.basic_info.use_channels,
            out_of_bounds_value=0,
        )
        intensities *= factor[[t]]
        intensities = intensity.compute_intensity(intensities).numpy()
        intensities = intensities.reshape((nb.basic_info.tile_sz, nb.basic_info.tile_sz, z_planes), order="F")

        images.append(intensities)
        names.append(f"Intensities tile={t}")

    # Add the anchor images as well.
    for t in tiles:
        anchor_image = nb.filter.images[
            t,
            nb.basic_info.anchor_round,
            nb.basic_info.anchor_channel,
            :,
            :,
            z_plane_start_from : z_plane_start_from + z_planes,
        ]
        images.append(anchor_image)
        names.append(f"Anchor tile={t}")

    # y, x, z -> z, y, x for napari.
    for i, image in enumerate(images):
        images[i] = image.swapaxes(1, 2).swapaxes(0, 1)

    viewer = napari.Viewer(title="Coppafisher intensities")
    for image, name in zip(images, names):
        viewer.add_image(image, name=name, rgb=False)

    napari.run()
