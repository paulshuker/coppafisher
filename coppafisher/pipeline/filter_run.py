import importlib.resources as importlib_resources
import math as maths
import os
from typing import Tuple

import joblib
import numpy as np
import psutil
import skimage
import tqdm
import zarr

from .. import log
from ..filter import radius_normalisation
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..utils import dict_io, dimensions, indexing, system, zarray

FILTER_DTYPE = np.float16
NINE_CHANNEL_DIR = "coppafisher.setup"
SEVEN_CHANNEL_DIR = "coppafisher.setup"
DAPI_CHANNEL_DIR = "coppafisher.setup"
SEVEN_CHANNEL_NAME = "seven_channel_normalisations.npz"
NINE_CHANNEL_NAME = "nine_channel_normalisations.npz"
DAPI_CHANNEL_NAME = "dapi_channel_normalisations.npz"


def run_filter(
    config: ConfigSection, nbp_file: NotebookPage, nbp_basic: NotebookPage
) -> Tuple[NotebookPage, NotebookPage]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (ConfigSection): config section obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns tuple containing:
        - NotebookPage: 'filter' notebook page.
        - NotebookPage: 'filter_debug' notebook page.

    Notes:
        - See `'filter'` and `'filter_debug'` sections of `notebook_page.py` file for description of variables.
    """
    filter_config = {config.name: config.to_dict()}
    nbp = NotebookPage("filter", filter_config)
    nbp_debug = NotebookPage("filter_debug", filter_config)

    config_path = os.path.join(nbp_file.output_dir, "filter_last_config.pkl")
    last_filter_config = dict_io.try_load_dict(config_path, filter_config.copy())
    assert type(last_filter_config) is dict
    config_unchanged = filter_config == last_filter_config
    dict_io.save_dict(filter_config, config_path)
    del filter_config, last_filter_config

    completed_indices_path = os.path.join(nbp_file.output_dir, "filter_completed_indices.pkl")
    if not config_unchanged:
        os.remove(completed_indices_path)
    completed_indices: dict[str, list[tuple[int, int, int]]] = dict_io.try_load_dict(completed_indices_path, {"a": []})

    log.debug("Filter started")

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_bad_trc=False,
    )

    max_ind = np.array(indices).max(0).tolist()
    shape = (max_ind[0] + 1, max_ind[1] + 1, max_ind[2] + 1, nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    chunks = (1, 1, 1, min(576, nbp_basic.tile_sz), min(576, nbp_basic.tile_sz), 5)
    images_path = os.path.join(nbp_file.output_dir, "filter_images.zarr")
    images_store = zarr.ZipStore(images_path, mode="w")
    images = zarr.open_array(
        images_store,
        "a",
        shape=shape,
        chunks=chunks,
        fill_value=np.nan,
        zarr_version=2,
        dtype=FILTER_DTYPE,
    )

    # Bad trc images are filled with zeros.
    for t, r, c in nbp_basic.bad_trc:
        images[t, r, c] = 0
        completed_indices["a"].append((t, r, c))
        dict_io.save_dict(completed_indices, completed_indices_path)

    if not os.path.isfile(nbp_file.psf):
        raise FileNotFoundError(f"Could not find the PSF at location {nbp_file.psf}")

    channel_radius_norm_filepath = config["channel_radius_normalisation_filepath"]
    channel_radius_norm = None
    dapi_radius_norm_filepath = config["dapi_radius_normalisation_filepath"]
    dapi_radius_norm = None

    if channel_radius_norm_filepath is None and nbp_basic.use_channels == [5, 9, 14, 15, 18, 23, 27]:
        channel_radius_norm_filepath = importlib_resources.files(SEVEN_CHANNEL_DIR).joinpath(SEVEN_CHANNEL_NAME)
    elif channel_radius_norm_filepath is None and nbp_basic.use_channels == [5, 9, 10, 14, 15, 18, 19, 23, 27]:
        channel_radius_norm_filepath = importlib_resources.files(NINE_CHANNEL_DIR).joinpath(NINE_CHANNEL_NAME)

    if (
        dapi_radius_norm_filepath is None
        and nbp_basic.dapi_channel == 0
        and nbp_basic.use_channels in ([5, 9, 14, 15, 18, 23, 27], [5, 9, 10, 14, 15, 18, 19, 23, 27])
    ):
        dapi_radius_norm_filepath = importlib_resources.files(DAPI_CHANNEL_DIR).joinpath(DAPI_CHANNEL_NAME)

    if channel_radius_norm_filepath is not None:
        channel_radius_norm = np.load(str(channel_radius_norm_filepath))["arr_0"]
        if channel_radius_norm.shape[0] != len(nbp_basic.use_channels):
            raise ValueError(
                f"Expected channel radius normalisation to have shape[0] == {len(nbp_basic.use_channels)}"
                + f", instead got {channel_radius_norm.shape[0]}"
            )
        radius_normalisation.validate_radius_normalisation(channel_radius_norm[0], nbp_basic.tile_sz)
    log.debug(f"Using channel radius normalisation: {channel_radius_norm is not None}")

    if dapi_radius_norm_filepath is not None:
        dapi_radius_norm = np.load(str(dapi_radius_norm_filepath))["arr_0"]
        radius_normalisation.validate_radius_normalisation(dapi_radius_norm, nbp_basic.tile_sz)
    log.debug(f"Using dapi radius normalisation: {dapi_radius_norm is not None}")

    # ZYX -> YXZ.
    psf = np.load(nbp_file.psf)["arr_0"].astype(np.float32).swapaxes(0, 2).swapaxes(0, 1)
    if np.max(psf.shape[:2]) < psf.shape[2]:
        log.warn(f"The given PSF has a strange shape of yxz = {psf.shape}")
    # Normalise psf so the min is 0 and the max is 1.
    psf = psf - psf.min()
    psf = psf / psf.max()
    # The PSF is tapered at every edge using a Hanning window.
    psf = (
        psf
        * np.hanning(psf.shape[0]).reshape(-1, 1, 1)
        * np.hanning(psf.shape[1]).reshape(1, -1, 1)
        * np.hanning(psf.shape[2]).reshape(1, 1, -1)
    )
    psf = dimensions.DimensionReducer().reduce(psf)
    nbp_debug.psf = psf

    batch_size: int | None = config["num_cores"]
    if batch_size is None:
        batch_size = max(1, maths.floor(system.get_available_memory() / 27))
    batch_size = min(batch_size, config["max_cores"])
    batch_count: int = maths.ceil(len(indices) / batch_size)

    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    image_reducer = dimensions.DimensionReducer()

    for batch_i in tqdm.trange(batch_count, desc="Filtering extract images", unit="batch"):
        index_min, index_max = batch_i * batch_size, min((batch_i + 1) * batch_size, len(indices))
        batch_images: list[np.np.ndarray] = []
        batch_trcs: list[tuple[int, int, int]] = []

        for t, r, c in indices[index_min:index_max]:
            if config_unchanged and (t, r, c) in completed_indices:
                # Already saved filtered images are not re-filtered.
                continue

            file_path_raw = nbp_file.tile_unfiltered[t][r][c]
            raw_image_exists = zarray.image_exists(file_path_raw)
            if not raw_image_exists:
                raise FileNotFoundError(f"Raw, extracted file at\n\t{file_path_raw}\nnot found")

            with zarr.ZipStore(file_path_raw, mode="r") as raw_store:
                image = zarr.open_array(raw_store)[:]
            image = image.astype(np.float64)

            if channel_radius_norm is not None and c in nbp_basic.use_channels:
                image = radius_normalisation.radius_normalise_image(
                    image, channel_radius_norm[nbp_basic.use_channels.index(c)]
                )
            elif dapi_radius_norm is not None and c == nbp_basic.dapi_channel:
                image = radius_normalisation.radius_normalise_image(image, dapi_radius_norm)

            image = image_reducer.reduce(image)
            batch_images.append(image)
            batch_trcs.append((t, r, c))
            del image

        assert len(batch_images) == len(batch_trcs)

        if len(batch_images) == 0:
            continue

        deconvolution_method = skimage.restoration.wiener
        if not config["use_wiener_deconvolution"]:
            deconvolution_method = lambda im, *_: im

        filtered_images = joblib.Parallel(n_jobs=len(batch_images), return_as="list", timeout=60 * 20)(
            joblib.delayed(deconvolution_method)(batch_images.pop(0), psf, config["wiener_constant"], clip=False)
            for _ in range(len(batch_images))
        )

        for filtered_image, (t, r, c) in zip(filtered_images, batch_trcs, strict=True):
            # All images are deconvolved, including the DAPI.
            filtered_image = filtered_image.astype(FILTER_DTYPE)
            filtered_image = image_reducer.undo(filtered_image)
            images[t, r, c] = filtered_image
            completed_indices["a"].append((t, r, c))
            dict_io.save_dict(completed_indices, completed_indices_path)
            del filtered_image

    os.remove(config_path)
    os.remove(completed_indices_path)
    images_store.close()

    # Following the joblib leak issue at https://github.com/joblib/joblib/issues/945, any remaining process after the
    # use of joblib are explicitly killed.
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        try:
            log.debug("Trying to kill process with pid {}".format(subproc))
            psutil.Process(subproc).terminate()
        except psutil.NoSuchProcess:
            continue

    nbp.images = images
    log.debug("Filter complete")

    return nbp, nbp_debug
